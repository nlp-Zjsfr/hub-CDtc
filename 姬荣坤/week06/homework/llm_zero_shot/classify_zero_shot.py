import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DATASET_NAME = "sookiemonster/asrs-narratives-rebalance"
SPLIT = "test"

# 获取项目根目录 (homework/)
# 因为本文件在 homework/llm_zero_shot/ 下，所以 parent.parent 就是 homework
ROOT_DIR = Path(__file__).resolve().parent.parent


def parse_label(text: str, label_names: list) -> int:
    """
    从生成文本里抽出类别名；抽不到返回 -1（记为解析失败）。
    容忍模型输出前后的废话（如 "类别：weather。"），使用包含匹配。
    """
    text = text.strip().lower()
    hits = [name for name in label_names if name.lower() in text]
    if not hits:
        return -1
    # 如果有多个命中，取最长的（最具体的）那个
    return label_names.index(max(hits, key=len))


def evaluate(y_true, y_pred, num_classes):
    # 评估函数与之前的脚本（规则/TF-IDF/BERT）完全一致
    n = len(y_true)
    unparsed = sum(1 for p in y_pred if p is None or p < 0)
    conf = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        if p is not None and p >= 0:
            conf[t][p] += 1
            
    correct = sum(conf[i][i] for i in range(num_classes))
    parsed = n - unparsed
    acc = correct / parsed if parsed else 0.0
    acc_incl = correct / n if n else 0.0
    
    per_recall, f1s, supports = {}, [], []
    for i in range(num_classes):
        tp = conf[i][i]
        row = sum(conf[i])
        col = sum(conf[r][i] for r in range(num_classes))
        recall = tp / row if row else 0.0
        precision = tp / col if col else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_recall[str(i)] = round(recall, 4)
        f1s.append(f1)
        supports.append(row)
        
    macro_f1 = sum(f1s) / num_classes if num_classes else 0.0
    weighted_f1 = sum(f * s for f, s in zip(f1s, supports)) / n if n else 0.0
    
    counts = {}
    for p in y_pred:
        key = str(p) if (p is not None and p >= 0) else "-1"
        counts[key] = counts.get(key, 0) + 1
    used = sum(1 for k, v in counts.items() if k != "-1" and v > 0)
    
    return {
        "n": n,
        "unparsed": unparsed,
        "unparsed_rate": round(unparsed / n, 4) if n else 0.0,
        "acc": round(acc, 4),
        "acc_incl_unparsed": round(acc_incl, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class_recall": per_recall,
        "confusion_matrix": conf,
        "pred_distribution": {"used_classes": used, "counts": counts},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="pretrain_models/Qwen3-0.6B",
                    help="相对 homework 目录的模型路径")
    ap.add_argument("--n", type=int, default=-1, help="评估条数；-1 表示全量测试集")
    ap.add_argument("--max_new_tokens", type=int, default=10, help="限制生成长度")
    ap.add_argument("--max_input_length", type=int, default=2048, help="输入文本最大截断长度，防止显存溢出")
    ap.add_argument("--out", default="outputs/zero_shot_pred.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device}")

    # 定位本地模型路径
    model_path = Path(args.model_name)
    if not model_path.is_absolute():
        model_path = ROOT_DIR / model_path
    if not model_path.exists():
        raise FileNotFoundError(f"未找到模型: {model_path}\n请确认模型是否放在了 {ROOT_DIR / 'pretrain_models'}/ 下")

    print(f"[加载模型] {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    
    # 注意：Qwen 模型只能用 torch_dtype，不能写 dtype
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), 
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        local_files_only=True
    ).to(device).eval()

    print(f"[加载数据] {DATASET_NAME} [{SPLIT}] ...")
    ds = load_dataset(DATASET_NAME)
    test = ds[SPLIT]
    label_names = ds["train"].features["label"].names
    
    n_eval = len(test) if args.n < 0 else min(args.n, len(test))
    test = test.select(range(n_eval))
    
    # 构建分类 Prompt
    label_str = ", ".join(label_names)
    system_prompt = (
        "You are an expert text classification assistant. "
        "Classify the given text into exactly ONE of the following categories:\n"
        f"{label_str}\n\n"
        "Output ONLY the exact category name. Do not include any explanations, punctuation, or other text."
    )

    preds, trues, raws = [], [], []
    t0 = time.time()

    print(f"[推理] 开始 Zero-Shot 预测 (共 {n_eval} 条)...")
    for d in tqdm(test):
        text = d["text"]
        true_label = d["label"]
        
        # 截断超长文本，避免 OOM
        inputs = tokenizer(text, truncation=True, max_length=args.max_input_length, return_tensors="pt")
        truncated_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Text to classify:\n{truncated_text}"}
        ]
        
        # 适配 Qwen 的 Chat 模板，关闭 thinking 过程
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=False 
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,  # 贪婪搜索
                pad_token_id=tokenizer.eos_token_id
            )
            
        # 提取新生成的 token
        gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
        raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        
        pred_idx = parse_label(raw_text, label_names)
        
        preds.append(pred_idx)
        trues.append(true_label)
        raws.append(raw_text)

    dt = time.time() - t0

    report = evaluate(trues, preds, len(label_names))
    report["method"] = "llm_zero_shot"
    report["sec_per_item"] = round(dt / n_eval, 4) if n_eval else 0.0

    meta = {
        "dataset": DATASET_NAME,
        "split": SPLIT,
        "model": str(model_path),
        "n_test": len(test),
        "n_eval": n_eval,
        "max_new_tokens": args.max_new_tokens,
    }

    preds_out = [
        {"text": t, "true": int(yt), "pred": int(yp), "raw": raw}
        for t, yt, yp, raw in zip(test["text"], trues, preds, raws)
    ]

    out = ROOT_DIR / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"meta": meta, "report": report, "preds": preds_out},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n评测 {n_eval} 条 | acc={report['acc']:.4f} | "
          f"macro_f1={report['macro_f1']:.4f} | weighted_f1={report['weighted_f1']:.4f}")
    print(f"未解析数量: {report['unparsed']} ({report['unparsed_rate']:.2%})")
    print(f"覆盖类别: {report['pred_distribution']['used_classes']}/{len(label_names)}")
    print(f"→ 结果写入 {out}")


if __name__ == "__main__":
    main()