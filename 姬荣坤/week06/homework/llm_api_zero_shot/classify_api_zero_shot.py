import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import json
import time
from pathlib import Path
import requests
from datasets import load_dataset
from tqdm import tqdm

DATASET_NAME = "sookiemonster/asrs-narratives-rebalance"
SPLIT = "test"

ROOT_DIR = Path(__file__).resolve().parent.parent


def parse_label(text: str, label_names: list) -> int:
    """从生成文本里抽出类别名；抽不到返回 -1"""
    if not text:
        return -1
    text = text.strip().lower()
    hits = [name for name in label_names if name.lower() in text]
    if not hits:
        return -1
    return label_names.index(max(hits, key=len))


def evaluate(y_true, y_pred, num_classes):
    """评估函数与之前的脚本完全一致"""
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
        "n": n, "unparsed": unparsed, "unparsed_rate": round(unparsed / n, 4) if n else 0.0,
        "acc": round(acc, 4), "acc_incl_unparsed": round(acc_incl, 4),
        "macro_f1": round(macro_f1, 4), "weighted_f1": round(weighted_f1, 4),
        "per_class_recall": per_recall, "confusion_matrix": conf,
        "pred_distribution": {"used_classes": used, "counts": counts},
    }


def call_deepseek_api(api_key, text, system_prompt, model_name, max_retries=3):
    url = "https://api.deepseek.com/anthropic/v1/messages"

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }

    payload = {
        "model": model_name,
        "max_tokens": 512,  # 🌟 关键修改 1：给足 max_tokens，防止思考过程把长度耗尽
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": f"Text to classify:\n{text}"}
        ]
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                res_json = response.json()

                # 🌟 关键修改 2：遍历所有 content 块，只提取 type 为 text 的内容
                if "content" in res_json and isinstance(res_json["content"], list):
                    for block in res_json["content"]:
                        if block.get("type") == "text" and "text" in block:
                            return block["text"].strip()
                            
                # 如果找了一圈没找到 text，可能是它纯粹没生成，或者被安全拦截
                print(f"\n[解析未命中] 原始返回: {json.dumps(res_json, ensure_ascii=False)}")
                return ""

            elif response.status_code == 429:
                time.sleep(2 ** attempt)
            else:
                print(f"\n[API 错误] 状态码 {response.status_code}: {response.text}")
                if response.status_code in [400, 401, 403]:
                    return ""

        except Exception as e:
            print(f"\n[网络或解析异常] {type(e).__name__}: {e}")
            time.sleep(1)

    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api_key", default=os.environ.get("DEEPSEEK_API_KEY", ""),
                    help="DeepSeek API Key。如果不传，默认从环境变量 DEEPSEEK_API_KEY 读取")
    ap.add_argument("--n", type=int, default=-1, help="评估条数；-1 表示全量测试集")
    ap.add_argument("--model_name", default="deepseek-flash",
                    help="模型名称，官方支持 deepseek-flash 或 deepseek-v4-pro")
    ap.add_argument("--out", default="outputs/zero_shot_api_pred.json")
    args = ap.parse_args()

    if not args.api_key:
        raise ValueError("❌ 未提供 API Key！请设置环境变量 DEEPSEEK_API_KEY 或通过 --api_key 传入。")

    print(f"[加载数据] {DATASET_NAME} [{SPLIT}] ...")
    ds = load_dataset(DATASET_NAME)
    test = ds[SPLIT]
    label_names = ds["train"].features["label"].names

    n_eval = len(test) if args.n < 0 else min(args.n, len(test))
    test = test.select(range(n_eval))

    label_str = ", ".join(label_names)
    system_prompt = (
        "You are an expert text classification assistant. "
        "Classify the given text into exactly ONE of the following categories:\n"
        f"{label_str}\n\n"
        "Output ONLY the exact category name. Do not include any explanations, punctuation, or other text."
    )

    preds, trues, raws = [], [], []
    t0 = time.time()

    print(f"[推理] 开始调用 API Zero-Shot 预测 (共 {n_eval} 条) | 模型: {args.model_name} ...")
    for d in tqdm(test):
        text = d["text"]
        true_label = d["label"]

        # API 通常有长度限制，截断防止报错
        truncated_text = text[:3000]

        raw_text = call_deepseek_api(
            api_key=args.api_key,
            text=truncated_text,
            system_prompt=system_prompt,
            model_name=args.model_name
        )

        pred_idx = parse_label(raw_text, label_names)

        preds.append(pred_idx)
        trues.append(true_label)
        raws.append(raw_text)

        # 稍微休眠一下，避免触发并发限流
        time.sleep(0.5)

    dt = time.time() - t0

    report = evaluate(trues, preds, len(label_names))
    report["method"] = "llm_api_zero_shot"
    report["sec_per_item"] = round(dt / n_eval, 4) if n_eval else 0.0

    meta = {
        "dataset": DATASET_NAME,
        "split": SPLIT,
        "model": args.model_name,
        "api_url": "https://api.deepseek.com/anthropic",
        "n_test": len(test),
        "n_eval": n_eval,
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

