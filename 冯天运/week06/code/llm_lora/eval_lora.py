"""
方法五（推理）：加载 LoRA adapter，在干净验证集上评测

【这个脚本在干什么】
把训练好的 LoRA adapter（或全量权重）挂回 Qwen3，在干净验证集上跑生成式评测，
输出与其它方法同口径的指标。

教学重点：
  1. 推理必须和训练用**同一个 chat 模板 + 同一个 enable_thinking=False**，
     否则格式错位，准确率会莫名掉一半。
  2. 对比 three-way：zero-shot 0.098 → LoRA SFT 0.484 → BERT 微调 0.534。
     SFT 把塌缩修好了，但在「封闭标签 + 短文本」上仍不如判别式——
     这就是选型的答案：不是谁更强，是谁更合适。
  3. 解析失败率要单独统计（-1 的条数）。生成式模型总会有不听话的时候，
     这个数字直接决定你要不要加约束解码。

用法：
  python llm_lora/eval_lora.py                       # 默认 outputs/lora_adapter
  python llm_lora/eval_lora.py --n 500 --adapter outputs/lora_adapter
"""

import argparse                        # 命令行参数
import json                           # 读写 JSON
import sys                            # 模块搜索路径
from pathlib import Path             # 路径处理

import torch                          # PyTorch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ★把 code/ 加进 sys.path

from common.data import load_clean_val                       # 统一评测集
from common.labels import LABELS                            # 15 类
from common.metrics import classification_report, pred_distribution, to_markdown_table
from common.model_paths import qwen3_path                    # 本地 Qwen3 目录

CODE_ROOT = Path(__file__).resolve().parent.parent

SYSTEM = ("你是一个新闻标题分类助手。请将给定的新闻标题分类到以下类别之一，"
          "只输出类别名称，不要输出任何其他内容。\n可选类别：" + "、".join(LABELS))
# ↑ ★必须与 train_lora.py 的 SYSTEM 逐字一致，否则训练/推理分布错位


def parse_label(text: str) -> int:
    """从生成文本里抽类别名；抽不到返回 -1（解析失败）"""
    hits = [n for n in LABELS if n in text]     # 哪些类别名出现了
    return LABELS.index(max(hits, key=len)) if hits else -1   # 取最长匹配；没有 → -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(CODE_ROOT / "outputs" / "lora_adapter"))  # 权重目录
    ap.add_argument("--n", type=int, default=500)     # 评测条数
    ap.add_argument("--max_new_tokens", type=int, default=8)   # 生成上限（只要一个类别名）
    ap.add_argument("--out", default="outputs/lora_pred.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(qwen3_path()))
    ckpt_dir = Path(args.adapter)
    if (ckpt_dir / "adapter_config.json").exists():
        # LoRA：先载基座，再挂 adapter
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(str(qwen3_path()), dtype=torch.float16 if device == "cuda" else torch.float32)
        model = PeftModel.from_pretrained(base, str(ckpt_dir)).to(device).eval()
        kind = "lora"
    else:
        # 全量微调：目录里就是完整权重，直接 from_pretrained（约 1.2 GB）
        model = AutoModelForCausalLM.from_pretrained(str(ckpt_dir), dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
        kind = "full_ft"
    # ↑ 用「有没有 adapter_config.json」自动区分两种权重形态，同一个脚本评两种模型
    print(f"[eval] 加载方式：{kind}（{ckpt_dir}）", flush=True)

    samples, meta = load_clean_val(args.n)     # 与其它方法同一批干净样本
    preds, raws = [], []
    for i, d in enumerate(samples):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": d["sentence"]}]
        # ★模板与训练完全一致（含 enable_thinking=False），否则输出格式会乱
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.eos_token_id)  # 贪心解码，结果可复现
        raw = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)  # 只解码新生成部分
        raws.append(raw)                       # 原始输出留着（查解析失败样本）
        preds.append(parse_label(raw))         # 解析成类别 id
        if i % 50 == 0:
            print(f"  {i}/{len(samples)}", flush=True)   # 进度

    y_true = [d["label"] for d in samples]
    report = classification_report(y_true, preds)   # 统一口径（-1 计入 acc_incl_unparsed）
    report["method"] = "qwen3_lora_sft"
    report["pred_distribution"] = pred_distribution(preds)   # SFT 后应该覆盖更多类别（塌缩被修好了）

    out = CODE_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "report": report,
                               "preds": [{"text": d["sentence"], "true": d["label"], "pred": p, "raw": r}
                                         for d, p, r in zip(samples, preds, raws)]},
                              ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"评测 {meta['n_eval']} 条（干净集）")
    print(to_markdown_table({f"Qwen3 + {kind}": report}))
    print(f"预测覆盖类别数：{report['pred_distribution']['used_classes']}/{len(LABELS)}")
    print(f"→ 结果写入 {out}")


if __name__ == "__main__":
    main()
