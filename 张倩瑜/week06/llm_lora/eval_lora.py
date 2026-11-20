"""评测 LoRA adapter（与 train 使用同一 chat 模板）。"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import load_clean_val
from common.labels import BOUNDARY_NOTES, LABELS, parse_label
from common.metrics import classification_report, pred_distribution
from common.model_paths import qwen_path

OUT = Path(__file__).resolve().parent.parent / "outputs"
LABEL_BLOCK = "\n".join(f"- {n}: {BOUNDARY_NOTES[n]}" for n in LABELS)
SYSTEM = (
    "你是宣传手法分类助手。给定英文新闻片段，把它分到下列类别之一。"
    "只输出类别名，必须与列表中的字符串完全一致。\n"
    + LABEL_BLOCK
)


@torch.no_grad()
def classify(model, tok, sentence, device):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": sentence}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=32, do_sample=False, pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return parse_label(gen), gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(OUT / "lora_adapter"))
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(OUT / "lora_pred.json"))
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = str(qwen_path())
    tok = AutoTokenizer.from_pretrained(args.adapter if Path(args.adapter).exists() else model_dir)
    base = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float16 if device == "cuda" else torch.float32
    )
    model = PeftModel.from_pretrained(base, args.adapter).to(device)
    model.eval()

    samples, meta = load_clean_val(n=args.n, seed=args.seed)
    ys, ps, raws = [], [], []
    for i, d in enumerate(samples):
        p, raw = classify(model, tok, d["sentence"], device)
        ys.append(d["label"])
        ps.append(p)
        raws.append(raw)
        if i % 20 == 0:
            print(f"  {i}/{len(samples)} pred={LABELS[p] if p >= 0 else 'UNPARSED'}", flush=True)

    report = classification_report(ys, ps, num_classes=len(LABELS))
    report["meta"] = meta
    report["method"] = "lora"
    report["pred_dist"] = pred_distribution(ps, num_classes=len(LABELS))

    OUT.mkdir(parents=True, exist_ok=True)
    pred = [{"idx": d["idx"], "gold": d["label"], "pred": p, "raw": r, "sentence": d["sentence"]}
            for d, p, r in zip(samples, ps, raws)]
    Path(args.out).write_text(json.dumps(pred, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "lora_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[lora] n={report['n']} acc_incl={report.get('acc_incl_unparsed')} "
          f"macro_f1={report.get('macro_f1')} unparsed={report.get('unparsed')}")


if __name__ == "__main__":
    main()
