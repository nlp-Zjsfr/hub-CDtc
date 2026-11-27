# -*- coding: utf-8 -*-
"""L4 评测：加载 LoRA adapter 生成 JSON 并对齐 span。"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))

from evaluate import evaluate
from io_utils import dump_jsonl, load_schema, load_split, make_pred_row
from model_paths import qwen_path

OUT = ROOT / "outputs"
ADAPTER = OUT / "l4_lora_adapter"


def build_messages(schema, text):
    schema_txt = "\n".join(f"- {k}: {', '.join(v)}" for k, v in schema.items())
    system = (
        "You extract data-lineage events from credit/market risk documentation. "
        "Return ONE JSON object only. "
        '{"type":"...","trigger":"...","args":[{"role":"...","text":"..."}]}\n'
        f"Types:\n{schema_txt}"
    )
    user = f"Sentence:\n{text}\n\nJSON:"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_json(raw: str):
    m = re.search(r"\{.*\}", raw.strip(), flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def align_event(text: str, obj: dict):
    if not obj:
        return None
    etype = str(obj.get("type", "")).strip()
    trig = str(obj.get("trigger", "")).strip()
    if not etype or not trig or trig not in text:
        return None
    ts = text.index(trig)
    args = []
    for a in obj.get("args") or []:
        if not isinstance(a, dict):
            continue
        role = str(a.get("role", "")).strip()
        frag = str(a.get("text", "")).strip()
        if not role or not frag or frag not in text:
            continue
        s = text.index(frag)
        args.append({"role": role, "text": frag, "start": s, "end": s + len(frag)})
    return {
        "type": etype,
        "trigger": {"text": trig, "start": ts, "end": ts + len(trig)},
        "args": args,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    schema = load_schema()
    val = load_split("val")
    if args.n > 0:
        val = val[: args.n]

    base = str(qwen_path())
    tok = AutoTokenizer.from_pretrained(ADAPTER if ADAPTER.exists() else base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.float16 if args.device.startswith("cuda") else torch.float32,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(ADAPTER))
    model.to(args.device)
    model.eval()

    preds, parse_fail = [], 0
    for i, r in enumerate(val):
        messages = build_messages(schema, r["text"])
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(args.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=256, do_sample=False, pad_token_id=tok.eos_token_id
            )
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        obj = parse_json(gen)
        ev = align_event(r["text"], obj) if obj else None
        if ev is None:
            parse_fail += 1
        preds.append(make_pred_row(r, ev))
        if (i + 1) % 20 == 0:
            print(f"done {i+1}/{len(val)} parse_fail={parse_fail}")

    metrics = evaluate(val, preds, verbose=True)
    metrics["parse_fail"] = parse_fail
    dump_jsonl(OUT / "l4_lora_pred.jsonl", preds)
    (OUT / "l4_lora_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("parse_fail:", parse_fail)


if __name__ == "__main__":
    main()
