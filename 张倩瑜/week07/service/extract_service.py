# -*- coding: utf-8 -*-
"""
L5 系统组合：BERT 主抽 + 规则补论元/类型兜底 + schema 校验。

组合策略（教学口径）：
  1. 优先用 L2 BERT 预测；
  2. 若缺触发词/类型，用规则触发词典兜底；
  3. 对 schema 内缺失角色，用训练集词表+规则补抽；
  4. 校验：论元必须是原文子串、角色必须属于该类型。
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
sys.path.insert(0, str(ROOT / "rule"))

from bio import bio_to_event, build_label_list, label2id  # noqa: E402
from evaluate import evaluate  # noqa: E402
from io_utils import dump_jsonl, load_schema, load_split, make_pred_row  # noqa: E402
from model_paths import bert_path  # noqa: E402
from run_rule import build_lexicon, extract_one  # noqa: E402

OUT = ROOT / "outputs"


class BertLinear(torch.nn.Module):
    def __init__(self, model_dir, num_tags):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_dir)
        self.dropout = torch.nn.Dropout(0.1)
        self.fc = torch.nn.Linear(self.bert.config.hidden_size, num_tags)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.fc(self.dropout(out.last_hidden_state))


TRIGGERS_BY_TYPE = {
    "DataDepends": ["is dependent on", "depends on", "relies on", "is driven by"],
    "DataProduces": ["produces", "outputs", "computes", "generates"],
    "ReportMaps": ["is populated from", "is sourced from", "traces back to", "maps from"],
    "SystemFlow": ["is transferred from", "is ingested from", "is loaded from", "flows from"],
    "TransformDerives": ["is obtained through", "is transformed by", "is calculated using", "is derived via"],
}


def repair_trigger(event, text):
    if not event:
        return None
    etype = event.get("type")
    for t in sorted(TRIGGERS_BY_TYPE.get(etype) or [], key=len, reverse=True):
        if t in text:
            s = text.index(t)
            event = dict(event)
            event["trigger"] = {"text": t, "start": s, "end": s + len(t)}
            break
    return event


def decode_one(logits, offsets, text, id2l, mask):
    pred_ids = logits.argmax(-1)[0].tolist()
    char_tags = ["O"] * len(text)
    for lab_id, (a, b), on in zip(pred_ids, offsets, mask[0].tolist()):
        if not on or a == b:
            continue
        tag = id2l[lab_id]
        if tag.startswith("B-") or tag.startswith("I-"):
            char_tags[a] = tag
            for k in range(a + 1, b):
                char_tags[k] = ("I-" + tag[2:]) if tag.startswith("B-") else tag
    return repair_trigger(bio_to_event(text, char_tags), text)


def validate(event, schema, text):
    if not event:
        return None
    etype = event.get("type")
    if etype not in schema:
        return None
    trig = event.get("trigger") or {}
    ttxt = trig.get("text", "")
    if not ttxt or ttxt not in text:
        return None
    allowed = set(schema[etype])
    args = []
    seen = set()
    for a in event.get("args") or []:
        role, frag = a.get("role"), a.get("text")
        if role not in allowed or not frag or frag not in text:
            continue
        key = (role, frag)
        if key in seen:
            continue
        seen.add(key)
        s = text.index(frag)
        args.append({"role": role, "text": frag, "start": s, "end": s + len(frag)})
    ts = text.index(ttxt)
    return {
        "type": etype,
        "trigger": {"text": ttxt, "start": ts, "end": ts + len(ttxt)},
        "args": args,
    }


def merge(bert_ev, rule_ev, schema, text):
    """BERT 为主，规则补洞。"""
    if bert_ev is None and rule_ev is None:
        return None
    if bert_ev is None:
        return validate(rule_ev, schema, text)
    base = validate(bert_ev, schema, text)
    if base is None:
        return validate(rule_ev, schema, text)
    if rule_ev is None:
        return base
    have = {(a["role"], a["text"]) for a in base["args"]}
    roles_have = {a["role"] for a in base["args"]}
    for a in rule_ev.get("args") or []:
        if a["role"] in roles_have:
            continue
        if a["text"] not in text:
            continue
        key = (a["role"], a["text"])
        if key in have:
            continue
        s = text.index(a["text"])
        base["args"].append({"role": a["role"], "text": a["text"], "start": s, "end": s + len(a["text"])})
        roles_have.add(a["role"])
    return validate(base, schema, text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    schema = load_schema()
    train, val = load_split("train"), load_split("val")
    lex = build_lexicon(train)

    ckpt_path = OUT / "l2_bert.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError("需要先跑 L2: python bert/train.py")

    l2i, labels = label2id()
    id2l = {i: l for l, i in l2i.items()}
    model_dir = str(bert_path())
    tok = AutoTokenizer.from_pretrained(model_dir)
    device = torch.device(args.device)
    model = BertLinear(model_dir, len(labels)).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    preds = []
    for r in val:
        text = r["text"]
        enc = tok(text, return_offsets_mapping=True, return_tensors="pt", truncation=True, max_length=128)
        offsets = enc.pop("offset_mapping")[0].tolist()
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            logits = model(ids, mask)
        bert_ev = decode_one(logits, offsets, text, id2l, mask)
        rule_ev = extract_one(text, lex, schema)
        final = merge(bert_ev, rule_ev, schema, text)
        preds.append(make_pred_row(r, final))

    metrics = evaluate(val, preds, verbose=True)
    dump_jsonl(OUT / "l5_service_pred.jsonl", preds)
    (OUT / "l5_service_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved:", OUT / "l5_service_metrics.json")


if __name__ == "__main__":
    main()
