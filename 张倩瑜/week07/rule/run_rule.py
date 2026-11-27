# -*- coding: utf-8 -*-
"""规则基线（词典+模板）：供 L5 组合与对照。不计入 L1–L5 主表时可作 L0。"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))

from evaluate import evaluate
from io_utils import dump_jsonl, load_schema, load_split, make_pred_row

OUT = ROOT / "outputs"

# 与 build_dataset 对齐的触发词 → 类型
TRIGGER_TYPE = {
    "depends on": "DataDepends",
    "is dependent on": "DataDepends",
    "relies on": "DataDepends",
    "is driven by": "DataDepends",
    "produces": "DataProduces",
    "outputs": "DataProduces",
    "computes": "DataProduces",
    "generates": "DataProduces",
    "maps from": "ReportMaps",
    "is sourced from": "ReportMaps",
    "is populated from": "ReportMaps",
    "traces back to": "ReportMaps",
    "flows from": "SystemFlow",
    "is ingested from": "SystemFlow",
    "is loaded from": "SystemFlow",
    "is transferred from": "SystemFlow",
    "is derived via": "TransformDerives",
    "is transformed by": "TransformDerives",
    "is obtained through": "TransformDerives",
    "is calculated using": "TransformDerives",
}

ROLE_HINTS = {
    "OutputData": ["metric", "measure", "emits", "writes", "materialize", "risk measure"],
    "InputData": ["dataset", "input", "feed", "reads", "consumes", "upstream table", "input series"],
    "Method": ["method", "routine", "function", "code path", "implementation", "job", "assembler",
               "orchestrator", "owner method", "mover", "control point", "implementation hook", "code"],
    "RiskDomain": ["credit_risk", "market_risk", "counterparty_credit", "cva_risk", "liquidity_risk"],
    "Transform": ["applying", "reshaped by", "intermediate", "via", "transformed by", "using"],
    "ReportField": ["cell", "field", "report field", "lineage worksheet"],
    "SourceData": ["from", "source", "sourced from", "populated from"],
    "Dataset": ["Dataset", "dataset"],
    "SourceSystem": ["from", "ingested from", "loaded from"],
    "TargetSystem": ["into", "landing zone", "lands in", "publishes", "to"],
}


def _find_all(text, frag):
    out, start = [], 0
    while True:
        i = text.find(frag, start)
        if i < 0:
            break
        out.append((i, i + len(frag)))
        start = i + 1
    return out


def build_lexicon(train_rows):
    """从训练集收集论元词表（按角色）。"""
    lex = defaultdict(set)
    for r in train_rows:
        ev = r.get("event") or {}
        for a in ev.get("args") or []:
            lex[a["role"]].add(a["text"])
    return {k: sorted(v, key=len, reverse=True) for k, v in lex.items()}


def extract_one(text, lexicon, schema):
    # 最长触发词优先
    trigs = sorted(TRIGGER_TYPE.keys(), key=len, reverse=True)
    found = None
    for t in trigs:
        spans = _find_all(text, t)
        if spans:
            found = (t, spans[0], TRIGGER_TYPE[t])
            break
    if not found:
        return None
    trig_text, (ts, te), etype = found
    roles = schema[etype]
    used = set()
    args = []
    for role in roles:
        cand = lexicon.get(role, [])
        best = None
        for name in cand:
            for s, e in _find_all(text, name):
                if any(not (e <= a or s >= b) for a, b in used):
                    continue
                # 简单启发式：Method 偏好靠近 method/function 等词
                score = e - s
                window = text[max(0, s - 24):s].lower()
                for hint in ROLE_HINTS.get(role, []):
                    if hint.lower() in window or hint.lower() in text[e:e + 24].lower():
                        score += 10
                if best is None or score > best[0]:
                    best = (score, s, e, name)
        if best:
            _, s, e, name = best
            used.add((s, e))
            args.append({"role": role, "text": name, "start": s, "end": e})
    return {
        "type": etype,
        "trigger": {"text": trig_text, "start": ts, "end": te},
        "args": args,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_pred", default=str(OUT / "rule_pred.jsonl"))
    ap.add_argument("--out_metrics", default=str(OUT / "rule_metrics.json"))
    args = ap.parse_args()

    train, val = load_split("train"), load_split("val")
    schema = load_schema()
    lex = build_lexicon(train)
    preds = []
    for r in val:
        ev = extract_one(r["text"], lex, schema)
        preds.append(make_pred_row(r, ev))
    metrics = evaluate(val, preds, verbose=True)
    dump_jsonl(args.out_pred, preds)
    Path(args.out_metrics).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved:", args.out_metrics)


if __name__ == "__main__":
    main()
