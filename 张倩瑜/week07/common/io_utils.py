# -*- coding: utf-8 -*-
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
SCHEMA_PATH = DATA_DIR / "schema.json"


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dump_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_schema(path=None):
    p = Path(path) if path else SCHEMA_PATH
    return json.loads(p.read_text(encoding="utf-8"))


def load_split(name: str):
    """name: train | val"""
    return load_jsonl(DATA_DIR / f"{name}.jsonl")


def row_key(row):
    if row.get("uid"):
        return str(row["uid"])
    if row.get("sent_id"):
        return f'{row.get("id", "")}|{row["sent_id"]}'
    return str(row.get("id", ""))


def as_event_list(row):
    if "events" in row and row["events"] is not None:
        evs = row["events"]
    elif "event" in row and row["event"]:
        evs = [row["event"]]
    else:
        evs = []
    out = []
    for e in evs:
        if not isinstance(e, dict):
            continue
        trig = e.get("trigger") or {}
        trig_text = (trig.get("text") if isinstance(trig, dict) else str(trig)) or ""
        args = []
        for a in e.get("args") or e.get("arguments") or []:
            if not isinstance(a, dict):
                continue
            text = a.get("text", a.get("argument", "")) or ""
            role = a.get("role", "") or ""
            if text:
                args.append((str(role).strip(), str(text).strip()))
        out.append({
            "type": str(e.get("type", e.get("event_type", ""))).strip(),
            "trigger": str(trig_text).strip(),
            "args": args,
            "raw": e,
        })
    return out


def make_pred_row(gold_row, event_or_none):
    """把模型输出整理成评测用预测行。"""
    row = {
        "id": gold_row.get("id"),
        "sent_id": gold_row.get("sent_id"),
        "uid": gold_row.get("uid") or row_key(gold_row),
        "text": gold_row.get("text", ""),
    }
    if event_or_none:
        row["events"] = [event_or_none]
    else:
        row["events"] = []
    return row
