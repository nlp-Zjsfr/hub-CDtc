# -*- coding: utf-8 -*-
"""规则基线：由训练集统计触发词词典，结合正则与公司名启发式抽取论元。"""

from __future__ import annotations

import argparse
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from common import (
    DATA_DIR,
    LOG_DIR,
    evaluate,
    load_extract_split,
    load_schema,
    make_pred_row,
    row_event,
    save_predictions,
    write_metrics,
)

MANUAL_TRIGGERS = {
    "股份回购": ["股份回购", "回购股份", "回购"],
    "解除质押": ["解除质押", "解除股份质押", "解押"],
    "质押": ["股权质押", "质押股份", "质押"],
    "股东增持": ["增持股份", "增持"],
    "股东减持": ["减持股份", "减持"],
    "企业收购": ["收购", "并购", "要约收购"],
    "企业融资": ["定增", "增资", "融资"],
    "公司上市": ["上市", "IPO", "首发"],
    "亏损": ["亏损", "预亏", "业绩亏损"],
    "中标": ["中标", "中标通知"],
    "高管变动": ["辞职", "辞去", "聘任", "任命", "换届"],
    "企业破产": ["破产", "破产重整", "破产清算"],
    "被约谈": ["约谈", "被约谈"],
}

NUM = r"[0-9０-９][0-9０-９,，]*\.?[0-9０-９]*"
MONEY_RE = re.compile(
    rf"(?:{NUM}\s*(?:亿|万|千)?\s*(?:元|美元|港元|港币)?\s*(?:至|-|~)\s*"
    rf"{NUM}\s*(?:亿|万|千)?\s*(?:元|美元|港元|港币)"
    rf"|{NUM}\s*(?:亿|万|千)?\s*(?:元|美元|港元|港币|人民币|欧元))"
)
DATE_RE = re.compile(
    r"(?:\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日"
    r"|\d{4}\s*年\s*\d{1,2}\s*月"
    r"|\d{4}\s*年(?:\s*(?:上半年|下半年|第[一二三四]季度|一季度|二季度|三季度|四季度|全年|年报|半年报|三季报))?"
    r"|\d{1,2}\s*月\s*\d{1,2}\s*日|\d{1,2}\s*日)"
)
PERCENT_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:%|％)")
SHARE_RE = re.compile(rf"{NUM}\s*(?:亿|万|千)?\s*股")
COMPANY_RE = re.compile(
    r"[\u4e00-\u9fa5A-Za-z0-9（）()]{2,12}?"
    r"(?:股份有限公司|有限责任公司|有限公司|集团|控股|股份|证券|信托|基金|银行|交易所|公司)"
)

DATE_KEYS = ("时间", "日期", "周期")
MONEY_KEYS = ("金额", "价格", "亏损", "市值", "募资", "规模")
PERCENT_KEYS = ("比例", "占比", "总股本", "所持比例")
SHARE_KEYS = ("数量",)
ENUM_KEYS = ("类型", "环节")
ORG_ROLES = {
    "股票简称", "上市公司", "公司名称", "被收购方", "收购方", "中标公司",
    "破产公司", "任职公司", "被投资方", "投资方", "领投方", "招标方",
    "质押物所属公司", "披露方", "变动后公司名称",
}


def role_to_kind(role: str) -> str:
    """按角色名称中的关键词选择对应抽取器。"""
    if any(key in role for key in DATE_KEYS):
        return "date"
    if any(key in role for key in PERCENT_KEYS):
        return "percent"
    if any(key in role for key in MONEY_KEYS):
        return "money"
    if any(key in role for key in SHARE_KEYS):
        return "share"
    if any(key in role for key in ENUM_KEYS):
        return "enum"
    return "entity"


def build_trigger_dict(train_rows: list[dict], min_count: int) -> dict:
    counts = defaultdict(Counter)
    for row in train_rows:
        event = row_event(row)
        if not event:
            continue
        trigger = (event.get("trigger") or {}).get("text", "")
        if trigger:
            counts[event["type"]][trigger] += 1
    return {
        event_type: [(word, freq) for word, freq in counter.most_common() if freq >= min_count]
        for event_type, counter in counts.items()
    }


def merge_trigger_dict(auto: dict, manual: dict) -> list[tuple[str, str]]:
    """合并自动统计与少量人工同义词，并按词长降序排列，避免短触发词抢配长触发词。"""
    pairs = []
    for event_type, words in auto.items():
        for word, _ in words:
            pairs.append((word, event_type))
    for event_type, words in manual.items():
        for word in words:
            pairs.append((word, event_type))
    return sorted(set(pairs), key=lambda item: (-len(item[0]), item[1]))


def build_company_dict(train_rows: list[dict]) -> set[str]:
    names = set()
    for row in train_rows:
        event = row_event(row)
        if not event:
            continue
        for arg in event.get("args") or []:
            if arg.get("role") not in ORG_ROLES:
                continue
            text = (arg.get("text") or "").strip()
            if 2 <= len(text) <= 10 and not re.search(r"\d", text):
                names.add(text)
    return names


def find_trigger(text: str, trig_pairs: list[tuple[str, str]]):
    best = None
    for word, event_type in trig_pairs:
        pos = text.find(word)
        if pos < 0:
            continue
        candidate = (len(word), -pos)
        if best is None or candidate > best[0]:
            best = (candidate, event_type, word, pos, pos + len(word))
    if best is None:
        return None
    _, event_type, word, start, end = best
    return event_type, word, start, end


def extract_entities(text: str, company_dict: set[str]) -> list[dict]:
    found = []
    for name in company_dict:
        start = 0
        while True:
            pos = text.find(name, start)
            if pos < 0:
                break
            found.append({"text": name, "start": pos, "end": pos + len(name)})
            start = pos + 1
    for match in COMPANY_RE.finditer(text):
        found.append({"text": match.group(0), "start": match.start(), "end": match.end()})
    unique, seen = [], set()
    for cand in sorted(found, key=lambda item: (item["start"], -len(item["text"]))):
        key = (cand["start"], cand["end"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(cand)
    out = []
    for cand in unique:
        covered = any(
            other is not cand
            and other["start"] <= cand["start"]
            and cand["end"] <= other["end"]
            and len(other["text"]) > len(cand["text"])
            for other in unique
        )
        if not covered:
            out.append(cand)
    return out


def extract_args(text: str, role: str, used_spans: list[tuple[int, int]],
                 entity_pool: list[dict]) -> list[dict]:
    """每个角色最多领取一个未占用片段，以控制规则系统的假阳性。"""
    kind = role_to_kind(role)
    out = []

    def free(start: int, end: int) -> bool:
        return all(not (start < used_end and end > used_start) for used_start, used_end in used_spans)

    if kind in ("money", "share", "percent", "date"):
        pattern = {
            "money": MONEY_RE,
            "share": SHARE_RE,
            "percent": PERCENT_RE,
            "date": DATE_RE,
        }[kind]
        for match in pattern.finditer(text):
            start, end = match.start(), match.end()
            if not free(start, end):
                continue
            out.append({"role": role, "text": text[start:end].strip(), "start": start, "end": end})
            break
        return out
    if kind == "enum":
        return []
    for index, cand in enumerate(entity_pool):
        if free(cand["start"], cand["end"]):
            out.append({
                "role": role,
                "text": cand["text"],
                "start": cand["start"],
                "end": cand["end"],
            })
            entity_pool.pop(index)
            break
    return out


def extract_one(text: str, schema: dict, trig_pairs: list[tuple[str, str]],
                company_dict: set[str]) -> dict | None:
    hit = find_trigger(text, trig_pairs)
    if hit is None:
        return None
    event_type, word, start, end = hit
    used = [(start, end)]
    entity_pool = extract_entities(text, company_dict)
    args = []
    for role in schema.get(event_type, []):
        got = extract_args(text, role, used, entity_pool)
        for arg in got:
            used.append((arg["start"], arg["end"]))
        args.extend(got)
    return {
        "type": event_type,
        "trigger": {"text": word, "start": start, "end": end},
        "args": args,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="规则基线事件抽取")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--min-trigger-count", type=int, default=2)
    parser.add_argument("--max-eval", type=int, default=0, help="仅评测前 N 条，0 表示全部")
    parser.add_argument("--run-name", default="rule")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    schema = load_schema(data_dir / "event_schema.json")
    train_rows = load_extract_split("train", data_dir)
    dev_rows = load_extract_split("dev", data_dir)
    if args.max_eval > 0:
        dev_rows = dev_rows[:args.max_eval]

    auto = build_trigger_dict(train_rows, min_count=args.min_trigger_count)
    trig_pairs = merge_trigger_dict(auto, MANUAL_TRIGGERS)
    company_dict = build_company_dict(train_rows)
    print(f"触发词条目 {len(trig_pairs)}，公司名词典 {len(company_dict)} 个")

    started = time.perf_counter()
    preds = []
    for row in dev_rows:
        event = extract_one(row["text"], schema, trig_pairs, company_dict)
        preds.append(make_pred_row(row, [event] if event else []))
    elapsed = time.perf_counter() - started
    save_predictions(args.run_name, preds)
    result = evaluate(dev_rows, preds)
    meta = {
        "method": "rule",
        "train_s": 0.0,
        "ms_per_sentence": round(elapsed * 1000 / max(len(dev_rows), 1), 3),
        "n_triggers": len(trig_pairs),
        "n_companies": len(company_dict),
        "num_eval": len(dev_rows),
    }
    write_metrics(args.run_name, result, meta)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / f"{args.run_name}_eval.json").write_text(
        __import__("json").dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
