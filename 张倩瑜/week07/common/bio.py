# -*- coding: utf-8 -*-
"""字符级 BIO 标签：联合事件类型 + 触发词/论元角色。"""
from __future__ import annotations

from io_utils import load_schema


def build_label_list(schema=None):
    schema = schema or load_schema()
    labels = ["O"]
    for etype, roles in schema.items():
        labels.append(f"B-TRG:{etype}")
        labels.append(f"I-TRG:{etype}")
        for role in roles:
            labels.append(f"B-ARG:{role}")
            labels.append(f"I-ARG:{role}")
    # 去重保序（角色跨类型共享）
    seen, uniq = set(), []
    for x in labels:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def label2id(labels=None):
    labels = labels or build_label_list()
    return {l: i for i, l in enumerate(labels)}, labels


def spans_to_bio(text: str, event: dict, labels=None) -> list:
    """把单事件转成字符级 BIO；重叠时后写覆盖（模板保证角色少重叠）。"""
    labels = labels or build_label_list()
    lab = ["O"] * len(text)
    if not event:
        return lab

    etype = event["type"]
    trig = event.get("trigger") or {}
    if isinstance(trig, dict) and trig.get("text") is not None:
        s, e = int(trig["start"]), int(trig["end"])
        if 0 <= s < e <= len(text):
            tag_b, tag_i = f"B-TRG:{etype}", f"I-TRG:{etype}"
            if tag_b in labels:
                lab[s] = tag_b
                for i in range(s + 1, e):
                    lab[i] = tag_i

    for a in event.get("args") or []:
        role = a.get("role", "")
        s, e = int(a["start"]), int(a["end"])
        tag_b, tag_i = f"B-ARG:{role}", f"I-ARG:{role}"
        if tag_b not in labels:
            continue
        if 0 <= s < e <= len(text):
            lab[s] = tag_b
            for i in range(s + 1, e):
                lab[i] = tag_i
    return lab


def bio_to_event(text: str, tags: list) -> dict | None:
    """从 BIO 解码出最多一个事件（取第一个触发词类型）。"""
    assert len(text) == len(tags)
    n = len(tags)
    i = 0
    trigger = None
    etype = None
    args = []
    while i < n:
        t = tags[i]
        if t.startswith("B-TRG:"):
            etype = t[len("B-TRG:"):]
            j = i + 1
            while j < n and tags[j] == f"I-TRG:{etype}":
                j += 1
            trigger = {"text": text[i:j], "start": i, "end": j}
            i = j
            continue
        if t.startswith("B-ARG:"):
            role = t[len("B-ARG:"):]
            j = i + 1
            while j < n and tags[j] == f"I-ARG:{role}":
                j += 1
            args.append({"role": role, "text": text[i:j], "start": i, "end": j})
            i = j
            continue
        i += 1

    if not etype or not trigger:
        return None
    return {"type": etype, "trigger": trigger, "args": args}


def make_vocab(texts, min_freq=1):
    from collections import Counter
    cnt = Counter()
    for t in texts:
        cnt.update(t)
    vocab = {"<pad>": 0, "<unk>": 1}
    for ch, c in cnt.most_common():
        if c >= min_freq and ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab
