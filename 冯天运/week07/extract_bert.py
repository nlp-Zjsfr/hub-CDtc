# -*- coding: utf-8 -*-
"""
本地 BERT 小模型抽取（**不需要 API key、不联网**）。

和 extract.py 平级、各司其职（仿 week06 的 train_base_model.py）：
  extract.py       → LLM / API 抽取（zero-shot、few-shot）
  extract_bert.py  → 本地 BERT 微调抽取（序列标注 BERT+Linear）
  compare.py       → 读两边同格式的 log.json，一张表对比

用法：
    python extract_bert.py --dataset duee_fin --model bert-base-chinese --epochs 3
    python extract_bert.py --dataset duee_fin --model chinese-roberta-wwm-ext --epochs 3
    python extract_bert.py --dataset cmeee --model bert-base-chinese --epochs 3
    python extract_bert.py --dataset duee_fin --epochs 1 --n 50 --tag bert_trial   # 快速试跑

输出（与 extract.py 完全一致，compare.py 可直接读）：
    out/<tag>_log.json
    out/<tag>_pred.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from extract import CMEEE_TYPES, evaluate, read_jsonl, write_json

ROOT = Path(__file__).resolve().parent      # 脚本所在的 week7/ 目录

MODEL_DIRS = ["bert-base-chinese", "chinese-roberta-wwm-ext"]


# ── 数据 ─────────────────────────────────────────────────────────────────

def find_processed():
    for p in [ROOT] + list(ROOT.parents):
        for rel in ("week7-信息抽取/code/data/processed", "code/data/processed"):
            cand = p / rel
            if (cand / "train_extract.jsonl").exists():
                return cand
    return None


def find_model(name):
    for p in [ROOT] + list(ROOT.parents):
        cand = p / "pretrain_models" / name
        if (cand / "config.json").exists():
            return cand
    return None


def load_data(dataset, split, n=0):
    """返回 (rows, schema, has_trigger)，统一格式（保留字符偏移供 BIO 标注用）。"""
    if dataset == "cmeee":
        rows = []
        for i, r in enumerate(read_jsonl(ROOT / "data" / "cmeee_mini.jsonl")):
            text = r["text"]
            args = []
            for a in r.get("spans") or []:
                s = text.find(a["text"])
                args.append({"role": a["role"], "text": a["text"],
                             "start": s, "end": s + len(a["text"]) if s >= 0 else -1})
            rows.append({"id": r.get("id", f"cmeee_{i}"), "sent_id": r.get("id", f"cmeee_{i}"),
                         "text": text,
                         "events": [{"type": "实体集合", "trigger": {"text": "", "start": -1, "end": -1},
                                     "args": args}] if args else []})
        return (rows[:n] if n else rows), {t: [] for t in CMEEE_TYPES}, False

    base = find_processed()
    if base is None:
        raise FileNotFoundError("找不到 week7 教学项目的 code/data/processed/")
    schema = json.loads((base / "event_schema.json").read_text(encoding="utf-8"))
    schema = {k: v for k, v in schema.items() if not k.startswith("__")}
    fn = "train_extract.jsonl" if split == "train" else "dev_extract.jsonl"
    rows = []
    for r in read_jsonl(base / fn):
        ev = r.get("event") or {}
        trig = ev.get("trigger") or {}
        args = [{"role": a.get("role", ""), "text": a.get("text", ""),
                 "start": a.get("start", -1), "end": a.get("end", -1)}
                for a in ev.get("args") or [] if a.get("text")]
        tt = trig.get("text", "")
        s = text_find(r.get("text", ""), tt)
        rows.append({"id": r.get("id", ""), "sent_id": r.get("sent_id", ""),
                     "text": r.get("text", ""),
                     "events": [{"type": ev.get("type", ""),
                                 "trigger": {"text": tt, "start": s,
                                             "end": s + len(tt) if s >= 0 else -1},
                                 "args": args}] if ev.get("type") else []})
    return (rows[:n] if n else rows), schema, True


def text_find(text, span):
    return text.find(span) if span else -1


# ── 标签体系（joint BIO）─────────────────────────────────────────────────

def build_labels(schema, has_trigger):
    """事件型：TRG:<类型> + ARG:<角色>；实体型：ENT:<类别>。角色跨类型共享。"""
    tags = set()
    if has_trigger:
        tags.update("TRG:" + t for t in schema)
        for roles in schema.values():
            tags.update("ARG:" + r for r in roles)
    else:
        tags.update("ENT:" + t for t in schema)
    labels = ["O"]
    for t in sorted(tags):
        labels += ["B-" + t, "I-" + t]
    return labels


def span_to_bio(text, events, label2id, has_trigger):
    """字符级 span → BIO id 序列（与 extract.py 的评测口径完全一致）。"""
    n = len(text)
    tags = ["O"] * n
    painted = [False] * n

    def paint(a, b, tag):
        # ⚠ label2id 的 key 是带 BIO 前缀的（B-ARG:xxx / I-ARG:xxx），
        # 传进来的 tag 是裸标签（ARG:xxx）。必须拼上 B- 再查！
        # 写成 `if tag not in label2id` 会永远为 True → 一个标签都刷不上 →
        # 训练目标全是 O，loss 降到接近 0 但 F1 永远是 0（极隐蔽的 bug）。
        if f"B-{tag}" not in label2id or a is None or b is None:
            return
        a, b = int(a), int(b)
        if a < 0 or b < 0 or a >= b or a >= n:
            return
        b = min(b, n)
        if any(painted[a:b]):
            return
        tags[a] = "B-" + tag
        for i in range(a + 1, b):
            tags[i] = "I-" + tag
        for i in range(a, b):
            painted[i] = True

    for e in events:
        if has_trigger:
            trig = e["trigger"]
            if trig["text"]:
                a = trig["start"] if trig["start"] >= 0 else text.find(trig["text"])
                paint(a, trig["end"] if a >= 0 else -1,
                      "TRG:" + e["type"])
        for arg in e["args"]:
            a = arg["start"] if arg["start"] >= 0 else text.find(arg["text"])
            tag = ("ARG:" if has_trigger else "ENT:") + arg["role"]
            paint(a, arg["end"] if a >= 0 else -1, tag)
    return [label2id[t] for t in tags]


class NerDataset(Dataset):
    """字符级 BIO → 子词级标签（只让每个汉字的首个子词背标签，其余 -100）。"""

    def __init__(self, rows, tokenizer, label2id, has_trigger, max_len):
        self.items = []
        for r in rows:
            chars = list(r["text"])[:max_len - 2]
            enc = tokenizer(chars, is_split_into_words=True, truncation=True,
                            max_length=max_len, return_tensors=None)
            wids = enc.word_ids()
            char_ids = span_to_bio("".join(chars), r["events"], label2id, has_trigger)
            labels, prev = [], None
            for w in wids:
                if w is None or w == prev:
                    labels.append(-100)
                else:
                    labels.append(char_ids[w] if w is not None and w < len(char_ids) else -100)
                    prev = w
            self.items.append((enc["input_ids"], enc["attention_mask"], labels))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch):
    m = max(len(x[0]) for x in batch)
    ids = torch.zeros(len(batch), m, dtype=torch.long)
    mask = torch.zeros(len(batch), m, dtype=torch.long)
    labels = torch.full((len(batch), m), -100, dtype=torch.long)
    for i, (a, b, c) in enumerate(batch):
        ids[i, :len(a)] = torch.tensor(a)
        mask[i, :len(b)] = torch.tensor(b)
        labels[i, :len(c)] = torch.tensor(c)
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


# ── 解码 ─────────────────────────────────────────────────────────────────

def decode_bio(tags):
    spans, cur = [], None
    for i, t in enumerate(tags):
        if t == "O":
            if cur:
                spans.append(cur)
                cur = None
        elif t.startswith("B-"):
            if cur:
                spans.append(cur)
            cur = {"tag": t[2:], "start": i, "end": i + 1}
        else:
            tag = t[2:]
            if cur and cur["tag"] == tag:
                cur["end"] = i + 1
            else:
                if cur:
                    spans.append(cur)
                cur = {"tag": tag, "start": i, "end": i + 1}
    if cur:
        spans.append(cur)
    return spans


def spans_to_events(spans, text, has_trigger):
    evs = []
    trig = None
    args = []
    for sp in spans:
        piece = text[sp["start"]:sp["end"]]
        if not piece:
            continue
        tag = sp["tag"]
        if has_trigger:
            if tag.startswith("TRG:"):
                if trig is None:
                    trig = (tag[4:], piece, sp["start"], sp["end"])
            elif tag.startswith("ARG:"):
                args.append({"role": tag[4:], "text": piece})
        elif tag.startswith("ENT:"):
            args.append({"role": tag[4:], "text": piece})
    if has_trigger and trig:
        evs.append({"type": trig[0],
                    "trigger": {"text": trig[1], "start": trig[2], "end": trig[3]},
                    "args": args})
    elif not has_trigger and args:
        evs.append({"type": "实体集合",
                    "trigger": {"text": "", "start": -1, "end": -1},
                    "args": args})
    return evs


# ── 主流程 ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="本地 BERT 小模型抽取（不需要 API）")
    ap.add_argument("--dataset", default="duee_fin", choices=["duee_fin", "cmeee"])
    ap.add_argument("--model", default="bert-base-chinese", choices=MODEL_DIRS)
    ap.add_argument("--epochs", default=3, type=int)
    ap.add_argument("--batch_size", default=8, type=int)
    ap.add_argument("--lr", default=3e-5, type=float)
    ap.add_argument("--max_len", default=256, type=int)
    ap.add_argument("--n", default=0, type=int, help="只取前 N 条（快速试跑）")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    from transformers import AutoModelForTokenClassification, AutoTokenizer

    path = find_model(args.model)
    if path is None:
        raise FileNotFoundError(
            f"找不到本地模型 {args.model}。"
            f"请放到课程根目录 pretrain_models/{args.model}/ 下（config.json 必须在）。")

    train, schema, has_trigger = load_data(args.dataset, "train", args.n)
    dev, _, _ = load_data(args.dataset, "dev", args.n)
    labels = build_labels(schema, has_trigger)
    id2label = {i: l for i, l in enumerate(labels)}
    label2id = {l: i for i, l in id2label.items()}

    tag = args.tag or f"{args.dataset}_{args.model}_bert{args.epochs}ep"
    print(f"[{tag}] train {len(train)} / dev {len(dev)}，标签 {len(labels)} 个")

    tok = AutoTokenizer.from_pretrained(str(path))
    train_ds = NerDataset(train, tok, label2id, has_trigger, args.max_len)
    dev_ds = NerDataset(dev, tok, label2id, has_trigger, args.max_len)
    tr_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dv_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = AutoModelForTokenClassification.from_pretrained(
        str(path), num_labels=len(labels), id2label=id2label, label2id=label2id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optim = AdamW(model.parameters(), lr=args.lr)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        step = 0
        for batch in tr_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            # 新版 transformers 的 forward 返回 tuple，旧版返回带 .loss 的 ModelOutput
            loss = out.loss if hasattr(out, "loss") else out[0]
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += float(loss.item())
            step += 1
            # CPU 上一个 epoch 可能要十几分钟，必须打进度，否则看起来像卡死
            if step % 50 == 0:
                print(f"    batch {step}/{len(tr_loader)}  "
                      f"loss={total / step:.4f}  ({time.time() - t0:.1f}s)", flush=True)
        print(f"  epoch {epoch}/{args.epochs}  loss={total / max(1, len(tr_loader)):.4f} "
              f"({time.time() - t0:.1f}s)", flush=True)

    # 逐条预测（格式与 extract.py 一致，交给同一个 evaluate 算）
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in dv_loader:
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device))
            if hasattr(out, "logits"):
                logits = out.logits
            elif isinstance(out, (tuple, list)):
                # tuple 里找形状为 (B, L, C) 的张量；找不到就取第一个
                cands = [x for x in out
                         if isinstance(x, torch.Tensor) and x.dim() == 3]
                logits = cands[0] if cands else out[0]
            else:
                logits = out
            logits = logits.cpu()
            for i in range(logits.shape[0]):
                # 从 dataset 里拿该条的 word_ids 不方便，这里重新 tokenize 一次（开发集不大）
                r = dev[len(preds)]
                chars = list(r["text"])[:args.max_len - 2]
                enc = tok(chars, is_split_into_words=True, truncation=True,
                          max_length=args.max_len)
                wid = enc.word_ids()
                curr, prev = {}, None
                for j, w in enumerate(wid):
                    if w is None or w == prev or j >= logits.shape[1]:
                        continue
                    curr[w] = id2label[int(logits[i, j].argmax())]
                    prev = w
                tags = [curr.get(k, "O") for k in range(len(chars))]
                spans = decode_bio(tags)
                preds.append({"id": r["id"], "sent_id": r["sent_id"],
                              "events": spans_to_events(spans, r["text"], has_trigger)})

    metrics = evaluate(dev, preds, has_trigger)
    elapsed = time.time() - t0
    log = {
        "tag": tag, "dataset": args.dataset,
        "task_type": "event" if has_trigger else "entity",
        "provider": "local-bert", "model": args.model,
        "mode": f"finetune-{args.epochs}ep", "n": len(dev),
        "metrics": metrics,
        "json_valid": len(dev), "repaired": 0, "parse_fail": 0,
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        "elapsed_s": round(elapsed, 1), "is_mock": False,
        "train_args": vars(args),
    }
    out = ROOT / "out"
    write_json(log, out / f"{tag}_log.json")
    write_json(preds, out / f"{tag}_pred.json")

    e = metrics["entity"]
    print("\n" + "=" * 66)
    print(f"结果：{tag}")
    print("=" * 66)
    print(f"  entity F1  : {e['f1']:.4f}  (P={e['p']:.4f} R={e['r']:.4f})")
    if has_trigger:
        print(f"  trigger F1 : {metrics['trigger']['f1']:.4f}")
        print(f"  event F1   : {metrics['event_exact']['f1']:.4f}")
    print(f"  耗时       : {elapsed:.1f}s")
    print(f"\n结果 → out/{tag}_log.json")


if __name__ == "__main__":
    main()
