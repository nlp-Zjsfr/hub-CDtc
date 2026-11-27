# -*- coding: utf-8 -*-
"""L2: BERT token classification（联合 BIO）数据血缘抽取。"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))

from bio import bio_to_event, build_label_list, label2id, spans_to_bio
from evaluate import evaluate
from io_utils import dump_jsonl, load_split, make_pred_row
from model_paths import bert_path

OUT = ROOT / "outputs"


class BertIEDataset(Dataset):
    def __init__(self, rows, tokenizer, l2i, max_len=128):
        self.rows = rows
        self.tok = tokenizer
        self.l2i = l2i
        self.max_len = max_len
        self.labels = build_label_list()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        text = r["text"]
        char_tags = spans_to_bio(text, r.get("event"), self.labels)
        enc = self.tok(
            text,
            truncation=True,
            max_length=self.max_len,
            return_offsets_mapping=True,
            return_tensors=None,
        )
        # 子词标签：仅首子词保留 B/I，其余 -100
        y = []
        for (a, b) in enc["offset_mapping"]:
            if a == b:
                y.append(-100)
            else:
                y.append(self.l2i[char_tags[a]])
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": y,
            "offsets": enc["offset_mapping"],
            "row": r,
            "text": text,
        }


def collate(batch, pad_id):
    max_len = max(len(b["input_ids"]) for b in batch)
    B = len(batch)
    ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    mask = torch.zeros(B, max_len, dtype=torch.long)
    labels = torch.full((B, max_len), -100, dtype=torch.long)
    offsets, rows, texts = [], [], []
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        mask[i, :n] = torch.tensor(b["attention_mask"])
        labels[i, :n] = torch.tensor(b["labels"])
        offsets.append(b["offsets"])
        rows.append(b["row"])
        texts.append(b["text"])
    return ids, mask, labels, offsets, rows, texts


class BertLinear(nn.Module):
    def __init__(self, model_dir, num_tags):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_dir)
        h = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(h, num_tags)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.fc(self.dropout(out.last_hidden_state))


# 多词触发词（"is sourced from"）在子词对齐后常被截成 "is"；用类型词典回填最长匹配
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
    cands = TRIGGERS_BY_TYPE.get(etype) or []
    best = None
    for t in sorted(cands, key=len, reverse=True):
        if t in text:
            best = t
            break
    if best:
        s = text.index(best)
        event = dict(event)
        event["trigger"] = {"text": best, "start": s, "end": s + len(best)}
    return event


def decode_batch(logits, offsets, texts, id2l, mask):
    pred_ids = logits.argmax(-1)
    events = []
    for i, text in enumerate(texts):
        char_tags = ["O"] * len(text)
        seq = pred_ids[i].tolist()
        m = mask[i].tolist()
        for j, (lab_id, (a, b), on) in enumerate(zip(seq, offsets[i], m)):
            if not on or a == b or lab_id < 0:
                continue
            tag = id2l[lab_id]
            # 只写首字符位置，I 标签填满 offset 区间
            if tag.startswith("B-") or tag.startswith("I-"):
                char_tags[a] = tag
                for k in range(a + 1, b):
                    if tag.startswith("B-"):
                        char_tags[k] = "I-" + tag[2:]
                    else:
                        char_tags[k] = tag
        events.append(repair_trigger(bio_to_event(text, char_tags), text))
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    model_dir = str(bert_path())
    tok = AutoTokenizer.from_pretrained(model_dir)
    l2i, labels = label2id()
    id2l = {i: l for l, i in l2i.items()}

    train, val = load_split("train"), load_split("val")
    tr_ds = BertIEDataset(train, tok, l2i, args.max_len)
    va_ds = BertIEDataset(val, tok, l2i, args.max_len)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                           collate_fn=lambda b: collate(b, tok.pad_token_id))
    va_loader = DataLoader(va_ds, batch_size=args.batch, shuffle=False,
                           collate_fn=lambda b: collate(b, tok.pad_token_id))

    device = torch.device(args.device)
    model = BertLinear(model_dir, len(labels)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss(ignore_index=-100)

    best_f1, best_state = -1, None
    for ep in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for ids, mask, y, _, _, _ in tr_loader:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            logits = model(ids, mask)
            loss = crit(logits.view(-1, logits.size(-1)), y.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

        model.eval()
        preds = []
        with torch.no_grad():
            for ids, mask, y, offsets, rows, texts in va_loader:
                ids, mask = ids.to(device), mask.to(device)
                logits = model(ids, mask)
                evs = decode_batch(logits, offsets, texts, id2l, mask)
                for r, ev in zip(rows, evs):
                    preds.append(make_pred_row(r, ev))
        metrics = evaluate(val, preds, verbose=False)
        f1 = metrics["role_macro_f1"]
        print(f"epoch {ep} loss={total/len(tr_loader):.4f} role_macro_f1={f1:.4f} "
              f"arg_f1={metrics['argument']['f1']:.4f} type_macro={metrics['type_macro_f1']:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.no_grad():
        for ids, mask, y, offsets, rows, texts in va_loader:
            ids, mask = ids.to(device), mask.to(device)
            logits = model(ids, mask)
            evs = decode_batch(logits, offsets, texts, id2l, mask)
            for r, ev in zip(rows, evs):
                preds.append(make_pred_row(r, ev))
    metrics = evaluate(val, preds, verbose=True)
    OUT.mkdir(parents=True, exist_ok=True)
    dump_jsonl(OUT / "l2_bert_pred.jsonl", preds)
    (OUT / "l2_bert_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save({"model": best_state, "labels": labels}, OUT / "l2_bert.pt")
    print("saved:", OUT / "l2_bert_metrics.json")


if __name__ == "__main__":
    main()
