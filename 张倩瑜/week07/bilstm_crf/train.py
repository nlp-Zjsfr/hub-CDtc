# -*- coding: utf-8 -*-
"""L1: 字符级 BiLSTM-CRF 数据血缘抽取。"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))

from bio import bio_to_event, build_label_list, label2id, make_vocab, spans_to_bio
from crf import CRF
from evaluate import evaluate
from io_utils import dump_jsonl, load_split, make_pred_row

OUT = ROOT / "outputs"


class CharDataset(Dataset):
    def __init__(self, rows, vocab, l2i, max_len=256):
        self.rows = rows
        self.vocab = vocab
        self.l2i = l2i
        self.max_len = max_len
        self.labels = build_label_list()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        text = r["text"][: self.max_len]
        tags = spans_to_bio(text, r.get("event"), self.labels)
        ids = [self.vocab.get(ch, 1) for ch in text]
        y = [self.l2i[t] for t in tags]
        return ids, y, r, text


def collate(batch):
    ids, ys, rows, texts = zip(*batch)
    L = max(len(x) for x in ids)
    B = len(ids)
    x = torch.zeros(B, L, dtype=torch.long)
    y = torch.zeros(B, L, dtype=torch.long)
    m = torch.zeros(B, L, dtype=torch.bool)
    for i, (a, b) in enumerate(zip(ids, ys)):
        n = len(a)
        x[i, :n] = torch.tensor(a)
        y[i, :n] = torch.tensor(b)
        m[i, :n] = True
    return x, y, m, rows, texts


class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, num_tags, emb=128, hidden=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden * 2, num_tags)
        self.crf = CRF(num_tags)

    def emissions(self, x, mask):
        h, _ = self.lstm(self.dropout(self.emb(x)))
        return self.fc(self.dropout(h))

    def loss(self, x, y, mask):
        em = self.emissions(x, mask)
        return self.crf.neg_log_likelihood(em, y, mask)

    def decode(self, x, mask):
        em = self.emissions(x, mask)
        return self.crf.decode(em, mask)


def run_eval(model, loader, device, id2l):
    model.eval()
    preds = []
    with torch.no_grad():
        for x, y, m, rows, texts in loader:
            x, m = x.to(device), m.to(device)
            paths = model.decode(x, m)
            for i, r in enumerate(rows):
                n = int(m[i].sum().item())
                tags = [id2l[int(t)] for t in paths[i, :n].tolist()]
                ev = bio_to_event(texts[i], tags)
                preds.append(make_pred_row(r, ev))
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    train = load_split("train")
    val = load_split("val")
    vocab = make_vocab([r["text"] for r in train])
    l2i, labels = label2id()
    id2l = {i: l for l, i in l2i.items()}

    tr_loader = DataLoader(CharDataset(train, vocab, l2i, args.max_len),
                           batch_size=args.batch, shuffle=True, collate_fn=collate)
    va_loader = DataLoader(CharDataset(val, vocab, l2i, args.max_len),
                           batch_size=args.batch, shuffle=False, collate_fn=collate)

    device = torch.device(args.device)
    model = BiLSTMCRF(len(vocab), len(labels)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_f1, best_state = -1, None
    for ep in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y, m, _, _ in tr_loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            loss = model.loss(x, y, m)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        preds = run_eval(model, va_loader, device, id2l)
        metrics = evaluate(val, preds, verbose=False)
        f1 = metrics["role_macro_f1"]
        print(f"epoch {ep} loss={total/len(tr_loader):.4f} role_macro_f1={f1:.4f} "
              f"arg_f1={metrics['argument']['f1']:.4f} type_macro={metrics['type_macro_f1']:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    preds = run_eval(model, va_loader, device, id2l)
    metrics = evaluate(val, preds, verbose=True)
    OUT.mkdir(parents=True, exist_ok=True)
    dump_jsonl(OUT / "l1_bilstm_pred.jsonl", preds)
    (OUT / "l1_bilstm_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    ckpt = {"model": best_state, "vocab": vocab, "labels": labels}
    torch.save(ckpt, OUT / "l1_bilstm_crf.pt")
    print("saved:", OUT / "l1_bilstm_metrics.json")


if __name__ == "__main__":
    main()
