# coding=utf-8
"""
BiLSTM 序列标注做 NER（conllpp，4 列 CoNLL 格式，只预测第 4 列）

与 bert/train.py 的关键差异：
  * 输入单位 = 数据集切好的词（转小写），**不用 BERT tokenizer**，因此没有子词对齐问题；
  * 词表由训练集构建（--min_freq），表外词 -> <UNK>；
    · min_freq=2 时，训练集里"只出现一次"的词会被当 UNK（约 5% 的 token）；
    · min_freq=1 时训练集 UNK=0，但 dev/test 里训练集没见过的词仍然是 UNK（约 7.5%）。
  * max_len 按"词数"：full=128（最长句 113 词），len_half=64。

用法：
  python lstm/train.py                        # full：BiLSTM + dropout + CRF
  python lstm/train.py --ablation no_crf      # 消融1：取消 CRF
  python lstm/train.py --ablation crf_hard    # 消融2：CRF 硬约束
  python lstm/train.py --ablation len_half    # 消融3：max_len 减半
  python lstm/train.py --all_ablations        # 顺序跑完 4 组并打印对比表
产物：output/lstm/model/<run_name>.pt、output/lstm/log/{metrics,log,console}_*
"""

import argparse
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

HOMEWORK_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = HOMEWORK_DIR / "data" / "conllpp"
MODEL_DIR = HOMEWORK_DIR / "output" / "lstm" / "model"
LOG_DIR = HOMEWORK_DIR / "output" / "lstm" / "log"

LABELS = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC"]
LABEL2ID = {t: i for i, t in enumerate(LABELS)}
ID2LABEL = {i: t for t, i in LABEL2ID.items()}
IGNORE_INDEX = -100
PAD, UNK = "<PAD>", "<UNK>"
NEG_INF = -1e4

os.environ.setdefault("HF_HUB_OFFLINE", "1")


def read_conll(path: Path) -> list[dict]:
    rows, tokens, tags = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if tokens:
                    rows.append({"words": tokens, "tags": tags})
                    tokens, tags = [], []
                continue
            parts = line.split()
            if len(parts) != 4 or parts[0] == "-DOCSTART-":
                continue
            tokens.append(parts[0])
            tags.append(parts[3])
    if tokens:
        rows.append({"words": tokens, "tags": tags})
    return rows


def build_vocab(rows: list[dict], min_freq: int) -> dict:
    """词表：训练集词（小写）频次 >= min_freq。标点也保留（它是重要上下文）。"""
    cnt = Counter(w.lower() for r in rows for w in r["words"])
    words = sorted(w for w, c in cnt.items() if c >= min_freq)
    return {PAD: 0, UNK: 1, **{w: i + 2 for i, w in enumerate(words)}}


class NERDataset(Dataset):
    def __init__(self, rows, vocab, max_len: int, with_labels: bool = True):
        self.rows, self.vocab, self.max_len, self.with_labels = rows, vocab, max_len, with_labels
        self.n_truncated = sum(1 for r in rows if len(r["words"]) > max_len)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        words = row["words"][: self.max_len]
        ids = [self.vocab.get(w.lower(), self.vocab[UNK]) for w in words]
        item = {"input_ids": torch.tensor(ids, dtype=torch.long)}
        if self.with_labels:
            item["labels"] = torch.tensor([LABEL2ID[t] for t in row["tags"][: self.max_len]],
                                          dtype=torch.long)
        return item


def collate_fn(batch):
    max_len = max(int(b["input_ids"].numel()) for b in batch)
    n = len(batch)
    out = {"input_ids": torch.zeros(n, max_len, dtype=torch.long),
           "attention_mask": torch.zeros(n, max_len, dtype=torch.long)}
    if "labels" in batch[0]:
        out["labels"] = torch.full((n, max_len), IGNORE_INDEX, dtype=torch.long)
        out["crf_tags"] = torch.zeros(n, max_len, dtype=torch.long)
    for i, b in enumerate(batch):
        L = int(b["input_ids"].numel())
        out["input_ids"][i, :L] = b["input_ids"]
        out["attention_mask"][i, :L] = 1
        if "labels" in b:
            out["labels"][i, :L] = b["labels"]
            out["crf_tags"][i, :L] = b["labels"]   # 词级 1:1，无续片
    return out


# ── BIO 解码 / 指标（与 BERT 脚本同一口径）──
def decode_bio_spans(tags: list[str]) -> list[tuple[int, int, str]]:
    spans, s, t = [], -1, None
    for i, tag in enumerate(tags + ["O"]):
        if tag.startswith("B-"):
            if s >= 0:
                spans.append((s, i, t))
            s, t = i, tag[2:]
        elif tag.startswith("I-"):
            if s < 0:
                s, t = i, tag[2:]
            elif t != tag[2:]:
                spans.append((s, i, t))
                s, t = i, tag[2:]
        else:
            if s >= 0:
                spans.append((s, i, t))
            s, t = -1, None
    return spans


def span_metrics(gold_rows, pred_rows) -> dict:
    per = {t: [0, 0, 0] for t in ("PER", "LOC", "ORG", "MISC")}
    tp = fp = fn = 0
    for g, p in zip(gold_rows, pred_rows):
        gs, ps = set(g), set(p)
        tp += len(gs & ps); fp += len(ps - gs); fn += len(gs - ps)
        for s in gs & ps:
            per[s[2]][0] += 1
        for s in ps - gs:
            per[s[2]][1] += 1
        for s in gs - ps:
            per[s[2]][2] += 1
    prf = lambda a, b, c: (a / (a + b) if a + b else 0.0, a / (a + c) if a + c else 0.0,
                           2 * a / (2 * a + b + c) if 2 * a + b + c else 0.0)
    P, R, F = prf(tp, fp, fn)
    return {"precision": P, "recall": R, "f1": F, "tp": tp, "fp": fp, "fn": fn,
            "per_type": {t: dict(zip(("precision", "recall", "f1"), prf(*v))) for t, v in per.items()}}


def illegal_bio_stats(tags_list) -> dict:
    illegal_seqs = illegal_start = illegal_trans = 0
    for tags in tags_list:
        bad = False
        prev = "O"
        for tag in tags:
            if tag.startswith("I-"):
                if prev == "O":
                    illegal_start += 1; bad = True
                elif prev[2:] != tag[2:]:
                    illegal_trans += 1; bad = True
            prev = tag
        illegal_seqs += int(bad)
    return {"total_seqs": len(tags_list), "illegal_seqs": illegal_seqs,
            "illegal_start": illegal_start, "illegal_transition": illegal_trans}


# ── CRF（与 BERT 脚本同实现）──
class CRFLayer(nn.Module):
    def __init__(self, num_tags: int, hard_constraint: bool = False):
        super().__init__()
        self.num_tags = num_tags
        self.trans = nn.Parameter(torch.randn(num_tags, num_tags) * 0.1)
        self.start = nn.Parameter(torch.randn(num_tags) * 0.1)
        self.end = nn.Parameter(torch.randn(num_tags) * 0.1)
        if hard_constraint:
            self.register_buffer("constraint", self.build_constraint())
        else:
            self.constraint = None

    def build_constraint(self):
        m = torch.zeros(self.num_tags, self.num_tags)
        for i, ti in enumerate(LABELS):
            for j, tj in enumerate(LABELS):
                if tj.startswith("I-"):
                    ok = (ti.startswith("B-") or ti.startswith("I-")) and ti[2:] == tj[2:]
                    if not ok:
                        m[i, j] = NEG_INF
        return m

    def transitions(self):
        return self.trans if self.constraint is None else self.trans + self.constraint

    def neg_log_likelihood(self, emissions, tags, mask):
        B, T, C = emissions.shape
        score = self.start[tags[:, 0]] + emissions[:, 0].gather(1, tags[:, 0:1]).squeeze(1)
        for t in range(1, T):
            m = mask[:, t].bool()
            if not m.any():
                continue
            score = torch.where(m, score + self.transitions()[tags[:, t - 1], tags[:, t]]
                                + emissions[:, t].gather(1, tags[:, t:t + 1]).squeeze(1), score)
        lens = mask.long().sum(1) - 1
        score = score + self.end[tags.gather(1, lens.unsqueeze(1)).squeeze(1)]
        alpha = self.start.unsqueeze(0) + emissions[:, 0]
        trans = self.transitions()
        for t in range(1, T):
            m = mask[:, t].bool()
            if not m.any():
                continue
            nxt = torch.logsumexp(alpha.unsqueeze(2) + trans.unsqueeze(0), dim=1) + emissions[:, t]
            alpha = torch.where(m.unsqueeze(1), nxt, alpha)
        logZ = torch.logsumexp(alpha + self.end.unsqueeze(0), dim=1)
        return (logZ - score).mean()

    def viterbi(self, emissions, mask):
        B, T, C = emissions.shape
        trans = self.transitions()
        alpha = self.start.unsqueeze(0) + emissions[:, 0]
        back = []
        for t in range(1, T):
            scores = alpha.unsqueeze(2) + trans.unsqueeze(0)
            best, idx = scores.max(dim=1)
            back.append(idx)
            alpha = torch.where(mask[:, t].bool().unsqueeze(1), best + emissions[:, t], alpha)
        alpha = alpha + self.end.unsqueeze(0)
        best_last = alpha.argmax(dim=1)
        lens = mask.long().sum(1)
        paths = []
        for b in range(B):
            L = int(lens[b].item())
            cur = int(best_last[b].item())
            path = [cur]
            for t in range(L - 2, -1, -1):
                cur = int(back[t][b, cur].item())
                path.append(cur)
            paths.append(list(reversed(path)))
        return paths


class BiLSTMNer(nn.Module):
    def __init__(self, vocab_size, num_tags, emb_dim=200, hidden=128, dropout=0.5,
                 use_crf=True, hard_constraint=False):
        super().__init__()
        self.use_crf = use_crf
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden * 2, num_tags)
        if use_crf:
            self.crf = CRFLayer(num_tags, hard_constraint)
        self.ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    def emissions(self, ids, mask):
        x = self.drop(self.emb(ids))
        x, _ = self.lstm(x)
        return self.head(self.drop(x))

    def forward(self, input_ids, attention_mask, labels=None):
        emit = self.emissions(input_ids, attention_mask)
        loss = None
        if labels is not None:
            if self.use_crf:
                # CRF 不支持 ignore_index：pad 的 -100 补成 O，由 attention_mask 屏蔽
                tags = labels.clone()
                tags[tags == IGNORE_INDEX] = 0
                loss = self.crf.neg_log_likelihood(emit, tags, attention_mask)
            else:
                loss = self.ce(emit.reshape(-1, emit.size(-1)), labels.reshape(-1))
        return emit, loss

    @torch.no_grad()
    def predict(self, input_ids, attention_mask):
        emit = self.emissions(input_ids, attention_mask)
        if self.use_crf:
            return self.crf.viterbi(emit, attention_mask)
        pred = emit.argmax(-1)
        return [[int(pred[b, t]) for t in range(int(attention_mask[b].sum()))]
                for b in range(pred.size(0))]


@torch.no_grad()
def predict_tags(model, loader, device):
    model.eval()
    all_paths, offset = [], 0
    for batch in loader:
        mask = batch["attention_mask"].to(device)
        paths = model.predict(batch["input_ids"].to(device), mask)
        all_paths.extend([[ID2LABEL[i] for i in p] for p in paths])
        offset += len(paths)
    return all_paths


@torch.no_grad()
def evaluate(model, loader, gold_rows, device):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        _, loss = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                        batch["labels"].to(device))
        total += float(loss.item()); n += 1
    tags = predict_tags(model, loader, device)
    pred_spans, gold_spans = [], []
    for row, tags_ in zip(gold_rows, tags):
        pred_spans.append(decode_bio_spans(tags_))
        gold_spans.append(decode_bio_spans(row["tags"][: len(tags_)]))
    met = span_metrics(gold_spans, pred_spans)
    met["illegal"] = illegal_bio_stats(tags)
    return total / max(n, 1), met


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(req=""):
    if req:
        return torch.device(req)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def latency_ms(model, loader, device, rows=500):
    model.eval()
    n = 0
    t0 = time.perf_counter()
    for batch in loader:
        model.predict(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        n += batch["input_ids"].size(0)
        if n >= rows:
            break
    return round((time.perf_counter() - t0) / max(n, 1) * 1000, 2)


def run(args):
    set_seed(args.seed)
    device = get_device(args.device)
    args.max_len = 64 if args.ablation == "len_half" else 128
    use_crf = args.ablation != "no_crf"
    hard = args.ablation == "crf_hard"

    train_rows = read_conll(DATA_DIR / "train.txt")
    dev_rows = read_conll(DATA_DIR / "dev.txt")
    test_rows = read_conll(DATA_DIR / "test.txt")
    vocab = build_vocab(train_rows, args.min_freq)
    print(f"数据 train/dev/test = {len(train_rows)}/{len(dev_rows)}/{len(test_rows)} 句；"
          f"词表 {len(vocab)}（min_freq={args.min_freq}，含 <PAD>/<UNK>）", flush=True)

    train_ds = NERDataset(train_rows, vocab, args.max_len)
    dev_ds = NERDataset(dev_rows, vocab, args.max_len)
    test_ds = NERDataset(test_rows, vocab, args.max_len)
    mk = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh,
                                   num_workers=0, collate_fn=collate_fn)
    train_loader, dev_loader, test_loader = mk(train_ds, True), mk(dev_ds, False), mk(test_ds, False)

    model = BiLSTMNer(len(vocab), len(LABELS), emb_dim=args.emb_dim, hidden=args.hidden,
                      dropout=args.dropout, use_crf=use_crf, hard_constraint=hard).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"设备 {device} | 参数 {n_params/1e6:.2f}M | 词向量 {args.emb_dim} | hidden {args.hidden} | "
          f"dropout {args.dropout} | decoder={'crf' if use_crf else 'linear'}"
          f"{'(硬约束)' if hard else ''} | max_len={args.max_len}", flush=True)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = MODEL_DIR / f"{args.run_name}.pt"
    log, best_f1, best_epoch = [], -1.0, 0
    t_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, nb = 0.0, 0
        t0 = time.perf_counter()
        for step, batch in enumerate(train_loader):
            opt.zero_grad()
            _, loss = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                            batch["labels"].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            total += float(loss.item()); nb += 1
            if (step + 1) % 200 == 0:
                print(f"  epoch {epoch} step {step+1}/{len(train_loader)} loss={total/nb:.4f}", flush=True)
        train_loss = total / max(nb, 1)
        dev_loss, dev_met = evaluate(model, dev_loader, dev_rows, device)
        print(f"epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | dev_loss={dev_loss:.4f} | "
              f"dev_P={dev_met['precision']:.4f} R={dev_met['recall']:.4f} F1={dev_met['f1']:.4f} | "
              f"非法序列={dev_met['illegal']['illegal_seqs']} | {time.perf_counter()-t0:.0f}s", flush=True)
        log.append({"epoch": epoch, "train_loss": round(train_loss, 6),
                    "dev_loss": round(dev_loss, 6), "dev_f1": round(dev_met["f1"], 6),
                    "dev_precision": round(dev_met["precision"], 6),
                    "dev_recall": round(dev_met["recall"], 6),
                    "dev_illegal_seqs": dev_met["illegal"]["illegal_seqs"],
                    "elapsed_s": round(time.perf_counter() - t0, 1)})
        if dev_met["f1"] > best_f1:
            best_f1, best_epoch = dev_met["f1"], epoch
            torch.save({"state_dict": model.state_dict(), "vocab": vocab, "labels": LABELS,
                        "args": vars(args), "epoch": epoch, "dev_f1": best_f1}, ckpt_path)
            print(f"  ★ dev 最优 → 保存 {ckpt_path.name}", flush=True)

    train_s = time.perf_counter() - t_start
    with open(LOG_DIR / f"log_{args.run_name}.json", "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    print(f"载入最优 checkpoint（epoch {ckpt['epoch']}，dev F1={best_f1:.4f}）", flush=True)

    _, dev_met = evaluate(model, dev_loader, dev_rows, device)
    test_loss, test_met = evaluate(model, test_loader, test_rows, device)
    print(f"【test 最终指标（只跑一次）】P={test_met['precision']:.4f} "
          f"R={test_met['recall']:.4f} F1={test_met['f1']:.4f}", flush=True)
    lat = latency_ms(model, test_loader, device)
    metrics = {"run_name": args.run_name, "ablation": args.ablation,
               "config": {**vars(args), "max_len": args.max_len, "decoder": "crf" if use_crf else "linear",
                          "hard_constraint": hard, "vocab_size": len(vocab)},
               "params_m": round(n_params / 1e6, 3), "train_s": round(train_s, 1),
               "best_epoch": best_epoch, "dev": dev_met, "test": test_met,
               "n_truncated_dev": dev_ds.n_truncated, "n_truncated_test": test_ds.n_truncated,
               "ms_per_sentence": lat}
    with open(LOG_DIR / f"metrics_{args.run_name}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"【{args.run_name} 汇总】dev F1={dev_met['f1']:.4f} test F1={test_met['f1']:.4f} | "
          f"非法 dev={dev_met['illegal']['illegal_seqs']} | {lat} ms/句 | "
          f"截断 dev={dev_ds.n_truncated} test={test_ds.n_truncated} | 训练 {train_s/60:.1f} 分钟", flush=True)
    return metrics


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", choices=["full", "no_crf", "crf_hard", "len_half"], default="full")
    ap.add_argument("--all_ablations", action="store_true")
    ap.add_argument("--data_dir", default="")
    ap.add_argument("--model_dir", default="")
    ap.add_argument("--log_dir", default="")
    ap.add_argument("--run_name", default="")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--emb_dim", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--min_freq", type=int, default=1)
    ap.add_argument("--max_grad_norm", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="")
    args = ap.parse_args()
    global DATA_DIR, MODEL_DIR, LOG_DIR
    if args.data_dir:
        DATA_DIR = Path(args.data_dir)
    if args.model_dir:
        MODEL_DIR = Path(args.model_dir)
    if args.log_dir:
        LOG_DIR = Path(args.log_dir)
    if not args.run_name:
        args.run_name = f"abl_{args.ablation}"
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.all_ablations:
        results = []
        for abl in ["full", "no_crf", "crf_hard", "len_half"]:
            a = argparse.Namespace(**vars(args))
            a.ablation, a.all_ablations, a.run_name = abl, False, f"abl_{abl}"
            results.append(run(a))
        print("\n=== 消融对比 ===")
        for m in results:
            print(f"{m['run_name']:16s} dev={m['dev']['f1']:.4f} test={m['test']['f1']:.4f} "
                  f"非法(dev)={m['dev']['illegal']['illegal_seqs']:3d} "
                  f"延迟={m['ms_per_sentence']}ms 截断(dev)={m['n_truncated_dev']}")
    else:
        run(args)
