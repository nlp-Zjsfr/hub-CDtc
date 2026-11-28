"""
方法 A：RoBERTa-wwm-ext + CRF 字符级序列标注（CLUENER 10 类，21 个 BIO 标签）

流程：
  train.jsonl ─字符级编码→ [CLS] c1 c2 ... cn [SEP] ─RoBERTa→ 768 维 ─Linear→ 21 维发射分 ─CRF→ loss
  每个 epoch 在 val 上算 span F1，保存最优权重；
  训练结束用最优权重在 dev 上解码两次（普通维特比 / 硬约束维特比），各写一份预测与指标。

字符级编码：中文 BERT 词表基本是单字，逐字查表（查不到先试小写，再退到 [UNK]），
保证「一个字符 ↔ 一个 token」，不需要子词对齐。

  # 冒烟（200 句 1 epoch）
  python crf_tagger/train.py --smoke
  # 正式
  python crf_tagger/train.py
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import BertModel, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "crf_tagger"))
from crf import CRF  # noqa: E402
from shared.data import TAG2ID, TAGS, bio_to_spans, count_illegal_bio, load_jsonl, save_jsonl, spans_to_bio, subsample  # noqa: E402
from shared.metrics import evaluate, format_report  # noqa: E402
from shared.paths import CKPT_DIR, DATA_DIR, LOG_DIR, METRIC_DIR, PRED_DIR, ROBERTA_DIR, ensure_dirs  # noqa: E402


# ─────────────────────────── 编码 ───────────────────────────

def load_vocab(model_dir):
    with open(Path(model_dir) / "vocab.txt", encoding="utf-8") as f:
        return {w.rstrip("\n"): i for i, w in enumerate(f)}


class CharEncoder:
    def __init__(self, vocab):
        self.v = vocab
        self.unk, self.cls, self.sep, self.pad = vocab["[UNK]"], vocab["[CLS]"], vocab["[SEP]"], vocab["[PAD]"]
        self.n_unk = 0

    def char_id(self, c):
        if c in self.v:
            return self.v[c]
        if c.lower() in self.v:
            return self.v[c.lower()]
        self.n_unk += 1
        return self.unk

    def encode(self, text):
        return [self.cls] + [self.char_id(c) for c in text] + [self.sep]


def make_batches(rows, enc, batch_size, shuffle, seed=0):
    """返回 list[dict]：input_ids/attn [B, T]，tags [B, T]（含 CLS/SEP 位，标 O），crf_mask [B, T]"""
    order = list(range(len(rows)))
    if shuffle:
        random.Random(seed).shuffle(order)
    batches = []
    for i in range(0, len(order), batch_size):
        chunk = [rows[j] for j in order[i:i + batch_size]]
        ids = [enc.encode(r["text"]) for r in chunk]
        T = max(len(x) for x in ids)
        inp = torch.full((len(chunk), T), enc.pad, dtype=torch.long)
        att = torch.zeros((len(chunk), T), dtype=torch.long)
        tag = torch.zeros((len(chunk), T), dtype=torch.long)       # 0 = "O"
        for b, (r, x) in enumerate(zip(chunk, ids)):
            inp[b, :len(x)] = torch.tensor(x)
            att[b, :len(x)] = 1
            if "entities" in r:
                bio = spans_to_bio(r["text"], r["entities"])
                tag[b, 1:1 + len(bio)] = torch.tensor([TAG2ID[t] for t in bio])
        # CRF 只看真实字符：去掉 CLS（第 0 位）与 SEP，让 crf_mask 左对齐从字符开始
        batches.append({"rows": chunk, "input_ids": inp, "attn": att, "tags": tag[:, 1:T - 1],
                        "crf_mask": att[:, 2:].bool()})        # 长度 T-2，第 i 位 = 第 i 个字符是否存在
    return batches


# ─────────────────────────── 模型 ───────────────────────────

class RobertaCRF(nn.Module):
    def __init__(self, model_dir, dropout=0.1):
        super().__init__()
        self.encoder = BertModel.from_pretrained(str(model_dir), add_pooling_layer=False)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.config.hidden_size, len(TAGS))
        self.crf = CRF(TAGS)

    def emissions(self, input_ids, attn):
        h = self.encoder(input_ids=input_ids, attention_mask=attn).last_hidden_state
        em = self.head(self.dropout(h))
        return em[:, 1:-1]                   # 去掉 CLS 与最后一列（最长句的 SEP），与 tags/crf_mask 对齐

    def loss(self, b):
        return self.crf(self.emissions(b["input_ids"], b["attn"]), b["tags"], b["crf_mask"])

    def decode(self, b, hard=False):
        return self.crf.decode(self.emissions(b["input_ids"], b["attn"]), b["crf_mask"], hard=hard)


def to_dev(b, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}


@torch.no_grad()
def predict(model, batches, device, hard=False):
    """返回 (pred_rows, 非法 BIO 转移数, 秒)"""
    model.eval()
    preds, n_illegal = [], 0
    t0 = time.time()
    for b in batches:
        paths = model.decode(to_dev(b, device), hard=hard)
        for r, p in zip(b["rows"], paths):
            tags = [TAGS[i] for i in p]
            n_illegal += count_illegal_bio(tags)
            preds.append({"id": r["id"], "text": r["text"], "entities": bio_to_spans(r["text"], tags)})
    if device.type == "mps":
        torch.mps.synchronize()
    return preds, n_illegal, time.time() - t0


# ─────────────────────────── 主流程 ───────────────────────────

def pick_device(name):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=str(ROBERTA_DIR))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5, help="编码器学习率")
    ap.add_argument("--head_lr", type=float, default=1e-3, help="Linear + CRF 学习率")
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--run_name", default="crf_tagger")
    ap.add_argument("--smoke", action="store_true", help="冒烟：train/val/dev 各取前 200 句，1 epoch")
    ap.add_argument("--frac", type=float, default=1.0, help="试运行：train/val/dev 各随机取该比例（其余配置不变）")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.run_name = 1, args.run_name + "_smoke"

    ensure_dirs()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    train = load_jsonl(DATA_DIR / "train.jsonl")
    val = load_jsonl(DATA_DIR / "val.jsonl")
    dev = load_jsonl(DATA_DIR / "dev.jsonl")
    if args.smoke:
        train, val, dev = train[:200], val[:200], dev[:200]
    train, val, dev = (subsample(x, args.frac, args.seed) for x in (train, val, dev))
    print(f"device={device}  train={len(train)} val={len(val)} dev={len(dev)}", flush=True)

    enc = CharEncoder(load_vocab(args.model_dir))
    val_b = make_batches(val, enc, 64, shuffle=False)
    dev_b = make_batches(dev, enc, 64, shuffle=False)

    model = RobertaCRF(args.model_dir, args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    no_decay = ("bias", "LayerNorm.weight")
    enc_named = list(model.encoder.named_parameters())
    groups = [
        {"params": [p for n, p in enc_named if not any(x in n for x in no_decay)], "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": [p for n, p in enc_named if any(x in n for x in no_decay)], "lr": args.lr, "weight_decay": 0.0},
        {"params": list(model.head.parameters()) + list(model.crf.parameters()), "lr": args.head_lr, "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups)
    steps_per_epoch = math.ceil(len(train) / args.batch_size)
    total = steps_per_epoch * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(total * args.warmup_ratio), total)

    log = {"run_name": args.run_name, "args": vars(args), "device": str(device),
           "n_params": n_params, "n_train": len(train), "epochs": []}
    steps_f = open(LOG_DIR / f"{args.run_name}_steps.jsonl", "w", encoding="utf-8")
    ckpt = CKPT_DIR / args.run_name / "best.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    best_f1, step, t_train = -1.0, 0, time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        t_ep, run_loss = time.time(), 0.0
        for i, b in enumerate(make_batches(train, enc, args.batch_size, shuffle=True, seed=args.seed + ep)):
            loss = model.loss(to_dev(b, device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            run_loss += loss.item()
            if step % args.log_every == 0:
                rec = {"step": step, "epoch": ep, "loss": round(run_loss / (i + 1), 4),
                       "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t_train, 1)}
                steps_f.write(json.dumps(rec) + "\n"); steps_f.flush()
                eta = rec["elapsed_s"] / step * (total - step)
                print(f"  ep{ep} step {step}/{total}  loss {rec['loss']:.4f}  "
                      f"{rec['elapsed_s'] / 60:.1f} min  ETA {eta / 60:.1f} min", flush=True)
        ep_time = time.time() - t_ep
        preds, n_ill, _ = predict(model, val_b, device)
        m = evaluate(val, preds)
        rec = {"epoch": ep, "train_loss": round(run_loss / steps_per_epoch, 4), "epoch_time_s": round(ep_time, 1),
               "val_span_f1": m["span"]["f1"], "val_illegal_bio": n_ill}
        log["epochs"].append(rec)
        better = m["span"]["f1"] > best_f1
        if better:
            best_f1 = m["span"]["f1"]
            torch.save(model.state_dict(), ckpt)
        print(f"[ep{ep}] loss {rec['train_loss']:.4f}  val span F1 {rec['val_span_f1']:.4f}  "
              f"非法BIO {n_ill}  {ep_time:.0f}s {'★best' if better else ''}", flush=True)
    steps_f.close()
    log["train_time_s"] = round(time.time() - t_train, 1)
    log["best_val_span_f1"] = best_f1

    # ── 用最优权重在 dev 上评测：普通维特比 vs 硬约束维特比 ──
    model.load_state_dict(torch.load(ckpt, map_location=device))
    log["dev"] = {}
    for hard in (False, True):
        tag = "hard" if hard else "viterbi"
        preds, n_ill, sec = predict(model, dev_b, device, hard=hard)
        m = evaluate(dev, preds)
        m.update({"illegal_bio": n_ill, "latency_ms_per_sent": round(sec / len(dev) * 1000, 2),
                  "train_time_s": log["train_time_s"], "n_params": n_params, "n_trainable": n_params})
        name = f"{args.run_name}_{tag}"
        save_jsonl([{"id": p["id"], "entities": p["entities"]} for p in preds], PRED_DIR / f"{name}.jsonl")
        (METRIC_DIR / f"{name}.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
        log["dev"][tag] = {"span_f1": m["span"]["f1"], "text_f1": m["text"]["f1"], "illegal_bio": n_ill}
        print(format_report(m, f"dev / {name}") + f"\n非法 BIO 转移 {n_ill}  延迟 {m['latency_ms_per_sent']} ms/句", flush=True)

    log["n_unk_chars"] = enc.n_unk
    (LOG_DIR / f"{args.run_name}_train.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
