"""
L2 bert-base-uncased 序列分类微调。

英文 span，标签是 SemEval-2020 Task 11 的 14 个手法。
评测与 L0/L1/L3/L4 相同：clean_val。
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import load_clean_val, load_splits
from common.labels import ID2LABEL, LABEL2ID, LABELS
from common.metrics import classification_report

OUT = Path(__file__).resolve().parent.parent / "outputs"
DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "pretrain_models" / "bert-base-uncased"


class SpanDataset(Dataset):
    def __init__(self, rows, tok, max_length):
        self.rows, self.tok, self.max_length = rows, tok, max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        enc = self.tok(
            row["sentence"],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": row["label"],
        }


def collate(batch, pad_id):
    width = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    labels = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        mask[i, :n] = torch.tensor(b["attention_mask"])
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def evaluate(model, tok, samples, device, max_length):
    model.eval()
    ys, ps = [], []
    loader = DataLoader(
        SpanDataset(samples, tok, max_length),
        batch_size=32,
        collate_fn=lambda b: collate(b, tok.pad_token_id),
    )
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            pred = logits.argmax(dim=-1).cpu()
            ys.extend(labels.tolist())
            ps.extend(pred.tolist())
    return ys, ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = "[PAD]"
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        id2label={str(i): name for i, name in ID2LABEL.items()},
        label2id=LABEL2ID,
    ).to(device)

    train, _ = load_splits()
    samples, meta = load_clean_val(seed=args.seed)
    loader = DataLoader(
        SpanDataset(train, tok, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, tok.pad_token_id),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(loader)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)
    log = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n_batch = 0.0, 0
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            running += loss.item()
            n_batch += 1
            if i % 50 == 0:
                print(f"  epoch {epoch} {i}/{len(loader)} loss={loss.item():.4f}", flush=True)
        row = {"epoch": epoch, "train_loss": round(running / max(1, n_batch), 4)}
        log.append(row)
        print(f"epoch {epoch}: train_loss={row['train_loss']}", flush=True)

    ys, ps = evaluate(model, tok, samples, device, args.max_length)
    report = classification_report(ys, ps, num_classes=len(LABELS))
    report["meta"] = meta
    report["method"] = "bert"
    report["train_log"] = log
    report["hparams"] = {
        "model": "bert-base-uncased",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_length": args.max_length,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    save_dir = OUT / "bert"
    model.save_pretrained(save_dir)
    tok.save_pretrained(save_dir)
    pred = [{"idx": d["idx"], "gold": g, "pred": p, "sentence": d["sentence"]}
            for d, g, p in zip(samples, ys, ps)]
    (OUT / "bert_pred.json").write_text(json.dumps(pred, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "bert_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[bert] n={report['n']} micro_f1={report.get('micro_f1')} macro_f1={report.get('macro_f1')}")


if __name__ == "__main__":
    main()
