# -*- coding: utf-8 -*-
"""字符级 BiLSTM-CRF 事件抽取。不使用预训练语言模型，从随机初始化开始训练。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common import (
    CKPT_DIR,
    DATA_DIR,
    LOG_DIR,
    CharDataset,
    SequenceTagger,
    build_joint_schema,
    collate_char,
    decode_bio_spans,
    ensure_dirs,
    evaluate,
    inner_split,
    load_enums,
    load_extract_split,
    load_schema,
    make_pred_row,
    make_vocab,
    peak_gpu_mb,
    save_predictions,
    set_seed,
    span_f1,
    spans_to_event,
    write_metrics,
)


class BiLSTMEncoder(nn.Module):
    """双向 LSTM 字符编码器，输出维度为 hidden_size * 2。"""

    def __init__(self, vocab_size: int, emb_dim: int = 128, hidden: int = 256,
                 num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            emb_dim,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.hidden_size = hidden * 2

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        embedded = self.dropout(self.emb(input_ids))
        output, _ = self.lstm(embedded)
        return output


def predict_events(model, loader, rows, id2label, enums, device):
    model.eval()
    events = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            paths = model.predict(input_ids, mask)
            for path in paths:
                tags = [id2label.get(tag_id, "O") for tag_id in path]
                events.append(tags)
    decoded = []
    for row, tags in zip(rows, events):
        decoded.append(spans_to_event(decode_bio_spans(tags), row["text"], enums=enums))
    return decoded


def parse_args():
    parser = argparse.ArgumentParser(description="BiLSTM-CRF 事件抽取训练")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--inner-val", type=float, default=0.1)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-eval", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="bilstm_crf")
    parser.add_argument("--no-save-model", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dirs()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"设备：{device}")

    data_dir = Path(args.data_dir)
    schema = load_schema(data_dir / "event_schema.json")
    enums = load_enums(data_dir / "event_schema.json")
    label_scheme = build_joint_schema(schema)
    id2label = label_scheme["id2label"]

    train_rows = load_extract_split("train", data_dir)
    dev_rows = load_extract_split("dev", data_dir)
    if args.max_train > 0:
        train_rows = train_rows[:args.max_train]
    if args.max_eval > 0:
        dev_rows = dev_rows[:args.max_eval]

    vocab = make_vocab(train_rows)
    fit_rows, val_rows = inner_split(train_rows, args.inner_val, args.seed)
    print(f"字表 {len(vocab)}，训练 {len(fit_rows)} / 内层验证 {len(val_rows)} / 评测 {len(dev_rows)}")

    fit_loader = DataLoader(
        CharDataset(fit_rows, vocab, label_scheme, args.max_len),
        batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_char,
    )
    val_loader = DataLoader(
        CharDataset(val_rows, vocab, label_scheme, args.max_len),
        batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_char,
    )
    dev_loader = DataLoader(
        CharDataset(dev_rows, vocab, label_scheme, args.max_len),
        batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_char,
    )

    encoder = BiLSTMEncoder(len(vocab), args.emb_dim, args.hidden, dropout=args.dropout)
    model = SequenceTagger(
        encoder, encoder.hidden_size, len(label_scheme["labels"]),
        dropout=args.dropout, use_crf=True,
    ).to(device)
    n_params = sum(param.numel() for param in model.parameters())
    print(f"参数量 {n_params / 1e6:.2f}M")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ckpt_path = CKPT_DIR / f"{args.run_name}.pt"
    history, best_f1, best_epoch = [], -1.0, 0
    train_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        total, steps = 0.0, 0
        for batch in fit_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["attention_mask"].to(device)
            _, loss = model(input_ids, mask, labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total += float(loss.item())
            steps += 1
        train_loss = total / max(steps, 1)
        val_events = predict_events(model, val_loader, val_rows, id2label, enums, device)
        val_f1 = span_f1(val_rows, val_events)
        elapsed = time.perf_counter() - epoch_start
        print(f"epoch {epoch}/{args.epochs} train_loss={train_loss:.4f} "
              f"inner_spanF1={val_f1:.4f} {elapsed:.0f}s", flush=True)
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "inner_span_f1": round(val_f1, 6),
            "elapsed_s": round(elapsed, 1),
        })
        if val_f1 >= best_f1:
            best_f1, best_epoch = val_f1, epoch
            if not args.no_save_model:
                torch.save({
                    "state_dict": model.state_dict(),
                    "vocab": vocab,
                    "labels": label_scheme["labels"],
                    "epoch": epoch,
                    "inner_span_f1": val_f1,
                    "args": vars(args),
                }, ckpt_path)

    train_s = time.perf_counter() - train_start
    if ckpt_path.exists() and not args.no_save_model:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])

    infer_start = time.perf_counter()
    dev_events = predict_events(model, dev_loader, dev_rows, id2label, enums, device)
    infer_s = time.perf_counter() - infer_start
    preds = [make_pred_row(row, [event] if event else []) for row, event in zip(dev_rows, dev_events)]
    save_predictions(args.run_name, preds)
    result = evaluate(dev_rows, preds)
    meta = {
        "method": "bilstm_crf",
        "train_s": round(train_s, 1),
        "params_m": round(n_params / 1e6, 2),
        "ms_per_sentence": round(infer_s * 1000 / max(len(dev_rows), 1), 3),
        "peak_gpu_mb": peak_gpu_mb(),
        "epochs": args.epochs,
        "max_len": args.max_len,
        "best_epoch": best_epoch,
        "inner_span_f1": round(best_f1, 4),
        "vocab_size": len(vocab),
        "num_labels": len(label_scheme["labels"]),
        "pretrained": False,
        "seed": args.seed,
        "num_eval": len(dev_rows),
    }
    write_metrics(args.run_name, result, meta)
    (LOG_DIR / f"{args.run_name}_train.json").write_text(
        json.dumps({"history": history, "meta": meta}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
