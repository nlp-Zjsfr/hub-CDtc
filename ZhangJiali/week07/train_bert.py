# -*- coding: utf-8 -*-
"""BERT 联合序列标注事件抽取。线性分类头对每个 token 独立预测 BIO 标签。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup

from common import (
    BioDataset,
    CKPT_DIR,
    DATA_DIR,
    LOG_DIR,
    PRETRAINED_DIR,
    SequenceTagger,
    build_joint_schema,
    collate_bio,
    decode_bio_spans,
    ensure_dirs,
    evaluate,
    inner_split,
    load_enums,
    load_extract_split,
    load_schema,
    make_pred_row,
    peak_gpu_mb,
    save_predictions,
    set_seed,
    span_f1,
    spans_to_event,
    write_metrics,
)


class BertEncoder(nn.Module):
    """加载本地 BERT，输出最后一层隐状态。"""

    def __init__(self, model_path: str):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_path, local_files_only=True)
        self.hidden_size = self.bert.config.hidden_size

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        return output.last_hidden_state


def predict_events(model, loader, rows, id2label, enums, device):
    """将子词级预测写回字符位置，仅使用每个字的首子词标签。"""
    model.eval()
    char_tag_seqs = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            paths = model.predict(input_ids, mask, token_type_ids)
            word_ids = batch["word_ids"].cpu().tolist()
            for b, path in enumerate(paths):
                max_wid = max(word_ids[b]) if word_ids[b] else -1
                n_chars = max_wid + 1 if max_wid >= 0 else 0
                tags = ["O"] * n_chars
                prev_wid = None
                for j, wid in enumerate(word_ids[b]):
                    if wid < 0 or wid >= n_chars:
                        prev_wid = wid
                        continue
                    if wid != prev_wid and j < len(path):
                        tags[wid] = id2label.get(path[j], "O")
                    prev_wid = wid
                char_tag_seqs.append(tags)
    decoded = []
    for row, tags in zip(rows, char_tag_seqs):
        decoded.append(spans_to_event(decode_bio_spans(tags), row["text"], enums=enums))
    return decoded


def build_optimizer(model, lr: float, head_lr_mult: float, weight_decay: float):
    encoder_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (encoder_params if name.startswith("encoder.") else head_params).append(param)
    groups = [{"params": encoder_params, "lr": lr}]
    if head_params:
        groups.append({"params": head_params, "lr": lr * head_lr_mult})
    return AdamW(groups, weight_decay=weight_decay)


def parse_args():
    parser = argparse.ArgumentParser(description="BERT+Linear 事件抽取训练")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--model-path", default=str(PRETRAINED_DIR))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head-lr-mult", type=float, default=5.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--inner-val", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-eval", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="bert_linear")
    parser.add_argument("--no-save-model", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dirs()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"设备：{device}")
    print(f"预训练模型：{args.model_path}")

    data_dir = Path(args.data_dir)
    schema = load_schema(data_dir / "event_schema.json")
    enums = load_enums(data_dir / "event_schema.json")
    label_scheme = build_joint_schema(schema)
    id2label = label_scheme["id2label"]
    print(f"联合标签数 {len(label_scheme['labels'])}")

    train_rows = load_extract_split("train", data_dir)
    dev_rows = load_extract_split("dev", data_dir)
    if args.max_train > 0:
        train_rows = train_rows[:args.max_train]
    if args.max_eval > 0:
        dev_rows = dev_rows[:args.max_eval]
    fit_rows, val_rows = inner_split(train_rows, args.inner_val, args.seed)
    print(f"训练 {len(fit_rows)} / 内层验证 {len(val_rows)} / 评测 {len(dev_rows)}")

    tokenizer = BertTokenizer.from_pretrained(args.model_path, local_files_only=True)
    fit_loader = DataLoader(
        BioDataset(fit_rows, tokenizer, label_scheme, args.max_len),
        batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_bio,
    )
    val_loader = DataLoader(
        BioDataset(val_rows, tokenizer, label_scheme, args.max_len),
        batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_bio,
    )
    dev_loader = DataLoader(
        BioDataset(dev_rows, tokenizer, label_scheme, args.max_len),
        batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_bio,
    )

    encoder = BertEncoder(args.model_path)
    model = SequenceTagger(
        encoder, encoder.hidden_size, len(label_scheme["labels"]),
        dropout=args.dropout, use_crf=False,
    ).to(device)
    n_params = sum(param.numel() for param in model.parameters())
    print(f"参数量 {n_params / 1e6:.1f}M")

    optimizer = build_optimizer(model, args.lr, args.head_lr_mult, 0.01)
    total_steps = max(1, len(fit_loader) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    ckpt_path = CKPT_DIR / f"{args.run_name}.pt"
    history, best_f1, best_epoch = [], -1.0, 0
    train_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        total, steps = 0.0, 0
        for batch in fit_loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            _, loss = model(input_ids, mask, labels, token_type_ids)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
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
        "method": "bert_linear",
        "pretrained": "bert-base-chinese",
        "train_s": round(train_s, 1),
        "params_m": round(n_params / 1e6, 1),
        "ms_per_sentence": round(infer_s * 1000 / max(len(dev_rows), 1), 3),
        "peak_gpu_mb": peak_gpu_mb(),
        "epochs": args.epochs,
        "max_len": args.max_len,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "best_epoch": best_epoch,
        "inner_span_f1": round(best_f1, 4),
        "num_labels": len(label_scheme["labels"]),
        "n_truncated": BioDataset(dev_rows, tokenizer, label_scheme, args.max_len).n_truncated,
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
