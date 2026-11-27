"""在 TNEWS 数据集上微调轻量中文 RoBERTa 并完成分类评测。"""

from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from common import LABELS, OUTPUT_DIR, ROOT, evaluate, load_data, save_result, set_seed


class NewsDataset(Dataset):
    """将新闻标题编码为模型所需的张量。"""

    def __init__(self, samples: list[dict], tokenizer, max_length: int) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        encoded = self.tokenizer(
            sample["sentence"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(sample["label"], dtype=torch.long),
        }


@torch.no_grad()
def predict(model, loader: DataLoader, device: torch.device) -> tuple[list[int], list[int]]:
    model.eval()
    y_true, y_pred = [], []
    for batch in tqdm(loader, desc="评测", leave=False):
        labels = batch.pop("labels")
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(**batch).logits
        y_true.extend(labels.tolist())
        y_pred.extend(logits.argmax(dim=-1).cpu().tolist())
    return y_true, y_pred


def main() -> None:
    parser = argparse.ArgumentParser(description="微调轻量中文 RoBERTa")
    parser.add_argument("--model-name", default=str(ROOT / "pretrained_model"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-eval", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-save-model", action="store_true")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val, metadata = load_data()
    if args.max_train:
        train = train[: args.max_train]
    if args.max_eval:
        val = val[: args.max_eval]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        local_files_only=args.local_files_only,
    ).to(device)

    train_loader = DataLoader(
        NewsDataset(train, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        NewsDataset(val, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    started = time.perf_counter()
    epoch_losses = []
    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"训练 {epoch + 1}/{args.epochs}")
        for batch in progress:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")
        epoch_losses.append(round(running_loss / len(train_loader), 4))
    train_seconds = time.perf_counter() - started

    started = time.perf_counter()
    y_true, y_pred = predict(model, val_loader, device)
    inference_seconds = time.perf_counter() - started
    result = evaluate(y_true, y_pred)
    result.update({
        "model_name": "uer/chinese_roberta_L-2_H-128",
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "n_train": len(train),
        "n_eval": len(val),
        "train_seconds": round(train_seconds, 2),
        "inference_seconds": round(inference_seconds, 2),
        "epoch_losses": epoch_losses,
    })
    payload = {"data": metadata, "result": result, "predictions": y_pred}
    path = save_result("roberta_result.json", payload)

    if not args.no_save_model:
        model_dir = OUTPUT_DIR / "roberta_model"
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)

    print(f"RoBERTa: Accuracy={result['accuracy']:.4f}, Macro-F1={result['macro_f1']:.4f}")
    print(f"结果已保存至 {path}")


if __name__ == "__main__":
    main()

