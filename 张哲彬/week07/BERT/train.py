import torch
import torch.nn as nn
import numpy as np
import pathlib
import json
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, get_linear_schedule_with_warmup
from tqdm import tqdm
import argparse

from model import build_model
from dataset import PeopleDailyDataset, load_labels
from evaluate import calculate_f1, evaluate

ROOT_DIR = pathlib.Path(__file__).parent.parent
DATA_PATH = ROOT_DIR / "peoples_daily"
TRAIN_PATH = DATA_PATH / "train.json"
VAL_PATH = DATA_PATH / "validation.json"
LABELS_PATH = DATA_PATH / "label_names.json"
BERT_PATH = ROOT_DIR.parent / "pretrain_models" / "bert-base-chinese"
CKPT_DIR = ROOT_DIR / "checkpoints"

def parse_args():
    parser = argparse.ArgumentParser(description="BERT NER 训练")
    parser.add_argument("--crf",            action="store_true", help="是否使用CRF")
    parser.add_argument("--epochs",         default=10,   type=int)
    parser.add_argument("--batch_size",     default=32,  type=int)
    parser.add_argument("--max_length",     default=128, type=int)
    parser.add_argument("--lr",             default=2e-5, type=float,
                        help="BERT 层学习率")
    parser.add_argument("--head_lr_mult",   default=5.0,  type=float,
                        help="分类头学习率倍数（head_lr = lr * head_lr_mult）")
    parser.add_argument("--dropout",        default=0.1,  type=float)
    parser.add_argument("--warmup_ratio",   default=0.1,  type=float,
                        help="warmup 步数占总步数的比例")
    return parser.parse_args()

def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, idx2labels, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    total_predictions = []
    total_labels = []
    progress = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", unit="batch")
    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, token_type_ids)

        valid_mask = labels != -100
        valid_count = valid_mask.sum().item()
        if valid_count == 0:
            continue

        if model.crf is None:
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1)
            )
        else:
            loss = model.crf_loss(logits, labels, attention_mask)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if model.crf is None:
            predictions = logits.argmax(dim=-1)
        else:
            predictions = model.crf_decode(logits, attention_mask)
        total_predictions.append(predictions.detach().cpu())
        total_labels.append(labels.detach().cpu())
        total_loss += loss.item() * valid_count
        total_tokens += valid_count

    total_predictions = torch.cat(total_predictions, dim=0)
    total_labels = torch.cat(total_labels, dim=0)
    f1_score = calculate_f1(total_predictions, total_labels, idx2labels)
    avg_loss = total_loss / total_tokens
    return avg_loss, f1_score
    
def main():
    args = parse_args()

    pathlib.Path(CKPT_DIR).mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"使用设备: {device}")

    # 准备数据
    labels2idx, idx2labels = load_labels(LABELS_PATH)
    num_labels = len(labels2idx)

    tokenizer = BertTokenizerFast.from_pretrained(str(BERT_PATH))
    train_data = PeopleDailyDataset(TRAIN_PATH, tokenizer, labels2idx, max_length=args.max_length)
    val_data = PeopleDailyDataset(VAL_PATH, tokenizer, labels2idx, max_length=args.max_length)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    model = build_model(
        bert_path=str(BERT_PATH),
        num_labels=num_labels,
        dropout=0.1,
        crf=args.crf
    ).to(device)

    # 损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    #优化器
    param_groups = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        lr = args.lr if name.startswith("bert.") else args.lr * args.head_lr_mult
        no_decay = name.endswith(".bias") or "LayerNorm.weight" in name or name.startswith("crf.")
        weight_decay = 0.0 if no_decay else 0.01

        param_groups.setdefault((lr, weight_decay), []).append(param)

    optimizer = torch.optim.AdamW([
        {
            "params": params,
            "lr": lr,
            "weight_decay": weight_decay,
        }
        for (lr, weight_decay), params in param_groups.items()
    ])

    # warmup
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_val_f1 = 0.0
    for epoch in range(1, args.epochs+1):
        avg_loss, f1_score = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, idx2labels, device, epoch, args.epochs)
        val_f1 = evaluate(model, val_loader, device, idx2labels, print_report=False)
        print(f"Epoch {epoch}/{args.epochs} | "
              f"train_loss={avg_loss:.4f} train_f1={f1_score:.4f} | "
              f"val_f1={val_f1:.4f}")

        # 只保存验证集最优的 checkpoint
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            ckpt_path = CKPT_DIR / f"best_BERT.pt" if not args.crf else CKPT_DIR / f"best_BERT_CRF.pt"
            torch.save({
                "epoch":           epoch,
                "state_dict":      model.state_dict(),
                "val_f1":    val_f1,
                "args":            vars(args),
            }, ckpt_path)
            print(f"  ✓ 新最优模型已保存 → {ckpt_path}  (val_f1={val_f1:.4f})")


if __name__ == "__main__":
    main()