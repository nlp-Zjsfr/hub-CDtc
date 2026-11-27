import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

DATASET_NAME = "sookiemonster/asrs-narratives-rebalance"

# 获取项目根目录 (homework/)
ROOT_DIR = Path(__file__).resolve().parent.parent


# [改] 本地/HF 双模路径解析
def resolve_model_name(name: str):
    """优先检查项目根目录下的路径，其次检查绝对路径，最后视为 HF 名"""
    p = Path(name).expanduser()
    if p.is_absolute() and p.exists():
        return str(p), True
    
    cand = ROOT_DIR / p
    if cand.exists():
        return str(cand.resolve()), True
        
    return name, False


# ---------------- 组件 ----------------
class BertWithPooling(nn.Module):
    def __init__(self, encoder, num_labels, pooling="cls", dropout=0.1):
        super().__init__()
        assert pooling in {"cls", "mean", "max"}, f"未知池化: {pooling}"
        self.encoder = encoder
        self.pooling = pooling
        hidden = encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state
        if self.pooling == "cls":
            x = h[:, 0]
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            x = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        else:
            mask = attention_mask.unsqueeze(-1).bool()
            x = h.masked_fill(~mask, torch.finfo(h.dtype).min).max(1).values
        return self.classifier(self.dropout(x))


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight, self.gamma = weight, gamma

    def forward(self, logits, targets):
        log_p = F.log_softmax(logits, dim=-1)
        log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        loss = -((1.0 - pt) ** self.gamma) * log_pt
        if self.weight is not None:
            loss = loss * self.weight[targets]
        return loss.mean()


class FGM:
    def __init__(self, model, epsilon=1.0):
        self.model, self.epsilon = model, epsilon
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if "word_embeddings" in name and param.requires_grad and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    param.data.add_(self.epsilon * param.grad / norm)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


# ================= 🌟 核心修改1：滑动窗口数据集 =================
class AsrsDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length, stride=128):
        self.input_ids = []
        self.attention_masks = []
        self.labels = []
        self.acns = []
        
        # dataset 是 HuggingFace 的 Dataset 对象，包含 text, label, acn
        texts = dataset["text"]
        labels = dataset["label"]
        acns = dataset["acn"]

        for text, label, acn in zip(texts, labels, acns):
            # 开启滑动窗口
            enc = tokenizer(
                text, 
                truncation=True, 
                max_length=max_length,
                stride=stride, 
                return_overflowing_tokens=True
            )
            # 一篇长文会被切成多个块
            for j in range(len(enc["input_ids"])):
                self.input_ids.append(enc["input_ids"][j])
                self.attention_masks.append(enc["attention_mask"][j])
                self.labels.append(label)
                self.acns.append(acn)  # 记录所属原报告ID，用于后续投票

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_masks[i],
            "label": self.labels[i],
            "acn": self.acns[i]
        }


# ================= 🌟 核心修改2：接受 acn 的 collate_fn =================
def collate_fn(batch, pad_token_id):
    max_len = max(len(b["input_ids"]) for b in batch)
    ids, mask, labels = [], [], []
    acns = []
    for b in batch:
        pad = max_len - len(b["input_ids"])
        ids.append(b["input_ids"] + [pad_token_id] * pad)
        mask.append(b["attention_mask"] + [0] * pad)
        labels.append(b["label"])
        acns.append(b["acn"])
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.tensor(mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "acn": acns
    }


def autocast_ctx(enabled):
    if not enabled:
        from contextlib import nullcontext
        return nullcontext()
    try:
        return torch.autocast(device_type="cuda", enabled=True)
    except AttributeError:
        return torch.cuda.amp.autocast(enabled=True)


def make_scaler(enabled):
    try:
        from torch.amp import GradScaler
        return GradScaler("cuda", enabled=enabled)
    except (ImportError, TypeError):
        from torch.cuda.amp import GradScaler
        return GradScaler(enabled=enabled)


# ================= 🌟 核心修改3：验证集聚合投票 =================
@torch.no_grad()
def predict(model, loader, device, use_fp16):
    model.eval()
    all_acns, all_labels, all_logits = [], [], []
    
    for batch in loader:
        acn = batch.pop("acn") # 取出 acn，不送入设备
        labels = batch["labels"]
        batch = {k: v.to(device) for k, v in batch.items()}
        
        with autocast_ctx(use_fp16 and device.type == "cuda"):
            logits = model(batch["input_ids"], batch["attention_mask"])
            
        all_acns.extend(acn)
        all_labels.extend(labels.tolist())
        all_logits.append(logits.cpu())
        
    all_logits = torch.cat(all_logits, dim=0)
    
    # 按 acn 分组进行软投票
    acn_logits_dict = defaultdict(list)
    acn_labels_dict = {}
    
    for i, acn in enumerate(all_acns):
        acn_logits_dict[acn].append(all_logits[i])
        acn_labels_dict[acn] = all_labels[i]
        
    final_preds, final_trues = [], []
    
    for acn, logits_list in acn_logits_dict.items():
        # 对同一个 acn 的多个切块结果求平均
        avg_logits = torch.stack(logits_list).mean(dim=0)
        pred = avg_logits.argmax().item()
        
        final_preds.append(pred)
        final_trues.append(acn_labels_dict[acn])
        
    return final_trues, final_preds


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="pretrain_models/bert-base-uncased",
                    help="本地目录（相对 homework 目录）或 HF 模型名")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--stride", type=int, default=128, help="滑动窗口步长(重叠字符数)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--eval_batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--pooling", choices=["cls", "mean", "max"], default="cls")
    ap.add_argument("--class_weight", action="store_true")
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--focal", action="store_true")
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--fgm", action="store_true")
    ap.add_argument("--fgm_epsilon", type=float, default=1.0)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_train", type=int, default=-1)
    ap.add_argument("--n_val", type=int, default=-1)
    ap.add_argument("--output_dir", default="outputs/bert_model")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device}")

    resolved_name, is_local = resolve_model_name(args.model_name)
    if is_local:
        print(f"[模型] 使用本地: {resolved_name}")
    else:
        print(f"[模型] 本地未找到 '{args.model_name}'，回退到 HuggingFace 名: {resolved_name}")
    load_kwargs = {"local_files_only": True} if is_local else {}

    print(f"[加载数据] {DATASET_NAME} ...")
    ds = load_dataset(DATASET_NAME)
    label_names = ds["train"].features["label"].names
    num_classes = len(label_names)
    print(f"[标签] {num_classes} 类")

    train, val = ds["train"], ds["validation"]
    n_train = len(train) if args.n_train < 0 else min(args.n_train, len(train))
    n_val = len(val) if args.n_val < 0 else min(args.n_val, len(val))
    train = train.select(range(n_train))
    val = val.select(range(n_val))

    tokenizer = AutoTokenizer.from_pretrained(resolved_name, **load_kwargs)
    encoder = AutoModel.from_pretrained(resolved_name, **load_kwargs)
    model = BertWithPooling(encoder, num_classes, pooling=args.pooling).to(device)
    
    print(f"[模型] {resolved_name} | pooling={args.pooling} | "
          f"参数量={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # 🌟 实例化带滑动窗口的数据集
    print(f"[滑动窗口] stride={args.stride} ...")
    train_ds = AsrsDataset(train, tokenizer, args.max_length, args.stride)
    val_ds = AsrsDataset(val, tokenizer, args.max_length, args.stride)
    print(f"[滑动窗口] 训练集样本切片: {len(train)} -> {len(train_ds)}；验证集: {len(val)} -> {len(val_ds)}")

    collate = lambda b: collate_fn(b, tokenizer.pad_token_id)
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=0, collate_fn=collate,
    )

    class_weights = None
    if args.class_weight:
        cnt = Counter(train["label"])
        total = sum(cnt.values())
        class_weights = torch.tensor(
            [total / (num_classes * max(cnt.get(i, 0), 1)) for i in range(num_classes)],
            dtype=torch.float, device=device,
        )
        print("[权重] " + ", ".join(f"{w:.2f}" for w in class_weights.cpu().tolist()))

    if args.focal:
        criterion = FocalLoss(weight=class_weights, gamma=args.focal_gamma)
        print(f"[Loss] Focal(gamma={args.focal_gamma})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
        print(f"[Loss] CE(label_smoothing={args.label_smoothing})")

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = make_scaler(args.fp16 and device.type == "cuda")
    fgm = FGM(model, epsilon=args.fgm_epsilon) if args.fgm else None
    if fgm:
        print(f"[FGM] epsilon={args.fgm_epsilon}")

    out_root = ROOT_DIR / args.output_dir
    best_dir = out_root / "best"
    last_dir = out_root / "last"
    for d in (best_dir, last_dir):
        d.mkdir(parents=True, exist_ok=True)

    def save_ckpt(model, tokenizer, target: Path, best_val):
        torch.save(model.state_dict(), target / "model.pt")
        model.encoder.config.save_pretrained(target / "encoder_config")
        tokenizer.save_pretrained(target / "tokenizer")
        cfg = {
            "model_name": resolved_name,
            "is_local": is_local,
            "pooling": args.pooling,
            "num_labels": num_classes,
            "label_names": label_names,
            "max_length": args.max_length,
            "best_val_macro_f1": best_val,
            "train_args": vars(args),
        }
        (target / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    best_val = -1.0
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        optimizer.zero_grad()
        for step, batch in enumerate(pbar):
            # 🌟 训练阶段把 acn 移除，以免 .to(device) 报错
            batch.pop("acn", None) 
            batch = {k: v.to(device) for k, v in batch.items()}

            with autocast_ctx(args.fp16 and device.type == "cuda"):
                logits = model(batch["input_ids"], batch["attention_mask"])
                loss = criterion(logits, batch["labels"]) / args.grad_accum
            scaler.scale(loss).backward()

            if fgm is not None:
                fgm.attack()
                with autocast_ctx(args.fp16 and device.type == "cuda"):
                    logits_adv = model(batch["input_ids"], batch["attention_mask"])
                    loss_adv = criterion(logits_adv, batch["labels"]) / args.grad_accum
                scaler.scale(loss_adv).backward()
                fgm.restore()

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            running += loss.item() * args.grad_accum
            pbar.set_postfix(loss=f"{running / (step + 1):.4f}")

        if len(train_loader) % args.grad_accum != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

        # 🌟 验证阶段使用聚合投票
        trues, preds = predict(model, val_loader, device, args.fp16)
        from sklearn.metrics import accuracy_score, f1_score
        val_acc = accuracy_score(trues, preds)
        val_f1 = f1_score(trues, preds, average="macro",
                          labels=list(range(num_classes)), zero_division=0)
        print(f"  [Epoch {epoch}] 聚合后 val_acc={val_acc:.4f}  val_macro_f1={val_f1:.4f}")

        save_ckpt(model, tokenizer, last_dir, best_val)
        if val_f1 > best_val:
            best_val = val_f1
            save_ckpt(model, tokenizer, best_dir, best_val)
            print(f"  ↑ 新 best 已保存 (val_macro_f1={best_val:.4f})")

    print(f"\n训练结束 | best_val_macro_f1={best_val:.4f}")
    print(f"best: {best_dir}")
    print(f"last: {last_dir}")
    print("→ 运行 bert_test.py 进行测试集评估")


if __name__ == "__main__":
    main()