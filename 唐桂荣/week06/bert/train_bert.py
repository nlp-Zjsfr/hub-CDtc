"""微调本地 BERT 和 RoBERTa，支持九组单因素实验。"""

import argparse
import json
import math
import sys
from functools import partial
from itertools import islice
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import torch
import transformers
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

PROJECT_DIR = Path(__file__).resolve().parent.parent
if not __package__:
    sys.path.insert(0, str(PROJECT_DIR))

from common.data import SPLIT_SEED, VAL_FRACTION, Sample, load_splits
from common.metrics import classification_report
from bert.experiments import EXPERIMENTS, MODEL_NAMES

SEED = 42
EPOCHS = 3
HEAD_LR_MULTIPLIER = 5
WARMUP_FRACTION = 0.1
WEIGHT_DECAY = 0.01
DROPOUT = 0.1


class BertClassifier(nn.Module):
    """池化最后一层的表示，用线性层预测商品类别。"""

    def __init__(self, model_dir: Path, n_labels: int, pool: str = "cls"):
        super().__init__()
        self.encoder, info = AutoModel.from_pretrained(
            model_dir, local_files_only=True, add_pooling_layer=False,
            output_loading_info=True,
        )
        if info.get("missing_keys") or info.get("mismatched_keys"):
            raise RuntimeError(
                f"{model_dir.name} 编码器权重加载不完整。"
                f"缺失 {info.get('missing_keys')}，尺寸不匹配 {info.get('mismatched_keys')}"
            )
        self.dropout = nn.Dropout(DROPOUT)
        self.pool = pool
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(2 * hidden_size if pool == "cls_mean" else hidden_size, n_labels)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True,
        ).last_hidden_state
        return self.classifier(self.dropout(pool_hidden(hidden, attention_mask, self.pool)))


def pool_hidden(hidden, attention_mask, pool):
    if pool == "cls":
        return hidden[:, 0]
    valid = attention_mask.unsqueeze(-1).bool()
    if pool == "max":
        return hidden.masked_fill(~valid, torch.finfo(hidden.dtype).min).max(dim=1).values
    mean = hidden.masked_fill(~valid, 0).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
    if pool == "cls_mean":
        return torch.cat((hidden[:, 0], mean), dim=1)
    if pool == "mean":
        return mean
    raise ValueError(f"未知池化方式：{pool}")


def encode_samples(samples: list[Sample], tokenizer, max_length: int = 64) -> TensorDataset:
    encoded = tokenizer(
        [sample["sentence"] for sample in samples],
        padding="max_length", truncation=True, max_length=max_length,
        return_tensors="pt",
    )
    labels = torch.tensor([sample["label"] for sample in samples])
    return TensorDataset(encoded["input_ids"], encoded["attention_mask"], labels)


def focal_loss(logits, labels, *, weights, gamma):
    log_probs = F.log_softmax(logits, dim=-1)
    target_log_probs = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    target_weights = weights[labels]
    modulation = (-target_log_probs.expm1()).pow(gamma)
    # 沿用加权交叉熵的分母。gamma=0 时两者相同。
    return -(target_weights * modulation * target_log_probs).sum() / target_weights.sum()


def batch_loss(model, ids, mask, labels, criterion, use_bf16):
    with torch.autocast(ids.device.type, dtype=torch.bfloat16, enabled=use_bf16):
        logits = model(ids, mask)
    return criterion(logits.float(), labels)


def fgm_backward(model, ids, mask, labels, criterion, use_bf16, epsilon):
    embedding = model.encoder.get_input_embeddings().weight
    with torch.no_grad():
        clean_embedding = embedding.clone()
        gradient = embedding.grad
        scale = gradient.norm().clamp_min(1e-12)
        embedding.add_(gradient / scale, alpha=epsilon)
    try:
        batch_loss(model, ids, mask, labels, criterion, use_bf16).backward()
    finally:
        with torch.no_grad():
            embedding.copy_(clean_embedding)


def train_epoch(model, batches, steps, optimizer, scheduler, criterion, device, use_bf16, fgm_eps=0.0):
    model.train()
    start = perf_counter()
    total_loss = torch.zeros((), device=device)
    for step, (ids, mask, labels) in enumerate(batches, start=1):
        optimizer.zero_grad(set_to_none=True)
        ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
        loss = batch_loss(model, ids, mask, labels, criterion, use_bf16)
        loss.backward()
        if fgm_eps:
            fgm_backward(model, ids, mask, labels, criterion, use_bf16, fgm_eps)
        optimizer.step()
        scheduler.step()
        total_loss += loss.detach()
        if step == 1 or step % 25 == 0 or step == steps:
            mean_loss = total_loss.item() / step
            if not math.isfinite(mean_loss):
                raise RuntimeError(f"第 {step} 步出现非有限损失")
            print(
                f"  train {step}/{steps}  mean_loss={mean_loss:.4f}"
                f"  elapsed={perf_counter() - start:.1f}s",
                flush=True,
            )
    return mean_loss, perf_counter() - start


@torch.inference_mode()
def evaluate(model, batches, device, use_bf16):
    model.eval()
    actual = []
    predictions = []
    for ids, mask, labels in batches:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(ids.to(device), mask.to(device))
        predictions.extend(logits.float().argmax(dim=1).cpu().tolist())
        actual.extend(labels.tolist())
    return actual, predictions


def write_checkpoint(path: Path, state: dict):
    # 在同一目录写临时文件，完整写入后再原子替换目标文件。
    with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save(state, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_report(path: Path, report: dict):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description="商品分类实验")
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS), default="exp1")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--precision", choices=("float32", "bf16"), default="float32")
    parser.add_argument("--smoke-steps", type=int, default=0, help="只跑指定步数和两个验证批次")
    parser.add_argument("--out", type=Path, help="正式训练的输出目录，相对路径以作业目录为准")
    args = parser.parse_args()
    experiment = EXPERIMENTS[args.experiment]
    if args.batch_size < 1 or args.smoke_steps < 0:
        parser.error("batch-size 应为正整数，smoke-steps 应为非负整数")

    device_name = args.device
    if device_name == "auto":
        device_name = "cpu"
        if torch.cuda.is_available():
            device_name = "cuda"
        elif torch.backends.mps.is_available():
            device_name = "mps"
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("当前会话无法使用 CUDA")
    if device_name == "mps" and not torch.backends.mps.is_available():
        parser.error("当前会话无法使用 MPS")
    device = torch.device(device_name)
    use_bf16 = args.precision == "bf16"

    torch.manual_seed(SEED)
    splits = load_splits()
    model_dir = PROJECT_DIR / "model" / args.model
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    train_data = encode_samples(splits.train, tokenizer, experiment.max_length)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    val_data = encode_samples(splits.val, tokenizer, experiment.max_length)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)

    model = BertClassifier(model_dir, len(splits.labels), experiment.pool).to(device)
    head_lr = experiment.encoder_lr * HEAD_LR_MULTIPLIER
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": experiment.encoder_lr},
            {"params": model.classifier.parameters(), "lr": head_lr},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_FRACTION)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    class_counts = train_data.tensors[-1].bincount(minlength=len(splits.labels))
    class_weights = len(train_data) / (len(splits.labels) * class_counts.float())
    weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=experiment.label_smoothing)
    loss_name = "weighted_cross_entropy"
    if experiment.focal_gamma:
        criterion = partial(focal_loss, weights=weights, gamma=experiment.focal_gamma)
        loss_name = "weighted_focal"

    print(
        f"{args.model} {args.experiment} ({experiment.description}) | {device}/{args.precision}"
        f" | batch_size={args.batch_size} |"
        f" train={len(splits.train)} | val={len(splits.val)} |"
        f" steps_per_epoch={len(train_loader)}",
        flush=True,
    )
    if args.smoke_steps:
        steps = min(args.smoke_steps, len(train_loader))
        train_epoch(
            model, islice(train_loader, steps), steps, optimizer, scheduler, criterion,
            device, use_bf16, experiment.fgm_eps,
        )
        start = perf_counter()
        actual, predictions = evaluate(model, islice(val_loader, 2), device, use_bf16)
        if len(actual) != len(predictions):
            raise RuntimeError("验证预测数量与标签数量不符")
        print(f"  smoke val={len(actual)}  elapsed={perf_counter() - start:.1f}s", flush=True)
        return

    out = PROJECT_DIR / (args.out or Path("outputs/bert") / args.model / args.experiment)
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "model": args.model,
        "experiment": args.experiment,
        "split": "val",
        "split_seed": SPLIT_SEED,
        "val_fraction": VAL_FRACTION,
        "split_sizes": {"train": len(splits.train), "val": len(splits.val), "test": len(splits.test)},
        "labels": splits.labels,
        "config": {
            "pool": experiment.pool, "max_length": experiment.max_length, "epochs": EPOCHS,
            "batch_size": args.batch_size, "seed": SEED,
            "encoder_lr": experiment.encoder_lr, "head_lr": head_lr,
            "weight_decay": WEIGHT_DECAY, "dropout": DROPOUT, "loss": loss_name,
            "fgm_eps": experiment.fgm_eps, "label_smoothing": experiment.label_smoothing,
            "focal_gamma": experiment.focal_gamma,
            "class_weights": class_weights.tolist(), "warmup_fraction": WARMUP_FRACTION,
            "warmup_steps": warmup_steps, "total_steps": total_steps,
            "device": device.type, "precision": args.precision,
            "torch_version": str(torch.__version__), "transformers_version": transformers.__version__,
        },
        "history": [],
        "best_epoch": None,
        "status": "running",
    }
    report_path = out / "result.json"
    write_report(report_path, report)
    best_macro_f1 = -math.inf
    for epoch in range(1, EPOCHS + 1):
        print(f"epoch {epoch}/{EPOCHS}", flush=True)
        train_loss, train_seconds = train_epoch(
            model, train_loader, len(train_loader), optimizer, scheduler, criterion,
            device, use_bf16, experiment.fgm_eps,
        )
        start = perf_counter()
        actual, predicted = evaluate(model, val_loader, device, use_bf16)
        val_seconds = perf_counter() - start
        metrics = classification_report(actual, predicted, splits.labels)
        report["history"].append({
            "epoch": epoch, "train_loss": train_loss, "train_seconds": train_seconds,
            "val_seconds": val_seconds, "macro_f1": metrics["macro_f1"],
            "accuracy": metrics["accuracy"],
        })
        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            report["best_epoch"] = epoch
            report["metrics"] = metrics
            report["predictions"] = [
                {"idx": sample["idx"], "label": label, "predicted_label": prediction}
                for sample, label, prediction in zip(splits.val, actual, predicted, strict=True)
            ]
            write_checkpoint(out / "best.pt", {
                "model": args.model, "experiment": args.experiment, "epoch": epoch,
                "labels": splits.labels, "config": report["config"],
                "state_dict": {name: tensor.cpu() for name, tensor in model.state_dict().items()},
            })
        if epoch == EPOCHS:
            report["status"] = "complete"
        write_report(report_path, report)
        print(
            f"epoch {epoch} | val Macro-F1={metrics['macro_f1']:.4f}"
            f" | Accuracy={metrics['accuracy']:.4f}"
            f" | best epoch={report['best_epoch']}",
            flush=True,
        )
    print(f"结果已写入 {report_path}", flush=True)


if __name__ == "__main__":
    main()
