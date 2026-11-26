"""
RoBERTa + Linear / CRF 联合事件抽取。

逐字符 BIO 标签使用课程数据层生成，由 RoBERTa 编码后交给分类层和所选解码器。
从 train 划分内层验证集选择最佳 epoch，dev 用于最终评测。
MPS 默认 batch_size=4、grad_accum=8，每 10 个 batch 输出进度。

使用方式（在 week7作业 目录运行）：
  python roberta/train.py
  python roberta/train.py --decoder linear
  python roberta/train.py --device mps
  python roberta/train.py --mock --device cpu --epochs 1

产物位于 outputs/checkpoints、outputs/logs、outputs/disc、outputs/metrics/disc。
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]   # 项目根目录（方法目录的上一级）
sys.path.insert(0, str(ROOT / "common"))     # 公共模块：event_data / crf / eval_utils / models / log_utils

from eval_utils import CKPT_DIR, LOG_DIR, Timer, evaluate_predictions
from log_utils import TrainLogger, peak_gpu_mb
from event_data import (BioDataset, DATA_DIR, ROOT, collate_bio,
                        build_joint_schema, decode_bio_spans,
                        illegal_bio_stats, load_enums, load_extract_split, load_schema,
                        make_pred_row, row_event, spans_to_event)
from model_paths import MOCK_DIR, resolve_model_path

from models import batch_to_device, build_model, load_tokenizer, predict_char_tags, select_device

IGNORE_INDEX = -100

# ─────────────────────────── 模型 ───────────────────────────

# ─────────────────────────── 训练 / 评估 ───────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def span_f1_from_char_tags(rows: list[dict], char_tags: list[list[str]]) -> float:
    """
    训练中做模型选择用的轻量指标：**span 级精确匹配 F1**（角色+文本完全一致），
    口径和最终评测的 argument F1 接近，但不依赖 930 条 dev、也更快。
    """
    tp = fp = fn = 0
    for row, tags in zip(rows, char_tags):
        ge = row_event(row) or {}
        gold = {(a["role"], a["text"]) for a in ge.get("args") or []}
        gold.add(("__trigger__", (ge.get("trigger") or {}).get("text", "")))
        ev = spans_to_event(decode_bio_spans(tags), row["text"])
        pred = {(a["role"], a["text"]) for a in (ev or {}).get("args", [])}
        pred.add(("__trigger__", (ev or {}).get("trigger", {}).get("text", "")))
        pred.discard(("__trigger__", ""))
        gold.discard(("__trigger__", ""))
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0

def evaluate_loss_and_f1(model, loader, id2label: dict, device) -> tuple[float, float]:
    """在验证集上算平均 loss 与 span F1。"""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            _, loss = model(batch["input_ids"], batch["attention_mask"],
                            batch["labels"], batch.get("token_type_ids"))
            total += float(loss.item())
            n += 1
    avg_loss = total / max(n, 1)
    tags, _ = predict_char_tags(model, loader, id2label, device)
    f1 = span_f1_from_char_tags(loader.dataset.rows, tags)
    return avg_loss, f1

def build_optimizer(model, args):
    """分层学习率：RoBERTa 用小 lr（2e-5），新初始化的分类头 用 args.head_lr_mult 倍。"""
    enc_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (enc_params if name.startswith("encoder.") else head_params).append(p)
    groups = [{"params": enc_params, "lr": args.lr}]
    if head_params:
        groups.append({"params": head_params, "lr": args.lr * args.head_lr_mult})
    return AdamW(groups, weight_decay=args.weight_decay)

def run_epoch(model, loader, optimizer, scheduler, device, args, epoch: int,
              logger=None) -> float:
    model.train()
    total, n = 0.0, 0
    optimizer.zero_grad()
    for step, batch in enumerate(loader):
        batch = batch_to_device(batch, device)
        _, loss = model(batch["input_ids"], batch["attention_mask"],
                        batch["labels"], batch.get("token_type_ids"))
        # 最后一组可能不足 grad_accum 个 batch，按实际组大小归一化。
        group_start = (step // args.grad_accum) * args.grad_accum
        group_size = min(args.grad_accum, len(loader) - group_start)
        (loss / group_size).backward()
        total += float(loss.item())
        n += 1
        if (step + 1) % args.grad_accum == 0:
            # 梯度裁剪，限制梯度范数
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            optimizer.step() # 用当前学习率更新模型参数
            scheduler.step() # 调整学习率，供下一次更新使用
            optimizer.zero_grad()
        if logger is not None:
            memory = {}
            if device.type == "mps" and step % args.log_every == 0:
                memory = {
                    "mps_allocated_mb": round(torch.mps.current_allocated_memory() / 1024**2, 1),
                    "mps_driver_allocated_mb": round(torch.mps.driver_allocated_memory() / 1024**2, 1),
                }
            logger.log_step(epoch=epoch, step=step, loss=float(loss.item()),
                            lr=scheduler.get_last_lr()[0], extra=memory)
        if (step + 1) % args.log_every == 0 or step + 1 == len(loader):
            print(f"  epoch {epoch} step {step + 1}/{len(loader)} loss={total / n:.4f}", flush=True)
    if len(loader) % args.grad_accum != 0: # 本轮最后是否还有不足一组的 batch，需要更新参数。
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return total / max(n, 1)

# ─────────────────────────── 主流程 ───────────────────────────

def main():
    args = parse_args()
    args.roberta_path = str(resolve_model_path(args.roberta_path))
    set_seed(args.seed)
    device = select_device(args.device)
    args.device = str(device)
    if args.batch_size is None:
        args.batch_size = 4 if device.type == "mps" else 32
    if args.grad_accum is None:
        args.grad_accum = 8 if device.type == "mps" else 1
    print(f"设备：{device}")
    print(f"batch_size={args.batch_size}，grad_accum={args.grad_accum}，"
          f"每次更新最多累计 {args.batch_size * args.grad_accum} 条样本")
    if device.type == "cuda":
        print(f"显存占用（启动时）：{torch.cuda.memory_allocated() / 1024 ** 2:.0f} MiB")

    data_dir = (Path(args.data_dir) if args.data_dir
                else (MOCK_DIR if args.mock else DATA_DIR))
    data_dir = data_dir.resolve()
    args.data_dir = str(data_dir)
    schema = load_schema(data_dir / "event_schema.json")
    enums = load_enums(data_dir / "event_schema.json")
    label_scheme = build_joint_schema(schema)
    labels = label_scheme["labels"]
    id2label = label_scheme["id2label"]
    print(f"标签体系：joint，{len(labels)} 个标签（{schema_stats_str(schema)}）")

    tokenizer = load_tokenizer(args.roberta_path)
    train_rows = load_extract_split("train", data_dir)
    dev_rows = load_extract_split("dev", data_dir)

    # 内层验证集：从 train 切 10%，只用于选 epoch；dev 完全不参与训练/选点
    rng = random.Random(args.seed)
    idxs = list(range(len(train_rows)))
    rng.shuffle(idxs)
    if len(train_rows) < 2:
        raise ValueError("至少需要两条训练样本，才能划分训练和内层验证集。")
    n_inner = min(len(train_rows) - 1, max(1, int(len(train_rows) * args.inner_val)))
    inner_idx = set(idxs[:n_inner])
    fit_rows = [r for i, r in enumerate(train_rows) if i not in inner_idx]
    val_rows = [r for i, r in enumerate(train_rows) if i in inner_idx]
    print(f"训练 {len(fit_rows)} 句 / 内层验证 {len(val_rows)} 句 / 评测 dev {len(dev_rows)} 句")

    fit_ds = BioDataset(fit_rows, tokenizer, label_scheme, args.max_len)
    val_ds = BioDataset(val_rows, tokenizer, label_scheme, args.max_len)
    dev_ds = BioDataset(dev_rows, tokenizer, label_scheme, args.max_len)
    fit_loader = DataLoader(fit_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_bio)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_bio)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_bio)

    model = build_model(args, label_scheme).to(device)
    initial_ckpt = None
    start_epoch = 1
    best_f1, best_epoch = -1.0, 0
    if args.init_ckpt:
        initial_ckpt = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        if initial_ckpt["label_scheme"]["labels"] != labels:
            raise ValueError("续训 checkpoint 的标签表与当前数据不一致。")
        saved_args = initial_ckpt["args"]
        if saved_args["decoder"] != args.decoder:
            raise ValueError("续训时 decoder 应与 checkpoint 一致。")
        for key in ("seed", "inner_val"):
            if saved_args[key] != getattr(args, key):
                raise ValueError(f"续训时 {key} 应与 checkpoint 一致，以保持训练/验证划分。")
        model.load_state_dict(initial_ckpt["state_dict"])
        best_epoch = int(initial_ckpt["epoch"])
        best_f1 = float(initial_ckpt["inner_val_span_f1"])
        start_epoch = best_epoch + 1
        if start_epoch > args.epochs:
            raise ValueError("epochs 是目标总轮数，续训时应大于 checkpoint 已完成的轮数。")
        print(f"从第 {best_epoch} 轮权重续训，接着训练第 {start_epoch}～{args.epochs} 轮。"
              "该 checkpoint 未保存优化器状态，本次重新初始化优化器和学习率计划。")
    num_layers = model.encoder.roberta.config.num_hidden_layers
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数：{n_params / 1e6:.1f}M（可训练 {n_trainable / 1e6:.1f}M），"
          f"解码器={args.decoder}，RoBERTa 层数={num_layers}")

    optimizer = build_optimizer(model, args)
    steps_per_epoch = (len(fit_loader) + args.grad_accum - 1) // args.grad_accum
    total_steps = steps_per_epoch * (args.epochs - start_epoch + 1)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"训练步数 {total_steps}（warmup {warmup_steps} 步，{args.warmup_ratio:.0%}）")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"{args.run_name}.pt"
    if initial_ckpt is not None:
        if ckpt_path.resolve() == Path(args.init_ckpt).resolve():
            raise ValueError("请用不同的 run_name 保存续训结果，以保留原 checkpoint。")
        # 将起始模型也纳入最佳模型比较，后续 F1 没有提升时仍可加载它。
        initial_ckpt["args"] = vars(args).copy()
        torch.save(initial_ckpt, ckpt_path)
        del initial_ckpt

    logger = TrainLogger(method=args.run_name, args=args, root=ROOT, params=n_params,
                         notes=f"RoBERTa 序列标注（{args.decoder}）训练过程")

    log = []
    train_start = time.perf_counter()

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss = run_epoch(model, fit_loader, optimizer, scheduler, device, args, epoch,
                               logger=logger)
        val_loss, val_f1 = evaluate_loss_and_f1(model, val_loader, id2label, device)
        elapsed = time.perf_counter() - t0
        print(f"epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | "
              f"inner_loss={val_loss:.4f} inner_spanF1={val_f1:.4f} | "
              f"{elapsed:.0f}s", flush=True)
        logger.log_epoch(epoch=epoch, train_loss=train_loss, elapsed=elapsed,
                         val_metrics={"inner_loss": val_loss, "inner_span_f1": val_f1},
                         extra={"best_so_far": round(best_f1, 6)})
        log.append({"epoch": epoch, "train_loss": round(train_loss, 6),
                    "inner_val_loss": round(val_loss, 6),
                    "inner_val_span_f1": round(val_f1, 6),
                    "elapsed_s": round(elapsed, 1)})
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            torch.save({"state_dict": model.state_dict(),
                        "label_scheme": {k: label_scheme[k] for k in
                                         ("labels", "label2id", "id2label", "name")},
                        "id2label": id2label,
                        "args": vars(args),
                        "epoch": epoch,
                        "inner_val_span_f1": val_f1}, ckpt_path)
            print(f"  ★ 内层验证最优 → 保存 {ckpt_path.name}", flush=True)

    train_s = time.perf_counter() - train_start
    with open(LOG_DIR / f"train_{args.run_name}.json", "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    # ── 用最优 checkpoint 在 dev 上出最终预测 ──
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    print(f"加载最优 checkpoint（epoch {ckpt['epoch']}，内层验证 spanF1={best_f1:.4f}）")

    with Timer() as timer:
        dev_tags, _ = predict_char_tags(model, dev_loader, id2label, device)
    dev_illegal = illegal_bio_stats(dev_tags)

    # 训练集上的非法统计也报一份（模型在见过的数据上表现如何，便于对照）
    with Timer() as timer_fit:
        fit_tags, _ = predict_char_tags(model, fit_loader, id2label, device)
    fit_illegal = illegal_bio_stats(fit_tags)

    preds = []
    n_empty = 0
    for row, tags in zip(dev_rows, dev_tags):
        ev = spans_to_event(decode_bio_spans(tags), row["text"], enums=enums) # 把字符级 BIO 标签转换成事件字典
        if ev is None:
            n_empty += 1
        preds.append(make_pred_row(row, [ev] if ev else []))
    print(f"dev 解码：{len(preds)} 句，其中没抽到触发词 {n_empty} 句")

    lat = timer.summary(len(dev_rows))
    meta = {
        "display_name": f"RoBERTa-{num_layers}L+{args.decoder.upper()}",
        "train_s": round(train_s, 1),
        "params_m": round(n_params / 1e6, 1),
        "ms_per_sentence": lat["ms_per_sentence"],
        "peak_gpu_mb": lat["peak_gpu_mb"],
        "decoder": args.decoder,
        "roberta_layers": num_layers,
        "max_len": args.max_len,
        "epochs": args.epochs,
        "lr": args.lr,
        "head_lr_mult": args.head_lr_mult,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "best_epoch": best_epoch,
        "inner_val_span_f1": round(best_f1, 4),
        "num_labels": len(labels),
        "illegal_dev": dev_illegal,
        "illegal_train_pred": fit_illegal,
        "n_truncated": dev_ds.n_truncated,
        "seed": args.seed,
        "selection_note": f"按 train 切出的 {args.inner_val:.0%} 内层验证集选 epoch；dev 用于最终评测",
        "roberta_path": args.roberta_path,
        "data_dir": args.data_dir,
        "init_ckpt": args.init_ckpt,
        "optimizer_reinitialized": bool(args.init_ckpt),
    }
    res = evaluate_predictions(args.run_name, preds, meta=meta,
                              gold_path=data_dir / "dev_extract.jsonl")

    print("\n【非法 BIO 统计（dev，模型原始输出）】")
    print(f"  序列数 {dev_illegal['total_seqs']}："
          f"含非法序列 {dev_illegal['illegal_seqs']} 条"
          f"（{dev_illegal['illegal_seq_rate']:.1%}），"
          f"其中非法开头 {dev_illegal['illegal_start']} 处、"
          f"非法转移 {dev_illegal['illegal_transition']} 处")
    print(f"  训练集对照：含非法序列 {fit_illegal['illegal_seqs']} 条"
          f"（{fit_illegal['illegal_seq_rate']:.1%}）")
    print(f"\n训练耗时 {train_s:.0f}s | 推理延迟 {lat['ms_per_sentence']} ms/句 | "
          f"峰值显存 {lat['peak_gpu_mb']:.0f} MiB")

    # 训练日志收尾（说明见 common/log_utils.py）：好让「训练过程」可被学生事后复看
    logger.finalize(
        total_s=train_s,
        best_epoch=best_epoch,
        best_inner_val_span_f1=round(best_f1, 6),
        dev_metrics={k: res[k]["f1"] for k in ("trigger", "argument", "event_exact")},
        inference_latency_ms=lat["ms_per_sentence"],
        peak_gpu_mb=peak_gpu_mb(),
        illegal_dev=dev_illegal,
        illegal_train_pred_sample=fit_illegal,
    )

def schema_stats_str(schema: dict) -> str:
    roles = {r for v in schema.values() for r in v}
    return f"{len(schema)} 类 / {sum(len(v) for v in schema.values())} 槽位 / {len(roles)} 去重角色"

def parse_args():
    ap = argparse.ArgumentParser(description="RoBERTa 序列标注做事件抽取（Linear / CRF）")
    ap.add_argument("--decoder", choices=["linear", "crf"], default="crf")
    ap.add_argument("--roberta_path", default="")
    ap.add_argument("--roberta_layers", type=int, default=0, help="0=完整 12 层；4=消融")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=None,
                    help="每批样本数；自动默认 MPS=4，CUDA/CPU=32")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head_lr_mult", type=float, default=5.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--grad_accum", type=int, default=None,
                    help="梯度累积批数；自动默认 MPS=8，CUDA/CPU=1")
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--inner_val", type=float, default=0.1, help="从 train 切多少做内层验证")
    ap.add_argument("--device", default="", help="cuda / mps / cpu；留空依次检测 CUDA、MPS、CPU")
    ap.add_argument("--log_every", type=int, default=10,
                    help="每多少步往 steps.jsonl 写一行（供学生画 loss 曲线）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run_name", default="")
    ap.add_argument("--init_ckpt", default="",
                    help="从已有模型权重接着训练；epochs 为目标总轮数，优化器重新初始化")
    ap.add_argument("--mock", action="store_true", help="用 data/_mock 的 20 条手造数据（离线冒烟）")
    ap.add_argument("--data_dir", default="", help="数据目录；留空=全量 data/processed（--mock 时用 data/_mock）")
    args = ap.parse_args()
    if args.epochs < 1 or any(v is not None and v < 1 for v in (args.batch_size, args.grad_accum)):
        ap.error("epochs、batch_size 和 grad_accum 必须大于 0")
    if args.log_every < 1:
        ap.error("log_every 必须大于 0")
    if not 0 < args.inner_val < 1:
        ap.error("inner_val 必须在 0 和 1 之间")
    if not args.roberta_path:
        args.roberta_path = str(ROOT / "roberta_base")
    if not args.run_name:
        tag = f"roberta{args.roberta_layers}L" if args.roberta_layers else "roberta"
        args.run_name = f"{tag}_{args.decoder}_{args.max_len}" + ("_mock" if args.mock else "")
        if args.init_ckpt:
            args.run_name += "_continued"
    return args

if __name__ == "__main__":
    main()
