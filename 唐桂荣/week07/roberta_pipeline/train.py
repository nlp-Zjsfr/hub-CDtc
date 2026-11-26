"""
RoBERTa 流水线事件抽取：事件检测 → 类型分类 → 类型条件论元抽取。

沿用课程 bert_pipeline 的三段流程。检测器和类型分类器使用完整编码器，
每种事件类型单独训练一个 Linear 论元专家，专家默认保留四层预训练编码器。
十三种类型都有训练样本时，共训练 15 个模型，依次加载和释放。

在 week7作业 目录运行：
  python roberta_pipeline/train.py --device mps
  python roberta_pipeline/train.py --skip_train --device mps
  python roberta_pipeline/train.py --only_experts --device mps
  python roberta_pipeline/train.py --mock --cls_epochs 1 --arg_epochs 1

数据默认读取相邻原课程目录，输出以 roberta_pipeline 为前缀。
--mock 使用独立的 roberta_pipeline_mock 前缀。
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import gc
import json
import random
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))

from event_data import (BioDataset, ClsDataset, build_typed_schema, collate_bio,
                        collate_cls, decode_bio_spans, load_detect_split, load_enums,
                        load_extract_split, load_jsonl, load_schema, make_pred_row,
                        row_event, spans_to_event)
from eval_utils import CKPT_DIR, METRIC_DIR, Timer, evaluate_predictions
from evaluate_events import detection_metrics
from log_utils import EvalLogger, TrainLogger
from model_paths import DATA_DIR, MOCK_DIR, resolve_model_path
from models import (RobertaCls, build_arg_model, load_tokenizer,
                    predict_char_tags, select_device)


def checkpoint_path(args, stage: str) -> Path:
    return CKPT_DIR / f"{args.run_name}_{stage}.pt"


def release_memory(device) -> None:
    """阶段模型的引用删除后，释放可回收的设备缓存。"""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def save_checkpoint(model, path: Path, args, **extra) -> None:
    """保存 CPU 权重，并在完整写入后替换 checkpoint。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "args": vars(args).copy(),
        **extra,
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def train_classifier(rows, labels_of, num_labels, tokenizer, args, device,
                     ckpt_name: str, task: str, logger=None) -> tuple[RobertaCls, dict]:
    """检测器和类型分类器共用训练流程。"""
    ds = ClsDataset(rows, tokenizer, args.cls_max_len, task=task,
                    type2id=labels_of if isinstance(labels_of, dict) else None)
    loader = DataLoader(ds, batch_size=args.cls_batch_size, shuffle=True,
                        num_workers=0, collate_fn=collate_cls)
    model = RobertaCls(str(args.roberta_path), num_labels, args.dropout).to(device)
    optimizer = AdamW(model.parameters(), lr=args.cls_lr, weight_decay=0.01)
    steps_per_epoch = (len(loader) + args.cls_grad_accum - 1) // args.cls_grad_accum
    total_steps = steps_per_epoch * args.cls_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)

    start = time.perf_counter()
    for epoch in range(1, args.cls_epochs + 1):
        model.train()
        optimizer.zero_grad()
        epoch_start = time.perf_counter()
        total, n = 0.0, 0
        for step, batch in enumerate(loader):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            tti = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)
            loss = model(ids, mask, tti, labels)[1]
            group_start = (step // args.cls_grad_accum) * args.cls_grad_accum
            group_size = min(args.cls_grad_accum, len(loader) - group_start)
            (loss / group_size).backward()
            if (step + 1) % args.cls_grad_accum == 0 or step + 1 == len(loader):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            total += float(loss.item())
            n += 1
            if logger is not None:
                logger.log_step(epoch=epoch, step=step, loss=float(loss.item()),
                                lr=scheduler.get_last_lr()[0], extra={"stage": task})
            if (step + 1) % args.log_every == 0 or step + 1 == len(loader):
                print(f"  [{task}] epoch {epoch}/{args.cls_epochs} "
                      f"step {step + 1}/{len(loader)} loss={total / n:.4f}", flush=True)
        if logger is not None:
            logger.log_epoch(epoch=epoch, train_loss=total / max(n, 1),
                             elapsed=time.perf_counter() - epoch_start,
                             extra={"stage": task})
    info = {"train_s": round(time.perf_counter() - start, 1), "epochs": args.cls_epochs,
            "n_rows": len(rows), "num_labels": num_labels}
    save_checkpoint(model, CKPT_DIR / ckpt_name, args, task=task,
                    type2id=labels_of if isinstance(labels_of, dict) else {}, training_info=info)
    return model, info


@torch.no_grad()
def predict_classifier(model, rows, tokenizer, args, device, task="detect",
                       type2id=None, batch_size=None) -> list[int]:
    ds = ClsDataset(rows, tokenizer, args.cls_max_len, task=task, type2id=type2id or {})
    loader = DataLoader(ds, batch_size=batch_size or args.cls_batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_cls)
    model.eval()
    out = []
    for batch in loader:
        logits, _ = model(batch["input_ids"].to(device),
                          batch["attention_mask"].to(device),
                          batch["token_type_ids"].to(device))
        out.extend(logits.argmax(-1).cpu().tolist())
    return out


def load_classifier(args, device, stage: str, num_labels: int, type2id=None):
    ckpt = torch.load(checkpoint_path(args, stage), map_location="cpu", weights_only=False)
    if type2id is not None and ckpt["type2id"] != type2id:
        raise ValueError("类型分类 checkpoint 的类型编号与当前 schema 不一致。")
    model = RobertaCls(str(args.roberta_path), num_labels, args.dropout).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model


def train_arg_experts(extract_rows, schema, tokenizer, args, device, logger=None) -> dict:
    """按类型分组，依次训练各类型的 Linear 论元专家。"""
    by_type = defaultdict(list)
    for row in extract_rows:
        event = row_event(row)
        if event:
            by_type[event["type"]].append(row)

    infos = {}
    all_start = time.perf_counter()
    for event_type in sorted(schema):
        rows = by_type.get(event_type, [])
        if not rows:
            print(f"  [arg:{event_type}] 没有训练样本，跳过", flush=True)
            continue
        scheme = build_typed_schema(schema, event_type)
        ds = BioDataset(rows, tokenizer, scheme, args.max_len, typed=True)
        loader = DataLoader(ds, batch_size=args.arg_batch_size, shuffle=True,
                            num_workers=0, collate_fn=collate_bio)
        model = build_arg_model(str(args.roberta_path), scheme, args).to(device)
        optimizer = AdamW(model.parameters(), lr=args.arg_lr, weight_decay=0.01)
        steps_per_epoch = (len(loader) + args.arg_grad_accum - 1) // args.arg_grad_accum
        total_steps = steps_per_epoch * args.arg_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
        start = time.perf_counter()
        for epoch in range(1, args.arg_epochs + 1):
            model.train()
            optimizer.zero_grad()
            epoch_start = time.perf_counter()
            total, n = 0.0, 0
            for step, batch in enumerate(loader):
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                tti = batch["token_type_ids"].to(device)
                labels = batch["labels"].to(device)
                loss = model(ids, mask, labels, tti)[1]
                group_start = (step // args.arg_grad_accum) * args.arg_grad_accum
                group_size = min(args.arg_grad_accum, len(loader) - group_start)
                (loss / group_size).backward()
                if (step + 1) % args.arg_grad_accum == 0 or step + 1 == len(loader):
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                total += float(loss.item())
                n += 1
                if logger is not None:
                    logger.log_step(epoch=epoch, step=step, loss=float(loss.item()),
                                    lr=scheduler.get_last_lr()[0],
                                    extra={"stage": f"arg:{event_type}"})
                if (step + 1) % args.log_every == 0 or step + 1 == len(loader):
                    print(f"  [arg:{event_type}] epoch {epoch}/{args.arg_epochs} "
                          f"step {step + 1}/{len(loader)} loss={total / n:.4f}", flush=True)
            if logger is not None:
                logger.log_epoch(epoch=epoch, train_loss=total / max(n, 1),
                                 elapsed=time.perf_counter() - epoch_start,
                                 extra={"stage": f"arg:{event_type}"})
        info = {"n_rows": len(rows), "num_labels": len(scheme["labels"]),
                "train_s": round(time.perf_counter() - start, 1)}
        save_checkpoint(model, checkpoint_path(args, f"arg_{event_type}"), args,
                        labels=scheme["labels"], etype=event_type, training_info=info)
        infos[event_type] = info
        del model, optimizer, scheduler, loss
        release_memory(device)
    infos["_total_train_s"] = round(time.perf_counter() - all_start, 1)
    return infos


def compose(dev_extract_rows, dev_detect_rows, tokenizer, args, device,
            oracle_type: bool = False) -> tuple[list[dict], dict]:
    """组合三段模型，并按输入顺序返回逐句事件和诊断信息。"""
    data_dir = Path(args.data_dir)
    schema = load_schema(data_dir / "event_schema.json")
    enums = load_enums(data_dir / "event_schema.json")

    det_model = load_classifier(args, device, "detector", 2)
    detected = predict_classifier(det_model, dev_extract_rows, tokenizer, args, device)
    del det_model
    release_memory(device)

    types_sorted = sorted(schema)
    type2id = {event_type: i for i, event_type in enumerate(types_sorted)}
    final_types = [(row_event(row) or {}).get("type") if oracle_type else None
                   for row in dev_extract_rows]
    # mock 数据含少量无事件句；类型给金标时，这些句子仍保留模型的类型预测。
    type_indices = [i for i, event_type in enumerate(final_types) if event_type is None]
    if type_indices:
        type_model = load_classifier(args, device, "type", len(types_sorted), type2id)
        type_rows = [dev_extract_rows[i] for i in type_indices]
        type_pred = predict_classifier(type_model, type_rows, tokenizer, args,
                                       device, task="type", type2id=type2id)
        for i, prediction in zip(type_indices, type_pred):
            final_types[i] = types_sorted[prediction]
        del type_model
        release_memory(device)

    preds = [None] * len(dev_extract_rows)
    idx_by_type = defaultdict(list)
    for i, (row, event_type) in enumerate(zip(dev_extract_rows, final_types)):
        if detected[i] == 0:
            preds[i] = make_pred_row(row, [], extra={"has_event": 0})
        else:
            idx_by_type[event_type].append(i)

    n_arg_truncated, n_missing_expert = 0, 0
    for event_type, idxs in idx_by_type.items():
        path = checkpoint_path(args, f"arg_{event_type}")
        if not path.exists():
            n_missing_expert += len(idxs)
            for i in idxs:
                preds[i] = make_pred_row(dev_extract_rows[i], [], extra={"has_event": 1})
            continue
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        scheme = build_typed_schema(schema, event_type)
        if ckpt["labels"] != scheme["labels"]:
            raise ValueError(f"{event_type} 专家 checkpoint 的标签表与当前 schema 不一致。")
        saved_args = ckpt["args"]
        expert_args = argparse.Namespace(arg_layers=saved_args["arg_layers"],
                                         dropout=saved_args.get("dropout", args.dropout))
        model = build_arg_model(str(args.roberta_path), scheme, expert_args).to(device)
        model.load_state_dict(ckpt["state_dict"])
        del ckpt
        rows = [dev_extract_rows[i] for i in idxs]
        ds = BioDataset(rows, tokenizer, scheme, args.max_len, typed=True)
        n_arg_truncated += ds.n_truncated
        loader = DataLoader(ds, batch_size=args.arg_batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_bio)
        tags, _ = predict_char_tags(model, loader, scheme["id2label"], device)
        for i, row, sequence in zip(idxs, rows, tags):
            event = spans_to_event(decode_bio_spans(sequence), row["text"],
                                   typed_type=event_type, enums=enums)
            preds[i] = make_pred_row(row, [event] if event else [], extra={"has_event": 1})
        del model
        release_memory(device)

    typed_pairs = [(row_event(row)["type"], prediction)
                   for row, prediction in zip(dev_extract_rows, final_types) if row_event(row)]
    n_type_wrong = sum(gold != prediction for gold, prediction in typed_pairs)
    diag = {
        "n_detect_dropped": sum(prediction == 0 for prediction in detected),
        "n_type_wrong": n_type_wrong,
        "n_type_evaluated": len(typed_pairs),
        "type_accuracy_on_dev_extract": round(1 - n_type_wrong / len(typed_pairs), 4)
                                        if typed_pairs else None,
        "n_arg_truncated": n_arg_truncated,
        "n_missing_expert_samples": n_missing_expert,
    }
    return preds, diag


def main():
    args = parse_args()
    args.roberta_path = str(resolve_model_path(args.roberta_path))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    args.device = str(device)
    if args.cls_batch_size is None:
        args.cls_batch_size = 4 if device.type == "mps" else 64
    if args.arg_batch_size is None:
        args.arg_batch_size = 4 if device.type == "mps" else 32
    if args.cls_grad_accum is None:
        args.cls_grad_accum = 16 if device.type == "mps" else 1
    if args.arg_grad_accum is None:
        # 专家按类型分组后样本较少，默认每个 batch 更新，避免更新次数不足。
        args.arg_grad_accum = 1
    data_dir = Path(args.data_dir) if args.data_dir else (MOCK_DIR if args.mock else DATA_DIR)
    data_dir = data_dir.resolve()
    args.data_dir = str(data_dir)
    print(f"设备：{device} | 数据：{data_dir}")
    print(f"分类 batch={args.cls_batch_size}，累积={args.cls_grad_accum}；"
          f"专家 batch={args.arg_batch_size}，累积={args.arg_grad_accum}")

    schema = load_schema(data_dir / "event_schema.json")
    type2id = {event_type: i for i, event_type in enumerate(sorted(schema))}
    tokenizer = load_tokenizer(args.roberta_path)
    train_extract = load_extract_split("train", data_dir)
    dev_extract = load_extract_split("dev", data_dir)
    train_detect = load_detect_split("train", data_dir)
    dev_detect = load_detect_split("dev", data_dir)
    clean_path = data_dir / "dev_detect_clean.jsonl"
    dev_detect_clean = load_jsonl(clean_path) if clean_path.exists() else []
    if not train_extract or not train_detect or not dev_extract or not dev_detect:
        raise ValueError("训练和评测的 extract/detect 数据应包含样本。")
    train_type = [row for row in train_extract if row_event(row)]
    if not train_type:
        raise ValueError("事件类型分类训练集应包含有事件的样本。")
    print(f"train_extract {len(train_extract)} / dev_extract {len(dev_extract)} / "
          f"train_detect {len(train_detect)} / dev_detect {len(dev_detect)}")

    if args.skip_train or args.only_experts:
        required = [checkpoint_path(args, stage) for stage in ["detector", "type"]]
        if args.skip_train:
            required += [checkpoint_path(args, f"arg_{event_type}")
                         for event_type in {row_event(row)["type"] for row in train_type}]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("复用模式缺少 checkpoint：\n" + "\n".join(missing))

    if args.skip_train:
        logger = EvalLogger(method=f"{args.run_name}_eval", args=vars(args), root=ROOT,
                            notes="RoBERTa 流水线 checkpoint 组合评测")
    else:
        log_name = args.run_name + ("_experts" if args.only_experts else "")
        logger = TrainLogger(method=log_name, args=args, root=ROOT,
                             notes="RoBERTa 检测器、类型分类器与类型条件论元专家")
    all_start = time.perf_counter()

    det_train_rows = train_detect
    if 0 < args.detect_limit < len(train_detect):
        pos = [row for row in train_detect if int(row.get("n_events", 0)) > 0]
        neg = [row for row in train_detect if int(row.get("n_events", 0)) <= 0]
        rng = random.Random(args.seed)
        rng.shuffle(pos)
        rng.shuffle(neg)
        n_pos = round(args.detect_limit * len(pos) / len(train_detect))
        if pos and neg and args.detect_limit >= 2:
            n_pos = max(1, min(args.detect_limit - 1, n_pos))
        det_train_rows = pos[:n_pos] + neg[:args.detect_limit - n_pos]
        rng.shuffle(det_train_rows)
        print(f"检测器分层抽样 {len(det_train_rows)} 条，正样本 {n_pos} 条")

    det_train_s, type_train_s = 0.0, 0.0
    if not args.skip_train and not args.only_experts:
        print("\n(a) 训练事件检测器", flush=True)
        trained, info = train_classifier(det_train_rows, None, 2, tokenizer, args, device,
                                         checkpoint_path(args, "detector").name,
                                         task="detect", logger=logger)
        det_train_s = info["train_s"]
        del trained
        release_memory(device)

    det_model = load_classifier(args, device, "detector", 2)
    det_start = time.perf_counter()
    det_pred = predict_classifier(det_model, dev_detect, tokenizer, args, device)
    det_ms = (time.perf_counter() - det_start) * 1000 / len(dev_detect)
    del det_model
    release_memory(device)
    rows_pred = [{"id": row["id"], "sent_id": row["sent_id"], "has_event": prediction}
                 for row, prediction in zip(dev_detect, det_pred)]
    det_full = detection_metrics(dev_detect, rows_pred)
    det_clean = None
    if dev_detect_clean:
        keys = {(row["id"], row["sent_id"]) for row in dev_detect_clean}
        subset = [row for row in rows_pred if (row["id"], row["sent_id"]) in keys]
        det_clean = detection_metrics(dev_detect_clean, subset)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    detector_result = {
        "method": f"{args.run_name}_detector",
        "dev_detect_full": det_full, "dev_detect_clean": det_clean,
        "meta": {"display_name": "RoBERTa 事件检测器", "train_s": det_train_s,
                 "ms_per_sentence": round(det_ms, 2), "detect_train_n": len(det_train_rows),
                 "dev_detect_n": len(dev_detect), "dev_detect_clean_n": len(dev_detect_clean)},
    }
    (METRIC_DIR / f"{args.run_name}_detector.json").write_text(
        json.dumps(detector_result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"检测器 dev F1={det_full['f1']:.4f}，accuracy={det_full['accuracy']:.4f}")

    if not args.skip_train and not args.only_experts:
        print("\n(b) 训练事件类型分类器", flush=True)
        trained, info = train_classifier(train_type, type2id, len(schema), tokenizer,
                                         args, device, checkpoint_path(args, "type").name,
                                         task="type", logger=logger)
        type_train_s = info["train_s"]
        del trained
        release_memory(device)

    if args.skip_train:
        arg_infos = {event_type: {"reused": True} for event_type in schema
                     if checkpoint_path(args, f"arg_{event_type}").is_file()}
        arg_infos["_total_train_s"] = 0.0
    else:
        print(f"\n(c) 依次训练 {len(schema)} 个类型条件论元专家", flush=True)
        arg_infos = train_arg_experts(train_extract, schema, tokenizer, args, device, logger)

    print("\n组合三段模型评测 dev_extract", flush=True)
    with Timer() as timer:
        preds, diag = compose(dev_extract, dev_detect, tokenizer, args, device)
    latency = timer.summary(len(dev_extract))
    type_result = {
        "method": f"{args.run_name}_type", "num_samples": diag["n_type_evaluated"],
        "accuracy": diag["type_accuracy_on_dev_extract"], "n_wrong": diag["n_type_wrong"],
        "type2id": type2id, "train_s": type_train_s,
    }
    (METRIC_DIR / f"{args.run_name}_type.json").write_text(
        json.dumps(type_result, ensure_ascii=False, indent=2), encoding="utf-8")
    training_s = det_train_s + type_train_s + arg_infos["_total_train_s"]
    meta = {
        "display_name": "RoBERTa 流水线（检测→类型→类型条件论元）",
        "train_s": round(training_s, 1), "ms_per_sentence": latency["ms_per_sentence"],
        "peak_gpu_mb": latency["peak_gpu_mb"] if device.type == "cuda" else None,
        "device": str(device), "roberta_path": args.roberta_path, "data_dir": args.data_dir,
        "cls_max_len": args.cls_max_len, "max_len": args.max_len,
        "cls_epochs": args.cls_epochs, "arg_epochs": args.arg_epochs,
        "arg_layers": args.arg_layers, "pipeline_stages": 3,
        "num_models": 2 + sum(not key.startswith("_") for key in arg_infos),
        "diag": diag, "arg_experts": arg_infos, "oracle_type": False,
        "latency_note": "组合推理计时包含模型加载",
        "training_mode": "eval_only" if args.skip_train else ("only_experts" if args.only_experts else "all"),
    }
    # 检测器指标在独立的 dev_detect 上报告，事件抽取在 dev_extract 上报告。
    result = evaluate_predictions(args.run_name, preds, meta=meta,
                                  gold_path=data_dir / "dev_extract.jsonl")

    print("\n类型给金标，评估检测和论元抽取阶段", flush=True)
    with Timer() as oracle_timer:
        oracle_preds, oracle_diag = compose(dev_extract, dev_detect, tokenizer, args,
                                            device, oracle_type=True)
    oracle_meta = dict(meta)
    oracle_meta.update({"display_name": "RoBERTa 流水线（类型给金标）",
                        "oracle_type": True, "diag": oracle_diag,
                        "ms_per_sentence": oracle_timer.summary(len(dev_extract))["ms_per_sentence"]})
    oracle_result = evaluate_predictions(f"{args.run_name}_oracle_type", oracle_preds,
                                         meta=oracle_meta, gold_path=data_dir / "dev_extract.jsonl")
    print(f"\n端到端 argument F1={result['argument']['f1']:.4f}；"
          f"类型给金标 argument F1={oracle_result['argument']['f1']:.4f}")
    metrics = {key: result[key]["f1"] for key in ("trigger", "argument", "event_exact")}
    total_s = time.perf_counter() - all_start
    if args.skip_train:
        logger.finalize(metrics=metrics, runtime_s=total_s, diag=diag,
                        pipeline_stages=3, num_models=meta["num_models"])
    else:
        logger.finalize(total_s=total_s, dev_metrics=metrics, diag=diag,
                        train_s=round(training_s, 1), pipeline_stages=3,
                        num_models=meta["num_models"])


def parse_args():
    ap = argparse.ArgumentParser(description="RoBERTa 流水线事件抽取")
    ap.add_argument("--roberta_path", default="")
    ap.add_argument("--cls_epochs", type=int, default=2)
    ap.add_argument("--detect_limit", type=int, default=0, help="检测训练样本上限，0 使用全量")
    ap.add_argument("--cls_lr", type=float, default=2e-5)
    ap.add_argument("--cls_batch_size", type=int, default=None)
    ap.add_argument("--cls_grad_accum", type=int, default=None)
    ap.add_argument("--cls_max_len", type=int, default=192)
    ap.add_argument("--arg_epochs", type=int, default=3)
    ap.add_argument("--arg_lr", type=float, default=3e-5)
    ap.add_argument("--arg_batch_size", type=int, default=None)
    ap.add_argument("--arg_grad_accum", type=int, default=None)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--arg_layers", type=int, default=4, help="专家保留的编码器层数，0 使用全部层")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--device", default="", help="cuda / mps / cpu；留空自动检测")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=10)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--skip_train", action="store_true", help="加载 checkpoint 组合评测")
    mode.add_argument("--only_experts", action="store_true", help="复用分类器，重训论元专家")
    ap.add_argument("--run_name", default="", help="模型和评测文件的实验前缀")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--data_dir", default="")
    args = ap.parse_args()
    values = [args.cls_epochs, args.arg_epochs, args.cls_max_len, args.max_len, args.log_every,
              args.cls_batch_size, args.arg_batch_size, args.cls_grad_accum, args.arg_grad_accum]
    if any(value is not None and value < 1 for value in values):
        ap.error("轮数、长度、批量、梯度累积和日志间隔必须大于 0")
    if args.detect_limit < 0 or args.arg_layers < 0:
        ap.error("detect_limit 和 arg_layers 应为非负整数")
    if not args.run_name:
        args.run_name = "roberta_pipeline" + ("_mock" if args.mock else "")
    return args


if __name__ == "__main__":
    main()
