"""
从 RoBERTa + Linear / CRF checkpoint 复评数据集并输出统一评测指标。

使用方式：
  python roberta/evaluate.py
  python roberta/evaluate.py --ckpt outputs/checkpoints/roberta_crf_256.pt
  python roberta/evaluate.py --hard_constraint --device mps
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]   # 项目根目录（方法目录的上一级）
sys.path.insert(0, str(ROOT / "common"))     # 公共模块：event_data / crf / eval_utils / models / log_utils

from eval_utils import CKPT_DIR, Timer, evaluate_predictions
from log_utils import EvalLogger
from event_data import (BioDataset, DATA_DIR, ROOT, build_bio_constraint_mask,
                        build_start_constraint_mask, collate_bio, decode_bio_spans,
                        illegal_bio_stats, load_enums, load_extract_split,
                        make_pred_row, spans_to_event)
from model_paths import MOCK_DIR, resolve_model_path

def main():
    ap = argparse.ArgumentParser(description="从 checkpoint 复评（dev 推理 + 统一评测）")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--run_name", default="")
    ap.add_argument("--data_dir", default="")
    ap.add_argument("--roberta_path", default="", help="覆盖 checkpoint 中的预训练模型路径")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--batch_size", type=int, default=None,
                    help="评测批量；默认 MPS=4，CUDA/CPU=32")
    ap.add_argument("--hard_constraint", action="store_true",
                    help="加载 CRF checkpoint 后施加 BIO 解码硬约束")
    ap.add_argument("--train_sample", type=int, default=0,
                    help="训练集抽样多少条来估非法率（0=不算）")
    ap.add_argument("--device", default="", help="cuda / mps / cpu；留空自动检测")
    args = ap.parse_args()

    from models import build_model, load_tokenizer, predict_char_tags, select_device

    device = select_device(args.device)
    if args.batch_size is None:
        args.batch_size = 4 if device.type == "mps" else 32
    if args.batch_size < 1:
        ap.error("batch_size 必须大于 0")
    ckpt_path = Path(args.ckpt) if args.ckpt else CKPT_DIR / "roberta_crf_256.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    label_scheme = ckpt["label_scheme"]
    id2label = ckpt["id2label"]
    train_args = ckpt["args"]
    if args.hard_constraint and train_args.get("decoder") != "crf":
        ap.error("解码硬约束需要使用 CRF checkpoint")

    class _A:
        pass

    a = _A()
    for k, v in train_args.items():
        setattr(a, k, v)
    a.roberta_path = str(resolve_model_path(args.roberta_path or train_args.get("roberta_path")))

    model = build_model(a, label_scheme).to(device)
    model.load_state_dict(ckpt["state_dict"])
    if args.hard_constraint:
        # checkpoint 的 buffer 已加载完毕，此处重新设置才会实际生效。
        model.crf.constraint = build_bio_constraint_mask(label_scheme).to(device)
        model.crf.start_constraint = build_start_constraint_mask(label_scheme).to(device)
        print(f"BIO硬约束已装载：禁止 {int((~model.crf.constraint).sum())} 个非法转移对、"
              f"{int((~model.crf.start_constraint).sum())} 个非法起始标签")
    suffix = "_hard" if args.hard_constraint else ""
    name = args.run_name or ckpt_path.stem + suffix
    eval_logger = EvalLogger(method=name, args=vars(args), root=ROOT,
                             notes=f"从 checkpoint 复评，解码硬约束={args.hard_constraint}")
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"加载 {ckpt_path.name}（epoch {ckpt.get('epoch')}，"
          f"内层验证 spanF1={ckpt.get('inner_val_span_f1', 0):.4f}，"
          f"解码器={train_args.get('decoder')}）")

    saved_data_dir = args.data_dir or train_args.get("data_dir")
    data_dir = Path(saved_data_dir) if saved_data_dir else DATA_DIR
    if not saved_data_dir and train_args.get("mock"):
        data_dir = MOCK_DIR
    tokenizer = load_tokenizer(a.roberta_path)
    rows = load_extract_split(args.split, data_dir)
    enums = load_enums(data_dir / "event_schema.json")
    ds = BioDataset(rows, tokenizer, label_scheme, train_args.get("max_len", 256))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
                        collate_fn=collate_bio)

    t0 = time.perf_counter()
    with Timer() as timer:
        tags, prediction_info = predict_char_tags(
            model, loader, id2label, device, collect_token_stats=True,
        )
    illegal = illegal_bio_stats(tags)

    preds = []
    for row, tg in zip(rows, tags):
        ev = spans_to_event(decode_bio_spans(tg), row["text"], enums=enums)
        preds.append(make_pred_row(row, [ev] if ev else []))
    lat = timer.summary(len(rows))

    meta = {k: v for k, v in train_args.items()
            if k in ("decoder", "roberta_layers", "max_len", "epochs", "lr", "batch_size",
                     "grad_accum", "head_lr_mult", "seed")}
    num_layers = model.encoder.roberta.config.num_hidden_layers
    meta.update({
        "display_name": f"RoBERTa-{num_layers}L+"
                        f"{str(train_args.get('decoder', 'linear')).upper()}{suffix}",
        "train_s": 0,      # 训练已在别处完成，这里只复评
        "params_m": round(n_params / 1e6, 1),
        "ms_per_sentence": lat["ms_per_sentence"],
        "best_epoch": ckpt.get("epoch"),
        "inner_val_span_f1": round(float(ckpt.get("inner_val_span_f1", 0)), 4),
        "illegal_dev": illegal,
        "illegal_token": prediction_info["illegal_token"],
        "hard_constraint": args.hard_constraint,
        "constraint_stage": "decode_only" if args.hard_constraint else "checkpoint",
        "base_ckpt": str(ckpt_path),
        "eval_batch_size": args.batch_size,
        "n_truncated": ds.n_truncated,
        "note": "由 roberta/evaluate.py 从 checkpoint 复评",
        "roberta_layers": num_layers,
        "roberta_path": a.roberta_path,
        "data_dir": str(data_dir),
    })

    # 训练集抽样上的非法率（便宜版对照，代替对全训练集做一次慢推理）
    if args.train_sample > 0:
        tr_rows = load_extract_split("train", data_dir)
        random.Random(train_args.get("seed", 42)).shuffle(tr_rows)
        sub = tr_rows[:args.train_sample]
        tr_ds = BioDataset(sub, tokenizer, label_scheme, train_args.get("max_len", 256))
        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=0, collate_fn=collate_bio)
        tr_tags, _ = predict_char_tags(model, tr_loader, id2label, device)
        meta["illegal_train_pred_sample"] = illegal_bio_stats(tr_tags)
        meta["illegal_train_pred_sample_n"] = len(sub)
        print(f"训练集抽样 {len(sub)} 句的非法率："
              f"{meta['illegal_train_pred_sample']['illegal_seq_rate']:.1%}")

    res = evaluate_predictions(name, preds, meta=meta,
                              gold_path=data_dir / f"{args.split}_extract.jsonl")
    print(f"\n非法统计（dev）：含非法序列 {illegal['illegal_seqs']} 条"
          f"（{illegal['illegal_seq_rate']:.1%}），"
          f"非法转移 {illegal['illegal_transition']} 处 / 非法开头 {illegal['illegal_start']} 处")
    token_illegal = prediction_info["illegal_token"]
    print(f"token级非法统计：{token_illegal['total_illegal']} 处，"
          f"含非法序列 {token_illegal['illegal_seqs']} 条")
    eval_logger.finalize(metrics={k: res[k]["f1"] for k in ("trigger", "argument", "event_exact")},
                        type_macro_f1=res.get("type_macro_f1"), illegal_dev=illegal,
                        inference_latency_ms=lat["ms_per_sentence"], ckpt=str(ckpt_path),
                        illegal_token=token_illegal, hard_constraint=args.hard_constraint)

    print(f"推理耗时 {time.perf_counter() - t0:.0f}s（{lat['ms_per_sentence']} ms/句）")

if __name__ == "__main__":
    main()
