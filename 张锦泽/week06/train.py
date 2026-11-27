"""基础中文文本分类对比：TF-IDF 基线与两种预训练模型微调。"""

import argparse
import csv
import gc
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from common import (ROOT, environment_info, load_splits, make_loader, metrics_for,
                    neural_predictions, resolve_device, save_evaluation, write_csv, write_json)

METHODS = ("tfidf_lr", "tfidf_nb", "bert", "roberta", "bert_lr5e5")
RESULT_FIELDS = ["method", "accuracy", "precision_macro", "recall_macro", "f1_macro",
                 "training_seconds", "validation_f1_macro", "best_epoch", "device", "seed",
                 "n_train", "n_val", "n_test", "data_sha256", "model_source", "learning_rate"]


def train_baseline(method, splits, labels, output_dir, args):
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline

    classifier = (LogisticRegression(C=1.0, max_iter=1000, random_state=args.seed)
                  if method == "tfidf_lr" else MultinomialNB(alpha=1.0))
    model = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(1, 2), sublinear_tf=True, max_features=20000)),
        ("classifier", classifier),
    ])
    started = time.perf_counter()
    model.fit(*splits["train"])
    validation = metrics_for(splits["val"][1], model.predict(splits["val"][0]), labels)
    joblib.dump(model, output_dir / "model.joblib", compress=3)
    elapsed = time.perf_counter() - started
    return model.predict(splits["test"][0]).tolist(), {
        "kind": "sklearn", "training_seconds": elapsed, "validation_f1_macro": validation["f1_macro"],
        "best_epoch": None, "device": "cpu", "model_source": "TF-IDF(char 1-2gram)",
        "learning_rate": None,
    }


def train_transformer(method, splits, labels, output_dir, args):
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = resolve_device(args.device)
    source = args.roberta_model if method == "roberta" else args.bert_model
    learning_rate = 5e-5 if method == "bert_lr5e5" else 2e-5
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        source, num_labels=len(labels), id2label=dict(enumerate(labels)),
        label2id={name: index for index, name in enumerate(labels)}, trust_remote_code=False,
    ).to(device)
    max_supported = getattr(model.config, "max_position_embeddings", args.max_length)
    if args.max_length > max_supported:
        raise ValueError(f"--max-length 超过模型位置上限 {max_supported}")
    revision = getattr(model.config, "_commit_hash", None)
    # 计时排除下载和基础模型加载，包含分词、训练、验证及最佳检查点写入。
    started = time.perf_counter()
    train_loader = make_loader(*splits["train"], tokenizer, args.max_length, args.batch_size, True, args.seed)
    val_loader = make_loader(*splits["val"], tokenizer, args.max_length, args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    best_f1, best_epoch, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_examples = 0.0, 0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise ValueError("训练 loss 非有限值，请检查数据和参数")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            size = batch["labels"].shape[0]
            total_loss += loss.item() * size
            total_examples += size
        val_metrics = metrics_for(splits["val"][1], neural_predictions(model, val_loader, device), labels)
        history.append({"epoch": epoch, "train_loss": total_loss / total_examples,
                        "validation_f1_macro": val_metrics["f1_macro"]})
        print(f"{method} epoch={epoch} loss={total_loss / total_examples:.4f} val_f1={val_metrics['f1_macro']:.4f}", flush=True)
        # 相同分数保留更早的轮次；这里不读取测试集指标。
        if val_metrics["f1_macro"] > best_f1:
            best_f1, best_epoch = val_metrics["f1_macro"], epoch
            model.save_pretrained(output_dir, safe_serialization=True)
            tokenizer.save_pretrained(output_dir)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    write_csv(output_dir / "history.csv", ["epoch", "train_loss", "validation_f1_macro"], history)
    # 用验证集选出的最佳权重执行一次最终测试，而非使用最后一轮权重。
    del optimizer, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model = AutoModelForSequenceClassification.from_pretrained(output_dir, local_files_only=True).to(device)
    test_loader = make_loader(*splits["test"], tokenizer, args.max_length, args.batch_size)
    predicted = neural_predictions(model, test_loader, device)
    return predicted, {
        "kind": "transformer", "training_seconds": elapsed, "validation_f1_macro": best_f1,
        "best_epoch": best_epoch, "device": str(device), "model_source": source,
        "model_revision": revision, "learning_rate": learning_rate,
        "epochs": args.epochs, "batch_size": args.batch_size, "max_length": args.max_length,
        "weight_decay": 0.01,
    }


def update_comparison(path, row):
    rows = []
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = [old for old in csv.DictReader(handle) if old["method"] != row["method"]]
    rows.append(row)
    rows.sort(key=lambda item: METHODS.index(item["method"]))
    write_csv(path, RESULT_FIELDS, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("baselines", "all") + METHODS, default="baselines")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--bert-model", default="google-bert/bert-base-chinese")
    parser.add_argument("--roberta-model", default="hfl/chinese-roberta-wwm-ext")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1 or args.max_length < 4 or not 0 <= args.seed < 2**32:
        parser.error("epochs、batch-size 必须为正数；max-length 至少为 4；seed 范围为 [0, 2**32)")
    try:
        splits, labels, counts, digest = load_splits(args.data_dir)
        methods = METHODS if args.method == "all" else (METHODS[:2] if args.method == "baselines" else (args.method,))
        settings = {"data_sha256": digest, "labels": labels, "seed": args.seed, "epochs": args.epochs,
                    "batch_size": args.batch_size, "max_length": args.max_length,
                    "bert_model": args.bert_model, "roberta_model": args.roberta_model}
        manifest = args.output_dir / "experiment.json"
        if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != settings:
            raise ValueError("该输出目录包含不同数据或参数的实验，请用 --output-dir 指定新目录")
        write_json(manifest, settings)
        write_json(args.output_dir / "data_summary.json", counts)
        for method in methods:
            print(f"\n开始实验：{method}", flush=True)
            output_dir = args.output_dir / method
            output_dir.mkdir(parents=True, exist_ok=True)
            if method.startswith("tfidf_"):
                predicted, metadata = train_baseline(method, splits, labels, output_dir, args)
            else:
                predicted, metadata = train_transformer(method, splits, labels, output_dir, args)
            metadata.update({"method": method, "labels": labels, "seed": args.seed, "data_sha256": digest,
                             "trained_at_utc": datetime.now(timezone.utc).isoformat(),
                             "environment": environment_info(),
                             **{f"n_{name}": len(values[0]) for name, values in splits.items()}})
            metrics = save_evaluation(output_dir, *splits["test"], predicted, labels)
            write_json(output_dir / "metadata.json", metadata)
            values = {**metadata, **metrics}
            row = {field: values.get(field) for field in RESULT_FIELDS}
            update_comparison(args.output_dir / "comparison.csv", row)
            print(json.dumps({"method": method, **metrics}, ensure_ascii=False, indent=2), flush=True)
        print(f"对比结果已保存：{args.output_dir / 'comparison.csv'}")
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(1, f"实验失败：{exc}\n依赖见 requirements-lite.txt / requirements.txt；此前成功的结果会保留。\n")


if __name__ == "__main__":
    main()
