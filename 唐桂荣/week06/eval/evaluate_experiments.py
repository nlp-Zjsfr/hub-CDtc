"""校验消融结果、按验证集锁定最佳配置，并在去重测试集上统一评测。"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from bert.experiments import EXPERIMENTS, MODEL_NAMES
from bert.run_experiments import OUTPUTS, PROJECT_DIR, read_completed
from bert.train_bert import (
    HEAD_LR_MULTIPLIER, BertClassifier, encode_samples, evaluate, write_report,
)
from common.data import SPLIT_SEED, VAL_FRACTION, load_splits
from common.metrics import classification_report


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_predictions(report, samples, labels):
    if tuple(report["labels"]) != labels:
        raise ValueError("结果使用了不同的类别表")
    rows = report["predictions"]
    if [(row["idx"], row["label"]) for row in rows] != [
        (sample["idx"], sample["label"]) for sample in samples
    ]:
        raise ValueError("结果使用了不同的评测样本")
    metrics = classification_report(
        [row["label"] for row in rows], [row["predicted_label"] for row in rows], labels,
    )
    if metrics != report["metrics"]:
        raise ValueError("逐条预测与汇总指标不匹配")


def config_with_defaults(config):
    # 兼容此前 exp1 保存的结果，其中三个扩展开关未写进配置。
    return {"fgm_eps": 0.0, "label_smoothing": 0.0, "focal_gamma": 0.0, **config}


def load_validation(splits):
    baseline_tfidf = read_json(OUTPUTS / "tfidf_val.json")
    all_reports = {}
    baseline_config = None
    for model in MODEL_NAMES:
        all_reports[model] = {}
        model_root = OUTPUTS / "bert" / model
        baseline = read_completed(model_root / "exp1", model, "exp1")
        if baseline is None:
            raise ValueError(f"{model} 缺少完整的 exp1")
        current_config = config_with_defaults(baseline["config"])
        if baseline_config is None:
            baseline_config = current_config
        if current_config != baseline_config:
            raise ValueError("两套模型的 exp1 训练配置不同")
        for exp_id, experiment in EXPERIMENTS.items():
            report = read_completed(model_root / exp_id, model, exp_id)
            if report is None:
                raise ValueError(f"{model} 缺少完整的 {exp_id}")
            expected_config = {
                **current_config,
                "pool": experiment.pool,
                "max_length": experiment.max_length,
                "encoder_lr": experiment.encoder_lr,
                "head_lr": experiment.encoder_lr * HEAD_LR_MULTIPLIER,
                "fgm_eps": experiment.fgm_eps,
                "label_smoothing": experiment.label_smoothing,
                "focal_gamma": experiment.focal_gamma,
                "loss": "weighted_focal" if experiment.focal_gamma else "weighted_cross_entropy",
            }
            if config_with_defaults(report["config"]) != expected_config:
                raise ValueError(f"{model} {exp_id} 修改了约定配置以外的参数")
            validate_predictions(report, splits.val, splits.labels)
            best_epoch = max(report["history"], key=lambda epoch: epoch["macro_f1"])
            if report["best_epoch"] != best_epoch["epoch"] or (
                report["metrics"]["macro_f1"] != best_epoch["macro_f1"]
            ):
                raise ValueError(f"{model} {exp_id} 保存的轮次与验证集最高分不一致")
            all_reports[model][exp_id] = report
    validate_predictions(baseline_tfidf, splits.val, splits.labels)
    return all_reports, baseline_tfidf


def juice_errors(report):
    labels = tuple(report["labels"])
    cold = labels.index("冷藏即饮果汁")
    ambient = labels.index("非冷藏即饮果汁")
    matrix = report["metrics"]["confusion_matrix"]
    return matrix[cold][ambient], matrix[ambient][cold]


def validation_rows(all_reports):
    base_scores = {model: runs["exp1"]["metrics"]["macro_f1"] for model, runs in all_reports.items()}
    rows = {}
    for exp_id, experiment in EXPERIMENTS.items():
        points = {}
        for model in MODEL_NAMES:
            report = all_reports[model][exp_id]
            macro = report["metrics"]["macro_f1"]
            points[model] = {
                "macro_f1": macro,
                "accuracy": report["metrics"]["accuracy"],
                "change_from_exp1_pp": (macro - base_scores[model]) * 100,
                "best_epoch": report["best_epoch"],
                "training_seconds": sum(epoch["train_seconds"] for epoch in report["history"]),
                "juice_errors": juice_errors(report),
            }
        bert, roberta = (points[model] for model in MODEL_NAMES)
        rows[exp_id] = {
            "description": experiment.description,
            "models": points,
            "roberta_minus_bert_pp": (roberta["macro_f1"] - bert["macro_f1"]) * 100,
        }
    return rows


def evaluate_checkpoint(report, samples, labels, device):
    model_name = report["model"]
    exp_id = report["experiment"]
    checkpoint_path = OUTPUTS / "bert" / model_name / exp_id / "best.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu", mmap=True)
    if (
        checkpoint["model"] != model_name or checkpoint["experiment"] != exp_id
        or checkpoint["epoch"] != report["best_epoch"] or tuple(checkpoint["labels"]) != labels
        or checkpoint["config"] != report["config"]
    ):
        raise ValueError(f"最佳权重与验证集记录不匹配：{checkpoint_path}")
    config = report["config"]
    model_dir = PROJECT_DIR / "model" / model_name
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = BertClassifier(model_dir, len(labels), config["pool"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    data = encode_samples(samples, tokenizer, config["max_length"])
    loader = DataLoader(data, batch_size=config["batch_size"])
    start = perf_counter()
    actual, predicted = evaluate(model, loader, device, config["precision"] == "bf16")
    predict_seconds = perf_counter() - start
    return {
        "model": model_name, "experiment": exp_id, "best_epoch": report["best_epoch"],
        "config": config, "split": "test", "device": device.type, "labels": labels,
        "metrics": classification_report(actual, predicted, labels),
        "predict_seconds": predict_seconds,
        "predictions": [
            {"idx": sample["idx"], "label": label, "predicted_label": prediction}
            for sample, label, prediction in zip(samples, actual, predicted, strict=True)
        ],
    }


def update_summary(rows, test_results, val_n, test_n):
    path = OUTPUTS / "summary.md"
    heading = "## 消融实验和最终测试"
    original = path.read_text(encoding="utf-8")
    prefix = original.partition(heading)[0].rstrip()
    lines = [
        heading, "",
        f"九组实验共用 {val_n:,} 条验证样本。Δ 表示相对各自 exp1 的 Macro-F1 变化，单位为百分点。",
        "",
        "| 实验 | 配置 | BERT Macro-F1 | BERT Δ | RoBERTa Macro-F1 | RoBERTa Δ | RoBERTa − BERT |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for exp_id, row in rows.items():
        bert, roberta = (row["models"][model] for model in MODEL_NAMES)
        lines.append(
            f"| {exp_id} | {row['description']} | {bert['macro_f1']:.4f} |"
            f" {bert['change_from_exp1_pp']:+.2f} | {roberta['macro_f1']:.4f} |"
            f" {roberta['change_from_exp1_pp']:+.2f} | {row['roberta_minus_bert_pp']:+.2f} |"
        )
    lines += [
        "", "### 输入长度与 FGM 的训练耗时", "",
        "耗时为在本机 MPS 上三轮训练的合计分钟数。",
        "",
        "| 模型 | exp1 | exp6 | exp6 耗时变化 | exp7 | exp7 耗时变化 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, model in zip(("BERT", "RoBERTa"), MODEL_NAMES, strict=True):
        base, shorter, fgm = (
            rows[exp]["models"][model]["training_seconds"] for exp in ("exp1", "exp6", "exp7")
        )
        lines.append(
            f"| {label} | {base / 60:.1f} | {shorter / 60:.1f} |"
            f" {(shorter / base - 1):+.1%} | {fgm / 60:.1f} | {(fgm / base - 1):+.1%} |"
        )
    lines += [
        "", "### 去重测试集", "",
        f"两套模型各用验证集 Macro-F1 最高的配置，与 TF-IDF 在同一批 {test_n:,} 条样本上比较。",
        "",
        "| 方法 | 配置 | Macro-F1 | 准确率 | 冷藏判为非冷藏 | 非冷藏判为冷藏 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for label, report in test_results.items():
        metrics = report["metrics"]
        cold, ambient = juice_errors(report)
        method = report.get("experiment", "字符 1–2 gram")
        lines.append(
            f"| {label} | {method} | {metrics['macro_f1']:.4f} | {metrics['accuracy']:.2%} |"
            f" {cold} | {ambient} |"
        )
    by_score = sorted(test_results, key=lambda label: test_results[label]["metrics"]["macro_f1"], reverse=True)
    leading_score = test_results[by_score[0]]["metrics"]["macro_f1"]
    leaders = [label for label in by_score if test_results[label]["metrics"]["macro_f1"] == leading_score]
    lines += [
        "",
        f"本次测试集 Macro-F1 最高的是{'、'.join(leaders)}，得分为 {leading_score:.4f}。"
        " [完整验证集结果](eval/validation.json) · [测试集对比](eval/test_comparison.json)",
    ]
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text(prefix + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description="汇总消融实验并运行最终测试")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()

    splits = load_splits()
    reports, tfidf_val = load_validation(splits)
    winners = {
        model: max(runs.values(), key=lambda report: report["metrics"]["macro_f1"])
        for model, runs in reports.items()
    }
    rows = validation_rows(reports)
    out = OUTPUTS / "eval"
    out.mkdir(exist_ok=True)
    write_report(out / "validation.json", {
        "split": "val", "n": len(splits.val), "split_seed": SPLIT_SEED, "val_fraction": VAL_FRACTION,
        "experiments": rows, "tfidf_macro_f1": tfidf_val["metrics"]["macro_f1"],
    })
    selection = {
        model: {
            "experiment": report["experiment"], "epoch": report["best_epoch"],
            "val_macro_f1": report["metrics"]["macro_f1"],
        }
        for model, report in winners.items()
    }
    write_report(out / "selection.json", selection)
    print(f"按验证集锁定配置：{selection}", flush=True)

    subprocess.run(
        [sys.executable, "-B", "-m", "tfidf.classify_tfidf", "--split", "test"],
        cwd=PROJECT_DIR, check=True,
    )
    tfidf_test = read_json(OUTPUTS / "tfidf_test.json")
    if tfidf_test["split"] != "test" or tfidf_test["config"] != tfidf_val["config"]:
        raise ValueError("TF-IDF 测试结果与验证集配置不匹配")
    validate_predictions(tfidf_test, splits.test, splits.labels)
    tests = {"TF-IDF + LinearSVC": tfidf_test}
    for label, model in zip(("BERT", "RoBERTa"), MODEL_NAMES, strict=True):
        winner = winners[model]
        requested_device = winner["config"]["device"] if args.device == "auto" else args.device
        report = evaluate_checkpoint(winner, splits.test, splits.labels, torch.device(requested_device))
        validate_predictions(report, splits.test, splits.labels)
        write_report(out / f"{model}_test.json", report)
        tests[label] = report
        print(f"{label} {winner['experiment']} test Macro-F1={report['metrics']['macro_f1']:.4f}", flush=True)
    write_report(out / "test_comparison.json", {
        "split": "test", "n": len(splits.test), "selection": selection, "labels": splits.labels,
        "methods": {
            label: {"experiment": report.get("experiment"), "metrics": report["metrics"]}
            for label, report in tests.items()
        },
    })
    update_summary(rows, tests, len(splits.val), len(splits.test))
    print(f"统一测试完成，已更新 {OUTPUTS / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
