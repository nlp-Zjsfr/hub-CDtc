"""模型选择冻结后执行最终测试。训练过程不导入此模块。"""
import csv
import math
from datetime import datetime
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import torch
from scipy import sparse
from sklearn.metrics import classification_report, confusion_matrix, log_loss

from .common import ROOT, dump_json, read_json, sha256
from .model import load_checkpoint
from .prepare_data import normalize_text, read_processed
from .train import metric_dict, seed_everything


@torch.inference_mode()
def predict_matrix(model, matrix, device, batch_size=1024):
    model.eval()
    logits_list = []
    for start in range(0, matrix.shape[0], batch_size):
        xb = torch.from_numpy(matrix[start:start + batch_size].toarray()).to(device)
        logits_list.append(model(xb).cpu())
    logits = torch.cat(logits_list)
    return logits, torch.softmax(logits, dim=1).numpy()


def wilson_interval(correct, total, z=1.959963984540054):
    p = correct / total
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [center - half, center + half]


def evaluate_final():
    torch.set_num_threads(4)
    seed_everything(42)
    selection = read_json(ROOT / "outputs/selection.json")
    assert sha256(ROOT / "models/best_model.pt") == selection["main_checkpoint_sha256"]
    assert sha256(ROOT / "models/linear_baseline.pt") == selection["linear_checkpoint_sha256"]
    assert sha256(ROOT / "models/tfidf.joblib") == selection["vectorizer_sha256"]
    assert sha256(ROOT / "data/processed/test.jsonl") == selection["test_processed_sha256"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vectorizer = joblib.load(ROOT / "models/tfidf.joblib")
    classes = read_json(ROOT / "models/classes.json")
    rows = read_processed("test")
    true = np.array([row["label"] for row in rows], dtype=np.int64)
    matrix = vectorizer.transform([row["normalized"] for row in rows]).tocsr()
    main, main_ckpt = load_checkpoint(ROOT / "models/best_model.pt", device)
    linear, _ = load_checkpoint(ROOT / "models/linear_baseline.pt", device)
    main_logits, main_probs = predict_matrix(main, matrix, device)
    _, linear_probs = predict_matrix(linear, matrix, device)
    predictions = {"selected_mlp": main_probs.argmax(axis=1), "linear_baseline": linear_probs.argmax(axis=1),
                   "majority_baseline": np.full(len(rows), selection["majority_label"])}
    metrics = {}
    for name, predicted in predictions.items():
        m = metric_dict(true, predicted)
        m["correct"] = int(np.sum(true == predicted))
        m["wrong"] = len(rows) - m["correct"]
        m["accuracy_95ci_wilson"] = wilson_interval(m["correct"], len(rows))
        metrics[name] = m
    metrics["selected_mlp"]["cross_entropy"] = float(log_loss(true, main_probs, labels=list(range(10))))
    metrics["linear_baseline"]["cross_entropy"] = float(log_loss(true, linear_probs, labels=list(range(10))))
    report = classification_report(true, predictions["selected_mlp"], labels=list(range(10)),
                                   target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(true, predictions["selected_mlp"], labels=list(range(10)))
    dump_json(ROOT / "outputs/evaluation/classification_report.json", report)
    dump_json(ROOT / "outputs/evaluation/confusion_matrix.json", cm.tolist())
    with (ROOT / "outputs/evaluation/classification_report.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["类别", "Precision", "Recall", "F1", "Support"])
        for name in classes:
            item = report[name]
            writer.writerow([name, item["precision"], item["recall"], item["f1-score"], item["support"]])
    details = []
    for i, row in enumerate(rows):
        order = np.argsort(-main_probs[i])[:3]
        details.append({"id": row["id"], "text": row["text"], "true_label": classes[row["label"]],
                        "prediction": classes[int(order[0])], "probability": float(main_probs[i, order[0]]),
                        "correct": bool(order[0] == row["label"]),
                        "top3": [{"label": classes[int(j)], "probability": float(main_probs[i, j])} for j in order]})
    for filename, subset in [("test_predictions.csv", details), ("errors.csv", [r for r in details if not r["correct"]])]:
        with (ROOT / "outputs/evaluation" / filename).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["样本ID", "标题", "真实类别", "预测类别", "最高Softmax概率", "是否正确", "Top3"])
            for row in subset:
                writer.writerow([row["id"], row["text"], row["true_label"], row["prediction"], row["probability"], row["correct"],
                                 "; ".join(f"{x['label']}:{x['probability']:.6f}" for x in row["top3"])])
    mistakes = [row for row in details if not row["correct"]]
    confusions = sorted([(int(cm[i, j]), classes[i], classes[j]) for i in range(10) for j in range(10) if i != j], reverse=True)
    error_analysis = {"most_common_confusions": [{"count": n, "true": a, "predicted": b} for n, a, b in confusions[:10]],
                      "high_probability_errors": sorted(mistakes, key=lambda r: r["probability"], reverse=True)[:12],
                      "low_probability_errors": sorted(mistakes, key=lambda r: r["probability"])[:12],
                      "examples_by_confusion": []}
    for count, a, b in confusions[:5]:
        matched = [r for r in mistakes if r["true_label"] == a and r["prediction"] == b]
        error_analysis["examples_by_confusion"].append({"true": a, "predicted": b, "count": count, "examples": matched[:3]})
    dump_json(ROOT / "outputs/evaluation/error_analysis.json", error_analysis)

    # 在同一批测试标题上做配对 bootstrap，量化 MLP 相对线性基线的准确率差异。
    difference = (predictions["selected_mlp"] == true).astype(float) - (predictions["linear_baseline"] == true).astype(float)
    rng = np.random.default_rng(2026)
    bootstrap = [float(difference[rng.integers(0, len(true), len(true))].mean()) for _ in range(1000)]
    paired = {"accuracy_difference_mlp_minus_linear": float(difference.mean()),
              "paired_bootstrap_95ci": np.percentile(bootstrap, [2.5, 97.5]).tolist(), "resamples": 1000,
              "mlp_only_correct": int(np.sum(difference == 1)), "linear_only_correct": int(np.sum(difference == -1))}
    # 以真实模型输出演示：交叉熵 = -log(正确类别概率)。
    first = 0
    ce_torch = torch.nn.functional.cross_entropy(main_logits[first:first + 1], torch.tensor([int(true[first])])).item()
    log_probability = torch.log_softmax(main_logits[first], dim=0)[true[first]].item()
    loss_example = {"title": rows[first]["text"], "true_label": classes[true[first]], "logits": main_logits[first].tolist(),
                    "probabilities": main_probs[first].tolist(), "true_class_probability": float(main_probs[first, true[first]]),
                    "negative_log_probability": -log_probability, "cross_entropy_torch": ce_torch,
                    "absolute_difference": abs(ce_torch + log_probability)}
    assert loss_example["absolute_difference"] < 1e-6
    dump_json(ROOT / "outputs/evaluation/cross_entropy_example.json", loss_example)

    # 主模型用同一最佳 checkpoint 重新评估训练集与验证集，不能用在线训练准确率代替。
    final_split_metrics = {}
    for split in ["train", "val"]:
        feature_matrix = sparse.load_npz(ROOT / "data/features" / f"{split}.npz")
        labels = np.load(ROOT / "data/features" / f"{split}_labels.npy")
        logits, probabilities = predict_matrix(main, feature_matrix, device)
        final_split_metrics[split] = {**metric_dict(labels, probabilities.argmax(axis=1)),
                                     "cross_entropy": torch.nn.functional.cross_entropy(logits, torch.from_numpy(labels)).item()}
    result = {"evaluated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "n_test": len(rows),
              "metrics": metrics, "paired_comparison": paired, "main_checkpoint": selection["main_checkpoint_sha256"],
              "frozen_at": selection["frozen_at"], "classes": classes, "final_split_metrics": final_split_metrics,
              "test_zero_vector_count": int(np.sum(matrix.getnnz(axis=1) == 0)),
              "test_label_counts": np.bincount(true, minlength=10).tolist(),
              "main_test_mean_cross_entropy_torch": torch.nn.functional.cross_entropy(main_logits, torch.from_numpy(true)).item()}
    dump_json(ROOT / "outputs/evaluation/final_metrics.json", result)

    # 自编标题仅作交互展示，不能计入有标注测试集的指标。
    demo_titles = ["央行宣布下调存款准备金率支持实体经济", "多地调整住房公积金贷款政策", "沪深两市成交额放大券商板块上涨",
                   "高校公布今年本科招生计划", "国产芯片研发取得新进展", "民警帮助走失老人找到家人",
                   "联合国大会讨论地区和平议题", "中国女排战胜对手晋级决赛", "新款角色扮演游戏开启公测", "新电影发布预告演员分享拍摄经历"]
    demo_matrix = vectorizer.transform([normalize_text(text) for text in demo_titles]).tocsr()
    _, demo_probs = predict_matrix(main, demo_matrix, device)
    demos = [{"title": title, "top3": [{"label": classes[int(j)], "probability": float(p[j])} for j in np.argsort(-p)[:3]]}
             for title, p in zip(demo_titles, demo_probs)]
    dump_json(ROOT / "outputs/evaluation/demo_predictions.json", demos)
    features = vectorizer.get_feature_names_out()
    weights = linear.network[0].weight.detach().cpu().numpy()
    dump_json(ROOT / "outputs/evaluation/linear_top_features.json", {
        name: [str(features[j]) for j in np.argsort(-weights[i])[:12]] for i, name in enumerate(classes)})
    print(f"最终测试集: {len(rows)} 条；MLP Accuracy={metrics['selected_mlp']['accuracy']:.4%}, "
          f"Macro-F1={metrics['selected_mlp']['macro_f1']:.4%}; 线性基线Accuracy={metrics['linear_baseline']['accuracy']:.4%}", flush=True)
    return result


if __name__ == "__main__":
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    evaluate_final()
