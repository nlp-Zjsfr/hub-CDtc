"""统一评测指标：Accuracy / Micro-F1 / Macro-F1 / per-class recall / 混淆矩阵。

Micro-F1 固定在任务的类别集合上计算。生成失败（pred < 0）不是合法类别，
不记为命中。预测全部落在类别集合内时，Micro-F1 等于准确率。
"""

import numpy as np


def classification_report(y_true, y_pred, num_classes=9):
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    valid = y_pred >= 0
    unparsed = int((~valid).sum())

    report = {
        "n": int(len(y_true)),
        "unparsed": unparsed,
        "unparsed_rate": round(unparsed / len(y_true), 4) if len(y_true) else 0.0,
    }
    if valid.sum() == 0:
        return report

    yt, yp = y_true[valid], y_pred[valid]
    report["acc"] = round(float(accuracy_score(yt, yp)), 4)
    n_correct = int((yt == yp).sum())
    report["acc_incl_unparsed"] = round(n_correct / len(y_true), 4) if len(y_true) else 0.0
    labels = list(range(num_classes))
    report["micro_f1"] = round(float(f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0)), 4)
    report["macro_f1"] = round(float(f1_score(yt, yp, average="macro", zero_division=0)), 4)
    report["weighted_f1"] = round(float(f1_score(yt, yp, average="weighted", zero_division=0)), 4)

    per_class = {}
    for c in labels:
        mask = yt == c
        r = float((yp[mask] == c).mean()) if mask.sum() else None
        per_class[c] = None if r is None else round(r, 4)
    report["per_class_recall"] = per_class
    report["confusion_matrix"] = confusion_matrix(yt, yp, labels=labels).tolist()
    return report


def pred_distribution(preds, num_classes=9):
    counts = {}
    for p in preds:
        counts[int(p)] = counts.get(int(p), 0) + 1
    used = sum(1 for k in counts if k >= 0)
    return {"used_classes": used, "counts": counts}


def to_markdown_table(results: list):
    """results: [(level, name, report), ...]，按 L0–L5 的顺序。"""
    lines = [
        "| 层级 | 方法 | Micro-F1 | Macro-F1 | 解析失败 |",
        "|---|---|---|---|---|",
    ]
    for level, name, r in results:
        micro = r.get("micro_f1", "-")
        macro = r.get("macro_f1", "-")
        up = r.get("unparsed", 0)
        lines.append(f"| {level} | {name} | {micro} | {macro} | {up} |")
    return "\n".join(lines)
