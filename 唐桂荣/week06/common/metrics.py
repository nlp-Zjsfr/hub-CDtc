"""分类实验共用的指标，混淆矩阵行是真实类别，列是预测类别。"""

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def classification_report(y_true, y_pred, labels: tuple[str, ...]) -> dict:
    label_ids = list(range(len(labels)))
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("真实和预测标签应非空且长度相同")
    if (set(y_true) | set(y_pred)) - set(label_ids):
        raise ValueError("真实或预测标签不在统一的类别表内")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=label_ids, zero_division=0,
    )
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1.mean()),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=label_ids).tolist(),
    }
