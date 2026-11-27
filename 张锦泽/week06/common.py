"""数据读取、指标与模型推理的共用函数。数据检查只依赖 Python 标准库。"""

import csv
import hashlib
import json
import platform
import unicodedata
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_data(path):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or sorted(reader.fieldnames) != ["label", "text"]:
            raise ValueError(f"{path}: 表头必须且只能包含 text,label 两列")
        texts, labels = [], []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}:{reader.line_num}: 列数不正确")
            text, label = row["text"].strip(), row["label"].strip()
            if not text or not label:
                raise ValueError(f"{path}:{reader.line_num}: 文本和标签均不能为空")
            texts.append(text)
            labels.append(label)
    if not texts:
        raise ValueError(f"{path}: 没有数据")
    return texts, labels


def load_splits(data_dir):
    """标签表只从训练集建立；拒绝重复文本、跨集合重叠及未知标签。"""
    data_dir = Path(data_dir)
    splits, summary, seen = {}, {}, {}
    fingerprint = hashlib.sha256()
    label_names = None
    for name in ("train", "val", "test"):
        path = data_dir / f"{name}.csv"
        texts, labels = read_data(path)
        if name == "train":
            label_names = sorted(set(labels))
            if len(label_names) < 2:
                raise ValueError("训练集至少需要两个类别")
        if set(labels) != set(label_names):
            raise ValueError(f"{path}: 每个集合必须包含训练集中的全部类别，且不能有新标签")
        for index, text in enumerate(texts, start=2):
            key = "".join(unicodedata.normalize("NFKC", text).casefold().split())
            if key in seen:
                raise ValueError(f"重复文本：{name}.csv:{index} 与 {seen[key]}")
            seen[key] = f"{name}.csv:{index}"
        label_to_id = {label: index for index, label in enumerate(label_names)}
        splits[name] = (texts, [label_to_id[label] for label in labels])
        summary[name] = dict(Counter(labels))
        content = path.read_bytes()
        fingerprint.update(name.encode() + b"\0" + str(len(content)).encode() + b"\0" + content)
    return splits, label_names, summary, fingerprint.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # BOM 方便 Windows Excel 正确显示中文。
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metrics_for(truth, predicted, label_names):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predicted, labels=list(range(len(label_names))), average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def save_evaluation(output_dir, texts, truth, predicted, label_names):
    from sklearn.metrics import classification_report, confusion_matrix

    output_dir = Path(output_dir)
    metrics = metrics_for(truth, predicted, label_names)
    metrics["n_samples"] = len(truth)
    write_json(output_dir / "metrics.json", metrics)
    ids = list(range(len(label_names)))
    report = classification_report(
        truth, predicted, labels=ids, target_names=label_names, output_dict=True, zero_division=0
    )
    write_json(output_dir / "classification_report.json", report)
    write_csv(output_dir / "predictions.csv", ["text", "true_label", "predicted_label"], [
        {"text": text, "true_label": label_names[actual], "predicted_label": label_names[guess]}
        for text, actual, guess in zip(texts, truth, predicted)
    ])
    matrix = confusion_matrix(truth, predicted, labels=ids)
    fields = ["true_label"] + [f"pred_{label}" for label in label_names]
    write_csv(output_dir / "confusion_matrix.csv", fields, [
        dict(zip(fields, [label_names[index]] + row.tolist()))
        for index, row in enumerate(matrix)
    ])
    return metrics


def environment_info():
    packages = {}
    for name in ("numpy", "scipy", "scikit-learn", "joblib", "torch", "transformers"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def resolve_device(requested):
    import torch

    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA 不可用，请使用 --device cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS 不可用，请使用 --device cpu")
    return torch.device(requested)


def make_loader(texts, labels, tokenizer, max_length, batch_size, shuffle=False, seed=42):
    import torch
    from torch.utils.data import DataLoader

    encoded = tokenizer(texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    samples = [{**{key: value[index] for key, value in encoded.items()}, "labels": torch.tensor(label)}
               for index, label in enumerate(labels)]
    return DataLoader(samples, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      generator=torch.Generator().manual_seed(seed))


def neural_predictions(model, loader, device):
    import torch

    model.eval()
    predicted = []
    with torch.no_grad():
        for batch in loader:
            inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
            predicted.extend(model(**inputs).logits.argmax(dim=-1).cpu().tolist())
    return predicted
