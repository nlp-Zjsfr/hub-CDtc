"""实验共享的数据处理、评测指标和随机性控制。"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

LABELS = [
    "故事", "文化", "娱乐", "体育", "财经",
    "房产", "汽车", "教育", "科技", "军事",
    "旅游", "国际", "证券", "农业", "电竞",
]


def set_seed(seed: int = 42) -> None:
    """固定 Python、NumPy 与 PyTorch 的随机状态。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(name: str) -> list[dict]:
    with (DATA_DIR / name).open(encoding="utf-8") as file:
        return json.load(file)


def load_data() -> tuple[list[dict], list[dict], dict]:
    """读取数据，并从验证集中移除与训练集完全重复的文本。"""
    train = load_json("train.json")
    raw_val = load_json("val.json")
    train_texts = {item["sentence"] for item in train}
    val = [item for item in raw_val if item["sentence"] not in train_texts]
    metadata = {
        "n_train": len(train),
        "n_val_raw": len(raw_val),
        "n_val_clean": len(val),
        "removed_duplicates": len(raw_val) - len(val),
    }
    return train, val, metadata


def evaluate(y_true: list[int], y_pred: list[int]) -> dict:
    """计算整体指标与各类别指标。"""
    details = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted")), 4),
        "per_class": details,
    }


def save_result(filename: str, payload: dict) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

