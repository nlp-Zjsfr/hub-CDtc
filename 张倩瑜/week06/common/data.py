"""
数据读取与干净验证集（全项目共用）

对照课件：同一批验证样本 + 去重，是横向比较的前提。
"""

import json
import random
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = CODE_ROOT / "data"


def _load(name: str):
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def load_splits():
    """返回 (train, val)。每条含 sentence / label / label_name，以及 article_id、start、end。"""
    return _load("train.json"), _load("val.json")


def clean_val(train=None, val=None):
    """去掉与训练集句子完全重复的验证样本。"""
    if train is None or val is None:
        train, val = load_splits()
    seen = {d["sentence"] for d in train}
    clean = [d for d in val if d["sentence"] not in seen]
    return clean, len(val) - len(clean)


def sample_subset(items, n, seed=42):
    if n is None or n >= len(items):
        return list(items)
    rng = random.Random(seed)
    return rng.sample(items, n)


def load_clean_val(n=None, seed=42):
    train, val = load_splits()
    clean, n_dup = clean_val(train, val)
    samples = sample_subset(clean, n if n else len(clean), seed)
    meta = {
        "n_train": len(train),
        "n_val": len(val),
        "n_dup": n_dup,
        "dup_rate": round(n_dup / len(val), 4) if val else 0.0,
        "n_clean": len(clean),
        "n_eval": len(samples),
        "seed": seed,
    }
    return samples, meta
