"""
中文文本领域分类数据（18 个国民经济行业大类，全项目共用）

数据来源：
  train_data/zh.test.csv.tgz（阿里达摩院「中文文本领域分类测试集」，Apache-2.0）
  压缩包里只有 zh.test.csv 一个文件，列名是 text / label:LABEL。

教学重点：
  1. 拿到手的只有一份**测试集**，没有训练集。真实项目里这很常见——
     要做有监督分类，第一步就是把这一份数据切成 train / val / test，
     并且**按类别分层**切，否则小类可能整类落进验证集，metric 会剧烈抖动。
  2. 切分固定 seed（默认 42）、固定比例（8:1:1），并缓存到 data/domain/。
     全项目所有方法都用同一份切分，数字才可比。
  3. 切分前先按整句去重：这份数据里存在少量完全重复的文本，
     留着会让「训练/测试泄漏」，把指标虚高。

用法：
  from common.domain_data import DOMAIN_LABELS, load_domain_splits
  train, val, test = load_domain_splits()
"""

import csv
import json
import random
import tarfile
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DATA_DIR = CODE_ROOT / "train_data"
CSV_PATH = TRAIN_DATA_DIR / "zh.test.csv"
TGZ_PATH = TRAIN_DATA_DIR / "zh.test.csv.tgz"
CACHE_DIR = CODE_ROOT / "data" / "domain"

# 18 个国民经济行业大类（顺序固定，改了会让旧 checkpoint 的 label id 全部错位）
DOMAIN_LABELS = [
    "交通运输仓储邮政", "住宿餐饮", "信息软件", "农业", "制造业", "卫生医疗",
    "国际组织", "建筑", "房地产", "政府组织", "教育", "文体娱乐",
    "水利环境", "电力燃气水生产", "科学技术", "租赁法律", "采矿", "金融",
]
LABEL2ID = {name: i for i, name in enumerate(DOMAIN_LABELS)}

SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
SPLIT_SEED = 42


def _ensure_csv() -> Path:
    """train_data 里只有 .tgz 时先解压出 csv；已存在则直接用。"""
    if CSV_PATH.exists():
        return CSV_PATH
    if not TGZ_PATH.exists():
        raise FileNotFoundError(f"未找到领域分类数据：{TGZ_PATH}")
    with tarfile.open(TGZ_PATH, "r:gz") as tf:
        member = next((m for m in tf.getmembers() if m.name.endswith("zh.test.csv")), None)
        if member is None:
            raise FileNotFoundError(f"{TGZ_PATH} 里没有 zh.test.csv")
        try:
            tf.extract(member, TRAIN_DATA_DIR, filter="data")   # Python 3.12+
        except TypeError:                                        # Python 3.10/3.11
            tf.extract(member, TRAIN_DATA_DIR)
    if not CSV_PATH.exists():                       # 压缩包内可能带一层目录
        found = next(TRAIN_DATA_DIR.rglob("zh.test.csv"), None)
        if found is None:
            raise FileNotFoundError("解压后仍未找到 zh.test.csv")
        found.replace(CSV_PATH)
    return CSV_PATH


def load_raw() -> list:
    """读 csv → [{"text": str, "label": int}]，并去掉空文本与整句重复。"""
    path = _ensure_csv()
    rows, seen = [], set()
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "text" not in reader.fieldnames:
            raise ValueError(f"{path} 列名不符合预期：{reader.fieldnames}")
        label_col = next(c for c in reader.fieldnames if c != "text")
        for row in reader:
            text = (row.get("text") or "").strip()
            label_name = (row.get(label_col) or "").strip()
            if not text or label_name not in LABEL2ID:
                continue
            if text in seen:                        # 整句去重，避免切分后泄漏
                continue
            seen.add(text)
            rows.append({"text": text, "label": LABEL2ID[label_name]})
    if not rows:
        raise ValueError(f"{path} 没有解析出任何样本")
    return rows


def build_splits(seed=SPLIT_SEED, ratios=None, rebuild=False):
    """
    分层切分 8:1:1 并缓存到 data/domain/{train,val,test}.json。
    返回 (splits, meta)；splits 是 {"train": [...], "val": [...], "test": [...]}。
    """
    from sklearn.model_selection import train_test_split

    ratios = ratios or SPLIT_RATIOS
    meta_path = CACHE_DIR / "meta.json"
    paths = {name: CACHE_DIR / f"{name}.json" for name in ratios}
    if not rebuild and meta_path.exists() and all(p.exists() for p in paths.values()):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("seed") == seed and meta.get("ratios") == ratios:
            return {name: json.loads(p.read_text(encoding="utf-8")) for name, p in paths.items()}, meta

    rows = load_raw()
    labels = [r["label"] for r in rows]
    n = len(rows)
    n_test = int(round(n * ratios["test"]))
    n_val = int(round(n * ratios["val"]))
    train_val, test = train_test_split(
        rows, test_size=n_test, random_state=seed, stratify=labels)
    val_size = n_val / len(train_val)
    train, val = train_test_split(
        train_val, test_size=val_size, random_state=seed,
        stratify=[r["label"] for r in train_val])

    splits = {"train": train, "val": val, "test": test}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        paths[name].write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    meta = {
        "source": str(CSV_PATH.relative_to(CODE_ROOT)),
        "n_total": n,
        "n_unique": n,
        "seed": seed,
        "ratios": ratios,
        "sizes": {k: len(v) for k, v in splits.items()},
        "labels": DOMAIN_LABELS,
        "class_counts": {
            name: {DOMAIN_LABELS[c]: sum(1 for r in items if r["label"] == c)
                   for c in range(len(DOMAIN_LABELS))}
            for name, items in splits.items()
        },
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return splits, meta


def load_domain_splits(seed=SPLIT_SEED, rebuild=False):
    """最常用入口：返回 (train, val, test)，每条是 {"text", "label"}。"""
    splits, _ = build_splits(seed=seed, rebuild=rebuild)
    return splits["train"], splits["val"], splits["test"]


def label_name(label_id: int) -> str:
    return DOMAIN_LABELS[label_id]


def sample_subset(items, n, seed=SPLIT_SEED):
    """固定 seed 抽样，方便冒烟/快速评测；n<=0 或超过总数时返回全部。"""
    if not n or n >= len(items):
        return list(items)
    rng = random.Random(seed)
    return rng.sample(items, n)


if __name__ == "__main__":
    # 自检：打印切分规模与类别分布
    splits, meta = build_splits(rebuild=True)
    print(json.dumps(meta["sizes"], ensure_ascii=False))
    for name in ("train", "val", "test"):
        counts = meta["class_counts"][name]
        print(f"{name}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
