"""
从 SemEval-2020 Task 11（PTC v2）抽出技术分类金标。

Hugging Face 上的 `SemEvalWorkshop/sem_eval_2020_task_11` 只有加载脚本，
正文和标注不在仓库里。公开文本与训练集金标是 Zenodo 3952415 的
`datasets-v2.tgz`（CC-BY 4.0），标签字符串与 HF ClassLabel 一致。

每条样本是一条 TC 标注：sentence = 字符偏移切出的 span，label = 原手法名。
公开包的 dev 只有待填的 `?`，没有金标，因此从训练文章里按文章持出验证集，
避免同一篇文章的 span 同时出现在 train 和 val。
"""

from __future__ import annotations

import argparse
import json
import random
import tarfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.labels import LABEL2ID

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache"
ZENODO_URL = "https://zenodo.org/records/3952415/files/datasets-v2.tgz?download=1"
ARCHIVE_NAME = "datasets-v2.tgz"


def ensure_corpus(force: bool = False) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    archive = CACHE / ARCHIVE_NAME
    extracted = CACHE / "datasets"
    marker = extracted / "train-task2-TC.labels"
    if marker.exists() and not force:
        return extracted
    if force or not archive.exists():
        print(f"downloading {ZENODO_URL}")
        urllib.request.urlretrieve(ZENODO_URL, archive)
    if extracted.exists():
        import shutil
        shutil.rmtree(extracted)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(CACHE)
    if not marker.exists():
        raise FileNotFoundError(f"archive missing {marker}")
    return extracted


def load_annotations(root: Path) -> list[dict]:
    articles = {}
    for path in (root / "train-articles").glob("*.txt"):
        aid = path.stem.removeprefix("article")
        articles[aid] = path.read_text(encoding="utf-8", newline="")

    rows = []
    bad = 0
    for line in (root / "train-task2-TC.labels").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        aid, tech, start_s, end_s = line.split("\t")
        start, end = int(start_s), int(end_s)
        text = articles.get(aid)
        if text is None or tech not in LABEL2ID or not (0 <= start < end <= len(text)):
            bad += 1
            continue
        span = text[start:end].strip()
        if not span:
            bad += 1
            continue
        rows.append(
            {
                "sentence": span,
                "label": LABEL2ID[tech],
                "label_name": tech,
                "article_id": aid,
                "start": start,
                "end": end,
            }
        )
    if bad:
        raise RuntimeError(f"{bad} annotations failed to align with article text")
    return rows


def split_by_article(rows: list[dict], val_frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    by = defaultdict(list)
    for row in rows:
        by[row["article_id"]].append(row)

    rng = random.Random(seed)
    ids = list(by)
    rng.shuffle(ids)
    target = max(1, int(round(len(rows) * val_frac)))
    val_ids: set[str] = set()
    n_val = 0
    for aid in ids:
        if n_val >= target:
            break
        val_ids.add(aid)
        n_val += len(by[aid])
    train_ids = set(ids) - val_ids

    def counts(idset: set[str]) -> Counter:
        c: Counter = Counter()
        for aid in idset:
            for row in by[aid]:
                c[row["label_name"]] += 1
        return c

    labels = set(LABEL2ID)
    for _ in range(len(ids)):
        val_c, train_c = counts(val_ids), counts(train_ids)
        missing_val = [name for name in labels if val_c[name] == 0]
        missing_train = [name for name in labels if train_c[name] == 0]
        if not missing_val and not missing_train:
            break
        if missing_val:
            name = missing_val[0]
            donor = next(aid for aid in train_ids if any(r["label_name"] == name for r in by[aid]))
            val_ids.add(donor)
            train_ids.remove(donor)
        else:
            name = missing_train[0]
            donor = next(aid for aid in val_ids if any(r["label_name"] == name for r in by[aid]))
            train_ids.add(donor)
            val_ids.remove(donor)

    def pack(idset: set[str]) -> list[dict]:
        packed = []
        for aid in sorted(idset):
            packed.extend(by[aid])
        for i, row in enumerate(packed):
            row["idx"] = i
        return packed

    return pack(train_ids), pack(val_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = ensure_corpus(force=args.force)
    rows = load_annotations(root)
    train, val = split_by_article(rows, args.val_frac, args.seed)

    (HERE / "train.json").write_text(json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "val.json").write_text(json.dumps(val, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"annotations={len(rows)} train={len(train)} val={len(val)}")
    print("train dist:", dict(Counter(r["label_name"] for r in train)))
    print("val dist:", dict(Counter(r["label_name"] for r in val)))
    print(f"wrote {HERE / 'train.json'}")
    print(f"wrote {HERE / 'val.json'}")


if __name__ == "__main__":
    main()
