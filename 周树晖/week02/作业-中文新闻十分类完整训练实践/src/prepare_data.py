"""下载固定版本的真实数据，审计、去重，并只在训练集上拟合 TF-IDF。"""
import csv
import re
import time
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from .common import ROOT, dump_json, read_json, sha256

REPOSITORY = "649453932/Chinese-Text-Classification-Pytorch"
COMMIT = "109e2aac8ada85096f839f6e5619a2f0a1774eb4"
BASE_URL = f"https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}/THUCNews/data"
CLASS_ZH = {"finance": "财经", "realty": "房产", "stocks": "股票", "education": "教育", "science": "科技",
            "society": "社会", "politics": "时政", "sports": "体育", "game": "游戏", "entertainment": "娱乐"}


def normalize_text(text):
    """训练和推理共用：统一全半角、小写、连续空白，不移除正文数字。"""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).lower()).strip()


def download_data():
    manifest = []
    for filename in ["class.txt", "train.txt", "dev.txt", "test.txt"]:
        target = ROOT / "data/raw" / filename
        url = f"{BASE_URL}/{filename}"
        if not target.exists():
            for attempt in range(3):
                try:
                    request = urllib.request.Request(url, headers={"User-Agent": "NewsClassificationHomework/1.0"})
                    with urllib.request.urlopen(request, timeout=90) as response:
                        content = response.read()
                    content.decode("utf-8")  # 拒绝把下载失败的非文本内容当作数据。
                    target.write_bytes(content)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2)
        manifest.append({"file": filename, "url": url, "bytes": target.stat().st_size, "sha256": sha256(target)})
        print(f"数据文件已就绪: {filename}, {target.stat().st_size:,} bytes", flush=True)
    manifest_path = ROOT / "data/raw/manifest.json"
    if manifest_path.exists():
        previous = read_json(manifest_path)
        old = {x["file"]: x["sha256"] for x in previous["files"]}
        for item in manifest:
            if item["sha256"] != old.get(item["file"]):
                raise ValueError(f"原始数据校验失败: {item['file']}")
    dump_json(manifest_path, {
        "repository": f"https://github.com/{REPOSITORY}", "commit": COMMIT,
        "prepared_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "description": "THUCNews 的公开十分类新闻标题子集；不是74万篇原始新闻全文。",
        "files": manifest,
    })


def load_rows(path):
    rows, invalid = [], []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            invalid.append({"line": line_number, "reason": "blank"})
            continue
        try:
            text, raw_label = line.rsplit("\t", 1)
            label = int(raw_label)
            normalized = normalize_text(text)
            if not normalized or label not in range(10):
                raise ValueError("empty title or invalid label")
            rows.append({"id": f"{path.stem}:{line_number}", "text": text, "normalized": normalized, "label": label})
        except ValueError:
            invalid.append({"line": line_number, "reason": "bad_format_or_label"})
    return rows, invalid


def read_processed(split):
    path = ROOT / "data/processed" / f"{split}.jsonl"
    return [__import__("json").loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def prepare():
    started = time.perf_counter()
    download_data()
    source_classes = (ROOT / "data/raw/class.txt").read_text(encoding="utf-8-sig").splitlines()
    classes = [CLASS_ZH[name] for name in source_classes]
    assert len(classes) == 10 and len(set(classes)) == 10
    dump_json(ROOT / "models/classes.json", classes)
    original, invalid = {}, {}
    for split, filename in [("train", "train.txt"), ("val", "dev.txt"), ("test", "test.txt")]:
        original[split], invalid[split] = load_rows(ROOT / "data/raw" / filename)
    expected_sizes = {"train": 180000, "val": 10000, "test": 10000}
    assert {s: len(rows) for s, rows in original.items()} == expected_sizes, "原始数据规模与公开数据说明不符"

    # 先做数据完整性审计；不使用任何模型测试表现进行清洗或调参。
    label_sets = defaultdict(set)
    for rows in original.values():
        for row in rows:
            label_sets[row["normalized"]].add(row["label"])
    conflicting = {text for text, labels in label_sets.items() if len(labels) > 1}
    raw_sets = {s: {r["normalized"] for r in rows} for s, rows in original.items()}
    overlaps_before = {
        "train_val": len(raw_sets["train"] & raw_sets["val"]),
        "train_test": len(raw_sets["train"] & raw_sets["test"]),
        "val_test": len(raw_sets["val"] & raw_sets["test"]),
    }
    # 保留作者划分；重复标题优先留在测试集，其次验证集，避免训练见过考题。
    seen, cleaned, removal_reasons = set(), {}, {}
    for split in ["test", "val", "train"]:
        cleaned[split] = []
        reasons = Counter()
        local_seen = set()
        for row in original[split]:
            key = row["normalized"]
            if key in conflicting:
                reasons["conflicting_label"] += 1
            elif key in local_seen:
                reasons["duplicate_within_split"] += 1
            elif key in seen:
                reasons["duplicate_in_reserved_split"] += 1
            else:
                cleaned[split].append(row)
                local_seen.add(key)
        seen.update(local_seen)
        removal_reasons[split] = dict(reasons)
    final_sets = {s: {r["normalized"] for r in rows} for s, rows in cleaned.items()}
    overlaps_after = {
        "train_val": len(final_sets["train"] & final_sets["val"]),
        "train_test": len(final_sets["train"] & final_sets["test"]),
        "val_test": len(final_sets["val"] & final_sets["test"]),
    }
    assert sum(overlaps_after.values()) == 0
    import json
    for split, rows in cleaned.items():
        (ROOT / "data/processed" / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    with (ROOT / "data/processed/removed_conflicts.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["normalized_title", "labels"])
        for text in sorted(conflicting):
            writer.writerow([text, ";".join(classes[i] for i in sorted(label_sets[text]))])

    stats = {
        "classes": classes, "source_classes": source_classes, "raw_sizes": {s: len(r) for s, r in original.items()},
        "clean_sizes": {s: len(r) for s, r in cleaned.items()}, "invalid_rows": invalid,
        "conflicting_title_count": len(conflicting), "removal_reasons": removal_reasons,
        "overlaps_before": overlaps_before, "overlaps_after": overlaps_after,
        "split_policy": "保留原始划分；移除全部标签冲突标题；规范化重复按 test > val > train 保留一次。",
        "raw_class_counts": {s: dict(Counter(r["label"] for r in rows)) for s, rows in original.items()},
        "clean_class_counts": {s: dict(Counter(r["label"] for r in rows)) for s, rows in cleaned.items()},
        "title_lengths": {s: {"min": min(map(lambda r: len(r['normalized']), rows)),
                               "median": float(np.median([len(r["normalized"]) for r in rows])),
                               "p95": float(np.percentile([len(r["normalized"]) for r in rows], 95)),
                               "max": max(map(lambda r: len(r['normalized']), rows))} for s, rows in cleaned.items()},
    }
    dump_json(ROOT / "outputs/data_audit.json", stats)
    print(f"清洗后样本数: {stats['clean_sizes']}; 原始跨集合重叠: {overlaps_before}", flush=True)

    # 中文使用字与相邻双字作为特征，不需要额外的分词词典。
    feature_config = read_json(ROOT / "configs/experiments.json")["features"]
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=tuple(feature_config["ngram_range"]),
                                 max_features=feature_config["max_features"], min_df=feature_config["min_df"],
                                 sublinear_tf=True, lowercase=False, dtype=np.float32, norm="l2")
    x_train = vectorizer.fit_transform([r["normalized"] for r in cleaned["train"]]).tocsr()
    x_val = vectorizer.transform([r["normalized"] for r in cleaned["val"]]).tocsr()
    # 此处不转换测试集特征。模型选择完成后，evaluate.py 才转换测试集。
    joblib.dump(vectorizer, ROOT / "models/tfidf.joblib")
    for split, matrix in [("train", x_train), ("val", x_val)]:
        sparse.save_npz(ROOT / "data/features" / f"{split}.npz", matrix)
        np.save(ROOT / "data/features" / f"{split}_labels.npy", np.array([r["label"] for r in cleaned[split]], dtype=np.int64))
    stats["features"] = {"dimension": x_train.shape[1], "analyzer": "char", "ngram_range": feature_config["ngram_range"],
                         "fit_split": "train_only", "min_df": feature_config["min_df"], "sublinear_tf": True,
                         "train_nonzero": int(x_train.nnz), "train_density": float(x_train.nnz / np.prod(x_train.shape)),
                         "val_zero_vector_count": int(np.sum(x_val.getnnz(axis=1) == 0))}
    stats["preparation_seconds"] = time.perf_counter() - started
    dump_json(ROOT / "outputs/data_audit.json", stats)
    print(f"TF-IDF 已保存: train={x_train.shape}, val={x_val.shape}; 耗时 {stats['preparation_seconds']:.1f}s", flush=True)
    return stats


if __name__ == "__main__":
    prepare()
