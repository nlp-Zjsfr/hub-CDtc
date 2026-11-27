"""字符 TF-IDF + LinearSVC 基线。

默认评估验证集。统一测试时可指定 --split test。
在作业目录运行 python -m tfidf.classify_tfidf，或运行本文件。
"""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

PROJECT_DIR = Path(__file__).resolve().parent.parent
if not __package__:
    sys.path.insert(0, str(PROJECT_DIR))

from common.data import SPLIT_SEED, VAL_FRACTION, load_splits
from common.metrics import classification_report


def main():
    parser = argparse.ArgumentParser(description="训练和评估商品分类 TF-IDF 基线")
    parser.add_argument("--split", choices=("val", "test"), default="val", help="评估集合")
    parser.add_argument("--out", type=Path, help="结果文件路径，相对路径以作业目录为准")
    args = parser.parse_args()

    splits = load_splits()
    samples = splits.val if args.split == "val" else splits.test
    ngram_range = (1, 2)
    max_features = 300_000
    c = 1.0
    model_seed = 42
    pipeline = make_pipeline(
        TfidfVectorizer(
            analyzer="char", ngram_range=ngram_range, max_features=max_features, lowercase=True,
        ),
        LinearSVC(C=c, random_state=model_seed),
    )

    start = perf_counter()
    pipeline.fit(
        [sample["sentence"] for sample in splits.train],
        [sample["label"] for sample in splits.train],
    )
    fit_seconds = perf_counter() - start

    start = perf_counter()
    predictions = pipeline.predict([sample["sentence"] for sample in samples]).tolist()
    predict_seconds = perf_counter() - start
    actual = [sample["label"] for sample in samples]
    report = classification_report(actual, predictions, splits.labels)
    result = {
        "method": "tfidf_linear_svc",
        "split": args.split,
        "split_seed": SPLIT_SEED,
        "val_fraction": VAL_FRACTION,
        "split_sizes": {"train": len(splits.train), "val": len(splits.val), "test": len(splits.test)},
        "labels": splits.labels,
        "config": {
            "analyzer": "char",
            "ngram_range": ngram_range,
            "max_features": max_features,
            "lowercase": True,
            "C": c,
            "random_state": model_seed,
            "scikit_learn_version": sklearn.__version__,
        },
        "n_features": len(pipeline.named_steps["tfidfvectorizer"].vocabulary_),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "metrics": report,
        "predictions": [
            {"idx": sample["idx"], "label": sample["label"], "predicted_label": prediction}
            for sample, prediction in zip(samples, predictions, strict=True)
        ],
    }
    output = PROJECT_DIR / (args.out or Path("outputs") / f"tfidf_{args.split}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"训练 {len(splits.train):,} 条 | {args.split} 评估 {len(samples):,} 条")
    print(f"Macro-F1 {report['macro_f1']:.4f} | Accuracy {report['accuracy']:.4f}")
    print(f"结果已写入 {output}")


if __name__ == "__main__":
    main()
