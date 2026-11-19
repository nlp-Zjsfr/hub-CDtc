"""训练并评测多数类、逻辑回归和线性支持向量机基线。"""

from __future__ import annotations

import argparse
import time

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from common import evaluate, load_data, save_result, set_seed


def build_models() -> dict[str, object]:
    """构建复杂度逐步增加的传统分类器。"""
    return {
        "多数类基线": DummyClassifier(strategy="most_frequent"),
        "TF-IDF + 逻辑回归": Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(1, 2), max_features=200_000)),
            ("classifier", LogisticRegression(max_iter=500, C=4.0, solver="liblinear")),
        ]),
        "TF-IDF + 线性SVM": Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(1, 2), max_features=200_000)),
            ("classifier", LinearSVC(C=1.0)),
        ]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="运行传统中文文本分类基线")
    parser.add_argument("--max-train", type=int, default=0, help="最多使用的训练样本数，0 表示全部")
    parser.add_argument("--max-eval", type=int, default=0, help="最多使用的验证样本数，0 表示全部")
    args = parser.parse_args()

    set_seed(42)
    train, val, metadata = load_data()
    if args.max_train:
        train = train[: args.max_train]
    if args.max_eval:
        val = val[: args.max_eval]

    x_train = [item["sentence"] for item in train]
    y_train = [item["label"] for item in train]
    x_val = [item["sentence"] for item in val]
    y_val = [item["label"] for item in val]

    all_results = {}
    for name, model in build_models().items():
        started = time.perf_counter()
        model.fit(x_train, y_train)
        train_seconds = time.perf_counter() - started
        started = time.perf_counter()
        predictions = model.predict(x_val).tolist()
        inference_seconds = time.perf_counter() - started
        result = evaluate(y_val, predictions)
        result.update({
            "train_seconds": round(train_seconds, 2),
            "inference_seconds": round(inference_seconds, 2),
            "n_train": len(train),
            "n_eval": len(val),
        })
        all_results[name] = result
        print(f"{name}: Accuracy={result['accuracy']:.4f}, Macro-F1={result['macro_f1']:.4f}")

    path = save_result("baseline_results.json", {"data": metadata, "results": all_results})
    print(f"结果已保存至 {path}")


if __name__ == "__main__":
    main()

