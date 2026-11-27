import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import json
import time
from pathlib import Path

from datasets import load_dataset
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

DATASET_NAME = "sookiemonster/asrs-narratives-rebalance"


# ---------- 评估（与 rule_classify.py 完全一致） ----------
def evaluate(y_true, y_pred, num_classes):
    n = len(y_true)
    unparsed = sum(1 for p in y_pred if p is None or p < 0)

    conf = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        if p is not None and p >= 0:
            conf[t][p] += 1

    correct = sum(conf[i][i] for i in range(num_classes))
    parsed = n - unparsed
    acc = correct / parsed if parsed else 0.0
    acc_incl = correct / n if n else 0.0

    per_recall, f1s, supports = {}, [], []
    for i in range(num_classes):
        tp = conf[i][i]
        row = sum(conf[i])
        col = sum(conf[r][i] for r in range(num_classes))
        recall = tp / row if row else 0.0
        precision = tp / col if col else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_recall[str(i)] = round(recall, 4)
        f1s.append(f1)
        supports.append(row)

    macro_f1 = sum(f1s) / num_classes if num_classes else 0.0
    weighted_f1 = sum(f * s for f, s in zip(f1s, supports)) / n if n else 0.0

    counts = {}
    for p in y_pred:
        key = str(p) if (p is not None and p >= 0) else "-1"
        counts[key] = counts.get(key, 0) + 1
    used = sum(1 for k, v in counts.items() if k != "-1" and v > 0)

    return {
        "n": n,
        "unparsed": unparsed,
        "unparsed_rate": round(unparsed / n, 4) if n else 0.0,
        "acc": round(acc, 4),
        "acc_incl_unparsed": round(acc_incl, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class_recall": per_recall,
        "confusion_matrix": conf,
        "pred_distribution": {"used_classes": used, "counts": counts},
    }


# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=-1,
                    help="训练条数；-1 全量（22992）")
    ap.add_argument("--n_test", type=int, default=-1,
                    help="测试条数；-1 全量（9868）")
    ap.add_argument("--out", default="outputs/tfidf_pred.json",
                    help="相对脚本上一级目录的输出路径")
    args = ap.parse_args()

    print(f"[加载] {DATASET_NAME} ...")
    ds = load_dataset(DATASET_NAME)
    label_names = ds["train"].features["label"].names
    num_classes = len(label_names)
    print(f"[标签] {num_classes} 类")

    train, test = ds["train"], ds["test"]
    n_train = len(train) if args.n_train < 0 else min(args.n_train, len(train))
    n_test = len(test) if args.n_test < 0 else min(args.n_test, len(test))

    X_train = list(train.select(range(n_train))["text"])
    y_train = list(train.select(range(n_train))["label"])
    X_test = list(test.select(range(n_test))["text"])
    y_test = list(test.select(range(n_test))["label"])

    print(f"[训练] {n_train} 条，[测试] {n_test} 条")

    # 1) TF-IDF: word(1,2) + char_wb(3,5) 组合
    word_vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.95,
        sublinear_tf=True, lowercase=True, strip_accents="unicode",
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2,
        sublinear_tf=True, lowercase=True, strip_accents="unicode",
    )

    t0 = time.time()
    print("[向量化] ...")
    X_train_w = word_vec.fit_transform(X_train)
    X_train_c = char_vec.fit_transform(X_train)
    X_train_vec = hstack([X_train_w, X_train_c]).tocsr()
    print(f"  TF-IDF 维度: {X_train_vec.shape[1]}  "
          f"(word {X_train_w.shape[1]} + char {X_train_c.shape[1]})  "
          f"耗时 {time.time() - t0:.1f}s")

    # 2) LinearSVC
    print("[训练] LinearSVC ...")
    clf = LinearSVC(C=1.0, class_weight="balanced", max_iter=3000, random_state=42)
    clf.fit(X_train_vec, y_train)
    print(f"  训练完成，累计 {time.time() - t0:.1f}s")

    # 3) 测试集预测
    print("[预测] ...")
    X_test_vec = hstack([word_vec.transform(X_test),
                         char_vec.transform(X_test)]).tocsr()
    y_pred = clf.predict(X_test_vec).tolist()
    print(f"  预测完成，累计 {time.time() - t0:.1f}s")

    # 4) 评估 + 输出（与 rule_pred.json 格式一致）
    report = evaluate(y_test, y_pred, num_classes)
    report["method"] = "tfidf_svm"
    report["sec_per_item"] = round((time.time() - t0) / n_test, 4) if n_test else 0.0

    meta = {
        "dataset": DATASET_NAME,
        "split_train": "train",
        "split_test": "test",
        "n_train": n_train,
        "n_val": len(ds["validation"]),
        "n_test": n_test,
        "n_eval": n_test,
        "seed": 42,
        "note": "TF-IDF(word 1-2 + char_wb 3-5) + LinearSVC；未做去重/清洗",
    }

    preds = [
        {"text": t, "true": yt, "pred": int(yp)}
        for t, yt, yp in zip(X_test, y_test, y_pred)
    ]

    out = Path(__file__).resolve().parent.parent / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"meta": meta, "report": report, "preds": preds},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n评测 {n_test} 条 | acc={report['acc']:.4f} | "
          f"macro_f1={report['macro_f1']:.4f} | "
          f"weighted_f1={report['weighted_f1']:.4f}")
    print(f"覆盖类别: {report['pred_distribution']['used_classes']}/{num_classes}")
    print(f"→ 结果写入 {out}")


if __name__ == "__main__":
    main()
