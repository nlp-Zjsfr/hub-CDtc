"""
L1 TF-IDF + LinearSVC：小数据 / CPU 场景基线。
英文 span 用词 1-2gram。
"""

import argparse
import json
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import load_clean_val, load_splits
from common.labels import LABELS
from common.metrics import classification_report

OUT = Path(__file__).resolve().parent.parent / "outputs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train, _ = load_splits()
    samples, meta = load_clean_val(n=args.n, seed=args.seed)

    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=50000)
    x_tr = vec.fit_transform([d["sentence"] for d in train])
    y_tr = [d["label"] for d in train]
    clf = LinearSVC(class_weight="balanced", max_iter=5000)
    clf.fit(x_tr, y_tr)

    x_va = vec.transform([d["sentence"] for d in samples])
    ps = clf.predict(x_va).tolist()
    ys = [d["label"] for d in samples]
    report = classification_report(ys, ps, num_classes=len(LABELS))
    report["meta"] = meta
    report["method"] = "tfidf"

    OUT.mkdir(parents=True, exist_ok=True)
    pred = [{"idx": d["idx"], "gold": d["label"], "pred": int(p), "sentence": d["sentence"]}
            for d, p in zip(samples, ps)]
    (OUT / "tfidf_pred.json").write_text(json.dumps(pred, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "tfidf_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[tfidf] n={report['n']} acc={report.get('acc')} macro_f1={report.get('macro_f1')}")


if __name__ == "__main__":
    main()
