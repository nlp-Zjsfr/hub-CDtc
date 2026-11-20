"""
L5 分流 CLI：规则快筛 → TF-IDF →（可选）置信度闸门 → 队列。

接口字段对照课件：topic/label、conf、queue、model、version。
"""

import argparse
import json
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import load_splits
from common.labels import LABELS, queue_name
from rule.classify_rule import predict_one as rule_predict

VERSION = "week06-semeval-tc-v1"


class Router:
    def __init__(self):
        train, _ = load_splits()
        self.vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=50000)
        x = self.vec.fit_transform([d["sentence"] for d in train])
        y = [d["label"] for d in train]
        # SGD(log_loss) 有 predict_proba，方便做置信度闸门
        self.clf = SGDClassifier(loss="log_loss", class_weight="balanced", max_iter=2000, random_state=42)
        self.clf.fit(x, y)

    def classify(self, text: str, use_rules: bool = False) -> dict:
        if use_rules:
            rid = rule_predict(text)
            # 规则命中给一个中等置信，便于演示分流
            conf = 0.66
            q = queue_name(rid, conf)
            return {
                "text": text,
                "label": LABELS[rid],
                "label_id": rid,
                "conf": conf,
                "queue": q,
                "model": "rule",
                "version": VERSION,
            }
        proba = self.clf.predict_proba(self.vec.transform([text]))[0]
        lid = int(proba.argmax())
        conf = float(proba.max())
        return {
            "text": text,
            "label": LABELS[lid],
            "label_id": lid,
            "conf": round(conf, 4),
            "queue": queue_name(lid, conf),
            "model": "tfidf_sgd",
            "version": VERSION,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--use-rules", action="store_true")
    args = ap.parse_args()

    router = Router()
    if args.text:
        res = router.classify(args.text, use_rules=args.use_rules)
        print(json.dumps(res, ensure_ascii=False, indent=2 if not args.json else None))
        return

    print("Propaganda technique CLI — 输入英文新闻片段，空行退出")
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            break
        res = router.classify(line, use_rules=args.use_rules)
        print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
