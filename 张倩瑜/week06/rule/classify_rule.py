"""
L0 规则 / 关键词快筛：冷启动兜底，可解释，但上限低。
关键词按英文宣传手法写，命不中时回落到语料里最多的 Loaded_Language。
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import load_clean_val
from common.labels import LABEL2ID, LABELS
from common.metrics import classification_report

OUT = Path(__file__).resolve().parent.parent / "outputs"

# 更具体的手法在前；Loaded_Language 最泛，放最后
RULES = [
    ("Whataboutism,Straw_Men,Red_Herring", [r"\bwhat about\b", r"\bstraw m[ae]n\b", r"\bred herring\b"]),
    ("Bandwagon,Reductio_ad_hitlerum", [r"\bhitler\b", r"\bnazis?\b", r"\beveryone (knows|agrees)\b", r"\beverybody\b"]),
    ("Thought-terminating_Cliches", [r"it is what it is", r"end of (the )?story", r"nothing to see here"]),
    ("Black-and-White_Fallacy", [r"\beither\b.+\bor\b", r"no other (choice|option)", r"\bonly two\b"]),
    ("Slogans", [r"\bmake america\b", r"\bbuild the wall\b", r"\byes we can\b"]),
    ("Appeal_to_Authority", [r"\baccording to\b", r"\bexperts?\b", r"\bofficials?\b", r"\bscientists?\b"]),
    ("Flag-Waving", [r"\bour (nation|country|flag)\b", r"\bpatriot", r"\bamerican people\b"]),
    ("Causal_Oversimplification", [r"\bcaused by\b", r"\bthe reason\b", r"\bbecause of\b", r"\bdue to\b"]),
    ("Appeal_to_fear-prejudice", [r"\bfear\b", r"\bthreat", r"\bdanger", r"\binvad", r"\bterror"]),
    ("Exaggeration,Minimisation", [r"\bworst\b", r"\bgreatest\b", r"\bnever\b", r"\balways\b", r"\binsignificant\b"]),
    ("Doubt", [r"\ballegedly\b", r"\bso-called\b", r"\bquestionable\b", r"\bdubious\b"]),
    ("Name_Calling,Labeling", [r"\btraitors?\b", r"\bthugs?\b", r"\bfascists?\b", r"\bradical", r"\bliars?\b", r"\bcrooks?\b"]),
    ("Loaded_Language", [r"\boutrage", r"\bshock", r"\bdevastat", r"\bdisgrace", r"\bhorrif", r"\bevil\b"]),
]


def _looks_like_repetition(text: str) -> bool:
    words = re.findall(r"[A-Za-z']{4,}", text.lower())
    return any(words.count(word) >= 3 for word in set(words))


def predict_one(text: str) -> int:
    for name, pats in RULES:
        for pat in pats:
            if re.search(pat, text, flags=re.IGNORECASE):
                return LABEL2ID[name]
    if _looks_like_repetition(text):
        return LABEL2ID["Repetition"]
    return LABEL2ID["Loaded_Language"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    samples, meta = load_clean_val(n=args.n, seed=args.seed)
    ys = [d["label"] for d in samples]
    ps = [predict_one(d["sentence"]) for d in samples]
    report = classification_report(ys, ps, num_classes=len(LABELS))
    report["meta"] = meta
    report["method"] = "rule"

    OUT.mkdir(parents=True, exist_ok=True)
    pred = [{"idx": d["idx"], "gold": d["label"], "pred": p, "sentence": d["sentence"]}
            for d, p in zip(samples, ps)]
    (OUT / "rule_pred.json").write_text(json.dumps(pred, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "rule_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[rule] n={report['n']} acc={report.get('acc')} macro_f1={report.get('macro_f1')}")


if __name__ == "__main__":
    main()
