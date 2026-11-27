"""
\u5b9e\u9a8c B \u7684\u914d\u5bf9\u663e\u8457\u6027\u68c0\u9a8c\uff08paired bootstrap\uff09+ \u9010\u7c7b F1 \u5bf9\u6bd4

\u3010\u4e3a\u4ec0\u4e48\u8981\u505a\u8fd9\u4e2a\u3011
\u201cA \u6bd4 B \u9ad8 3 \u4e2a\u70b9\u201d\u8fd9\u53e5\u8bdd\uff0c\u5728\u6837\u672c\u91cf\u4e0d\u5927\u65f6\u53ef\u80fd\u53ea\u662f\u566a\u58f0\u3002
\u914d\u5bf9 bootstrap \u628a\u201c\u5dee\u503c\u5dee\u5f02\u5728\u54ea\u91cc\u201d\u91cf\u5316\u6210\u4e00\u4e2a\u533a\u95f4\uff1a
  - \u533a\u95f4\u5305\u542b 0  -> \u8fd9\u4e2a\u5dee\u5f02\u4e0d\u53ef\u9760\uff08\u53ef\u80fd\u662f\u566a\u58f0\uff09
  - \u533a\u95f4\u6574\u4f53\u5728 0 \u4e00\u4fa7 -> \u8fd9\u4e2a\u5dee\u5f02\u7edf\u8ba1\u4e0a\u663e\u8457

\u3010\u600e\u4e48\u505a\u3011
\u4e24\u4efd\u9884\u6d4b\u7528\u7684\u662f**\u540c\u4e00\u6279\u6837\u672c**\uff08\u540c\u4e00 seed=42 \u7684\u53bb\u91cd\u5e72\u51c0\u9a8c\u8bc1\u96c6\uff09\uff0c\u6240\u4ee5\u53ef\u4ee5\u505a\u201c\u914d\u5bf9\u201d\u91cd\u91c7\u6837\uff1a
\u6bcf\u6b21\u91cd\u91c7\u4e00\u6279\u4e0b\u6807\uff08\u6709\u653e\u56de\uff09\uff0c\u5728\u8fd9\u6279\u4e0a\u540c\u65f6\u7b97\u4e24\u4e2a\u6a21\u578b\u7684 Macro-F1\uff0c\u53d6\u5dee\uff0c\u91cd\u590d 2000 \u6b21\u3002
\u5206\u5e03\u7684 2.5% / 97.5% \u5206\u4f4d\u70b9\u5c31\u662f 95% \u7f6e\u4fe1\u533a\u95f4\u3002

\u3010\u8fd8\u80fd\u770b\u51fa\u4ec0\u4e48\u3011
  - \u7ffb\u8f6c\u6837\u672c\u6570\uff1a\u6709\u591a\u5c11\u6761\u6837\u672c\u201cA \u5bf9 B \u9519\u201d\u53c8\u201cB \u5bf9 A \u9519\u201d -> \u51c0\u6539\u53d8\u91cf\uff08\u771f\u6b63\u88ab\u5f71\u54cd\u7684\u6837\u672c\u6570\uff09
  - \u9010\u7c7b F1 \u5dee\uff1a\u63d0\u5347\u96c6\u4e2d\u5728\u54ea\u4e9b\u7c7b\uff08\u957f\u5c3e\u7c7b\u8fd8\u662f\u5934\u90e8\u7c7b\uff09

\u3010\u4e0e\u6807\u51c6\u7b54\u6848\u7684\u5bf9\u9f50\u3011
\u6807\u51c6\u7b54\u6848\u5728 8,901 \u6761\u5168\u91cf\u5e72\u51c0\u96c6\u4e0a\u505a\u540c\u6837\u7684\u914d\u5bf9 bootstrap\uff08\u5b83\u7684 CI \u662f [+0.0019, +0.0178]\uff09\uff0c
\u672c\u811a\u672c\u7528\u540c\u4e00\u65b9\u6cd5\u3001\u540c\u4e00\u6837\u672c\u91cf\uff08\u5168\u91cf 8,901 \u6761\uff09\u91cd\u505a\uff0c\u6240\u4ee5\u7ed3\u8bba\u53ef\u4e0e\u6807\u51c6\u7b54\u6848\u5e76\u5217\u5bf9\u7167\u3002
\u6ce8\uff1a\u672c\u5b9e\u9a8c\u7684\u8bad\u7ec3\u9884\u7b97\u662f 50 \u6b65\uff08\u7b49\u6b65\u6570\u5bf9\u7167\uff09\uff0c\u6807\u51c6\u7b54\u6848\u662f 3 epoch \u5168\u91cf\uff0c
    \u56e0\u6b64\u4e24\u8005\u7684\u7edd\u5bf9\u503c\u4e0d\u53ef\u76f4\u63a5\u6bd4\uff08\u6211\u4eec\u7684\u7edd\u5bf9\u503c\u4f4e\uff0c\u56e0\u4e3a\u53ea\u8bad\u4e86 50 \u6b65\uff09\uff0c
    \u4f46\u300c\u66f4\u6362\u57fa\u5ea7\u5e26\u6765\u7684\u76f8\u5bf9\u53d8\u5316\u300d\u662f\u53ef\u4ee5\u5bf9\u7167\u7684\u3002

\u7528\u6cd5\uff08\u5728\u4f5c\u4e1a\u76ee\u5f55\u4e0b\u6267\u884c\uff09\uff1a
  python \u914d\u5bf9bootstrap\u68c0\u9a8c.py                          # \u9ed8\u8ba4\u6bd4\u8f83\u4e24\u4e2a\u5168\u91cf\u53e3\u5f84\u7684\u9884\u6d4b\u6587\u4ef6
  python \u914d\u5bf9bootstrap\u68c0\u9a8c.py --iters 5000
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
CODE = HERE / "code"
sys.path.insert(0, str(CODE))
from common.labels import LABELS          # noqa: E402

N_CLS = len(LABELS)


def macro_f1(y, p):
    """Macro-F1\uff1a\u56fa\u5b9a 15 \u7c7b\uff08\u4e0d\u4f9d\u8d56\u672c\u6b21\u51fa\u73b0\u8fc7\u7684\u7c7b\uff09"""
    return float(f1_score(y, p, average="macro", labels=list(range(N_CLS)), zero_division=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=str(CODE / "outputs" / "bert_pred_bert-base-chinese_s50_full.json"))
    ap.add_argument("--b", default=str(CODE / "outputs" / "bert_pred_chinese-roberta-wwm-ext_s50_full.json"))
    ap.add_argument("--iters", type=int, default=2000, help="bootstrap \u91cd\u91c7\u6837\u6b21\u6570")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(HERE / "outputs" / "paired_bootstrap.json"))
    args = ap.parse_args()

    A = json.loads(Path(args.a).read_text(encoding="utf-8"))
    B = json.loads(Path(args.b).read_text(encoding="utf-8"))
    pa, pb = A["preds"], B["preds"]
    ya = [x["true"] for x in pa]
    yb = [x["true"] for x in pb]
    if ya != yb or len(pa) != len(pb):
        # \u914d\u5bf9\u68c0\u9a8c\u7684\u524d\u63d0\uff1a\u4e24\u4efd\u6587\u4ef6\u5fc5\u987b\u662f\u540c\u4e00\u6279\u6837\u672c\u3001\u540c\u4e00\u987a\u5e8f
        print("[abort] \u4e24\u4efd\u9884\u6d4b\u7684\u771f\u503c\u5e8f\u5217\u4e0d\u4e00\u81f4\uff0c\u65e0\u6cd5\u505a\u914d\u5bf9\u68c0\u9a8c")
        return
    y = np.array(ya)
    pa = np.array([x["pred"] for x in pa])
    pb = np.array([x["pred"] for x in pb])
    n = len(y)

    f1a, f1b = macro_f1(y, pa), macro_f1(y, pb)
    delta = f1b - f1a

    rng = np.random.default_rng(args.seed)
    diffs = np.empty(args.iters)
    for i in range(args.iters):
        idx = rng.integers(0, n, n)          # \u6709\u653e\u56de\u91c7\u6837
        diffs[i] = macro_f1(y[idx], pb[idx]) - macro_f1(y[idx], pa[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    sig = (lo > 0) or (hi < 0)               # \u533a\u95f4\u4e0d\u542b 0 -> \u663e\u8457

    only_a = int(((pa == y) & (pb != y)).sum())
    only_b = int(((pb == y) & (pa != y)).sum())
    both = int(((pa == y) & (pb == y)).sum())
    neither = n - both - only_a - only_b

    fa = f1_score(y, pa, average=None, labels=list(range(N_CLS)), zero_division=0)
    fb = f1_score(y, pb, average=None, labels=list(range(N_CLS)), zero_division=0)
    per_class = sorted([{"cls": LABELS[i], "f1_a": round(float(fa[i]), 4),
                          "f1_b": round(float(fb[i]), 4),
                          "delta": round(float(fb[i] - fa[i]), 4)} for i in range(N_CLS)],
                        key=lambda r: -r["delta"])

    res = {"n": n, "iters": args.iters, "seed": args.seed,
           "macro_f1_a": round(f1a, 4), "macro_f1_b": round(f1b, 4),
           "delta_macro_f1": round(delta, 4),
           "ci95": [round(float(lo), 4), round(float(hi), 4)],
           "significant": bool(sig),
           "sample_flips": {"both_correct": both, "only_a_correct": only_a,
                             "only_b_correct": only_b, "neither_correct": neither,
                             "net_changed": only_b - only_a,
                             "net_changed_pct": round((only_b - only_a) / n * 100, 2)},
           "per_class": per_class,
           "source_a": args.a, "source_b": args.b}
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\u914d\u5bf9 bootstrap\uff08{args.iters} \u6b21\u91cd\u91c7\u6837\uff0cn={n}\uff09")
    print(f"  A {Path(args.a).stem}: Macro-F1 = {f1a:.4f}")
    print(f"  B {Path(args.b).stem}: Macro-F1 = {f1b:.4f}")
    print(f"  \u0394 Macro-F1 = {delta:+.4f}   95% CI = [{lo:+.4f}, {hi:+.4f}]   "
          f"\u663e\u8457\uff1a{'YES' if sig else 'NO'}")
    print(f"  \u7ffb\u8f6c\u6837\u672c\uff1a\u53ea\u6709 A \u5bf9 {only_a} \u6761 / \u53ea\u6709 B \u5bf9 {only_b} \u6761 "
          f"-> \u51c0\u53d8\u5316 {only_b - only_a:+d} \u6761\uff08{(only_b - only_a) / n * 100:+.2f}%\uff09")
    print("\n| \u7c7b\u522b | A | B | \u0394 |")
    print("|---|---|---|---|")
    for r in per_class:
        print(f"| {r['cls']} | {r['f1_a']:.4f} | {r['f1_b']:.4f} | {r['delta']:+.4f} |")
    print(f"\n\u2192 {op}")


if __name__ == "__main__":
    main()
