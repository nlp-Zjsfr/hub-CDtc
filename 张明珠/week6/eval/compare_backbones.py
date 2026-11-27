"""
换基座的配对对比实验（A/B）：同一批样本上，两个 checkpoint 的预测是否真的不同？

为什么不能只比两个 Macro-F1 的小数点：
  500 条样本上 Macro-F1 的 95% 置信区间宽约 10 个点，±1 个点的差距完全可能是抽样噪声。
  正确做法是**配对比较**——先看两份预测在同一批样本上「谁对谁错」的分歧格子数，
  再做 McNemar 精确检验 + 配对 bootstrap，给出「差值」的置信区间。

用法：
  python eval/compare_backbones.py \
      --a outputs/bert_pred.json --b outputs/roberta_pred.json \
      --name_a BERT-base --name_b RoBERTa-base \
      --out outputs/backbone_ab.json

输出：控制台一张表 + 可选 JSON（含每类 recall 差值、McNemar p、差值 CI），可直接贴进文档。
"""

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score


def load_preds(path):
    """读取 predict_*.py 写出的 json，返回 {句子: (真标签, 预测)} —— 用句子做 key 才能跨文件对齐样本。"""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    preds = d["preds"]
    return {x["text"]: (int(x["true"]), int(x["pred"])) for x in preds}, d.get("report", {}), d.get("meta", {})


def mcnemar_exact(n01, n10):
    """双侧精确 McNemar：在「只有分歧格子」的二项分布下算 p。n01/n10 是两种方向的分歧数。"""
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def bootstrap_delta(y, pa, pb, times=2000, seed=42):
    """配对 bootstrap：同一批下标重采样，分别算两个模型的 Macro-F1，得到差值的分布。"""
    rng = np.random.default_rng(seed)
    out = np.empty(times)
    for i in range(times):
        idx = rng.integers(0, len(y), len(y))
        out[i] = f1_score(y[idx], pb[idx], average="macro") - f1_score(y[idx], pa[idx], average="macro")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="对照组预测 json（旧基座）")
    ap.add_argument("--b", required=True, help="实验组预测 json（新基座）")
    ap.add_argument("--name_a", default="A")
    ap.add_argument("--name_b", default="B")
    ap.add_argument("--out", default="", help="可选：把结果写成 json")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    A, ra, ma = load_preds(args.a)
    B, rb, mb = load_preds(args.b)
    keys = [k for k in A if k in B]
    if not keys:
        raise SystemExit("两份预测没有共同的句子——确认它们是在同一批样本上跑的（同 seed、同 --n）")
    if len(keys) != len(A) or len(keys) != len(B):
        print(f"[warn] 样本不完全重合：A={len(A)} B={len(B)} 交集={len(keys)}，只用交集做配对比较")

    y = np.array([A[k][0] for k in keys])
    pa = np.array([A[k][1] for k in keys])
    pb = np.array([B[k][1] for k in keys])
    acc_a, acc_b = float((pa == y).mean()), float((pb == y).mean())
    f_a, f_b = f1_score(y, pa, average="macro"), f1_score(y, pb, average="macro")

    # 分歧格子：a 错 b 对 / a 对 b 错
    n01 = int(((pa != y) & (pb == y)).sum())
    n10 = int(((pa == y) & (pb != y)).sum())
    p_mc = mcnemar_exact(n01, n10)

    d = bootstrap_delta(y, pa, pb, times=args.boot) * 100
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    p_boot = float((d > 0).mean())

    print(f"\n配对样本 n = {len(keys)}")
    print(f"| 模型 | 准确率 | Macro-F1 |")
    print(f"|---|---|---|")
    print(f"| {args.name_a} | {acc_a:.4f} | {f_a:.4f} |")
    print(f"| {args.name_b} | {acc_b:.4f} | {f_b:.4f} |")
    print(f"| 差值（{args.name_b} - {args.name_a}） | {100*(acc_b-acc_a):+.2f} pt | {100*(f_b-f_a):+.2f} pt |")
    print(f"\nMcNemar（配对卡方）: A错B对 = {n01}，A对B错 = {n10}，双侧精确 p = {p_mc:.3f}")
    print(f"配对 bootstrap 差值 95%CI = [{lo:+.2f}, {hi:+.2f}] pt，P(差值>0) = {p_boot:.2f}")
    if p_mc > 0.05 and lo < 0 < hi:
        print("→ 结论：**这批样本上分不出胜负**，差值在噪声范围内（别拿 0.x 个点当结论）")
    elif p_mc <= 0.05:
        print("→ 结论：分歧显著，换基座确实改变了模型行为")

    # 每类 recall 差值：看「涨在哪、掉在哪」，比总分有用
    per_class = []
    for c in sorted(set(y.tolist())):
        m = y == c
        r_a, r_b = float((pa[m] == c).mean()), float((pb[m] == c).mean())
        per_class.append({"label": int(c), "n": int(m.sum()), "recall_a": r_a, "recall_b": r_b,
                          "delta_pt": 100 * (r_b - r_a)})
    per_class.sort(key=lambda x: -x["delta_pt"])
    print("\n逐类 recall（按差值排序，n<20 的类一两句话就能翻盘，别过度解读）：")
    print("| 类别 | n | A recall | B recall | 差值 |")
    print("|---|---|---|---|---|")
    for r in per_class:
        print(f"| {r['label']} | {r['n']} | {r['recall_a']:.3f} | {r['recall_b']:.3f} | {r['delta_pt']:+.1f} pt |")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "n_paired": len(keys),
            "a": {"name": args.name_a, "file": args.a, "acc": acc_a, "macro_f1": f_a, "meta": ma, "report": ra},
            "b": {"name": args.name_b, "file": args.b, "acc": acc_b, "macro_f1": f_b, "meta": mb, "report": rb},
            "delta_pt": {"acc": 100 * (acc_b - acc_a), "macro_f1": 100 * (f_b - f_a)},
            "mcnemar": {"a_wrong_b_right": n01, "a_right_b_wrong": n10, "p_exact": p_mc},
            "bootstrap_delta_ci95_pt": [lo, hi],
            "p_delta_gt_0": p_boot,
            "per_class": per_class,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[out] 已写入 {args.out}")


if __name__ == "__main__":
    main()
