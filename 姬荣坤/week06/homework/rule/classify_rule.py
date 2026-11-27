import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import json
import re
import time
from pathlib import Path

from datasets import load_dataset

DATASET_NAME = "sookiemonster/asrs-narratives-rebalance"
SPLIT = "test"


# ---------- 关键词构建 ----------
def build_keywords(label_names):
    """每个标签 -> 关键词集合：整体名 + 按分隔符切出的子词（长度>1）"""
    out = []
    for name in label_names:
        s = name.lower()
        parts = {s}
        for sep in ["/", "-", "_", " "]:
            if sep in s:
                parts.update(p.strip() for p in s.split(sep) if len(p.strip()) > 1)
        out.append(parts)
    return out


def compile_patterns(keywords):
    """预编译：单词边界 + 可选复数 s/es，忽略大小写"""
    return [
        [re.compile(r"\b" + re.escape(kw) + r"(?:s|es)?\b", re.IGNORECASE) for kw in kws]
        for kws in keywords
    ]


def classify_by_keywords(text, patterns):
    """返回 (pred_idx, counts)：每个标签的关键词命中总数取最高；无命中返回 -1"""
    counts = []
    for pats in patterns:
        cnt = 0
        for p in pats:
            cnt += len(p.findall(text))
        counts.append(cnt)

    best_idx, best_cnt = -1, 0
    for i, c in enumerate(counts):
        if c > best_cnt:          # 严格大于，保证并列时取索引最小者（确定性 tie-break）
            best_cnt, best_idx = c, i
    return best_idx, counts


# ---------- 评估 ----------
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
    ap.add_argument("--n", type=int, default=500,
                    help="评估条数；-1 表示使用全部 test 数据")
    ap.add_argument("--out", default="outputs/rule_pred.json",
                    help="相对脚本上一级目录的输出路径")
    args = ap.parse_args()

    print(f"[加载] {DATASET_NAME} ...")
    ds = load_dataset(DATASET_NAME)
    label_names = ds[SPLIT].features["label"].names
    num_classes = len(label_names)
    print(f"[标签] {num_classes} 类: {label_names}")

    test = ds[SPLIT]
    n_eval = len(test) if args.n is None or args.n < 0 else min(args.n, len(test))
    test_subset = test.select(range(n_eval))

    patterns = compile_patterns(build_keywords(label_names))

    preds, details = [], []
    t0 = time.time()
    for i, d in enumerate(test_subset):
        pred, counts = classify_by_keywords(d["text"], patterns)
        preds.append(pred)
        details.append({
            "text": d["text"],
            "true": d["label"],
            "pred": pred,
            "n_kw_matches": counts,   # 每个标签的命中次数，便于调试
        })
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n_eval}  {time.time() - t0:.1f}s", flush=True)

    y_true = [d["label"] for d in test_subset]
    report = evaluate(y_true, preds, num_classes)
    report["method"] = "rule_keyword"
    report["sec_per_item"] = round((time.time() - t0) / n_eval, 4) if n_eval else 0.0

    meta = {
        "dataset": DATASET_NAME,
        "split": SPLIT,
        "n_train": len(ds["train"]),
        "n_val": len(ds["validation"]),
        "n_test": len(test),
        "n_eval": n_eval,
        "seed": 42,
        "note": "规则脚本未做去重/清洗，n_dup/n_clean 字段不适用",
    }

    out = Path(__file__).resolve().parent.parent / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"meta": meta, "report": report, "preds": details},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n评测 {n_eval} 条 | acc={report['acc']:.4f} | macro_f1={report['macro_f1']:.4f}")
    print(f"unparsed={report['unparsed']} ({report['unparsed_rate']:.2%})")
    print(f"覆盖类别: {report['pred_distribution']['used_classes']}/{num_classes}")
    print(f"→ 结果写入 {out}")


if __name__ == "__main__":
    main()