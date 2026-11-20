"""把 L0–L5 的预测汇总成 Micro-F1 对比表。

优先用预测文件重算，保证各层用的是同一个 classification_report。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.labels import LABELS
from common.metrics import classification_report, to_markdown_table

OUT = Path(__file__).resolve().parent.parent / "outputs"

# (层级, 显示名, 预测文件, 指标文件)
ROWS = [
    ("L0", "规则 / 关键词", "rule_pred.json", "rule_metrics.json"),
    ("L1", "TF-IDF + LinearSVC", "tfidf_pred.json", "tfidf_metrics.json"),
    ("L2", "BERT 微调", "bert_pred.json", "bert_metrics.json"),
    ("L3", "Qwen2.5-0.5B zero-shot", "zero_shot_pred.json", "zero_shot_metrics.json"),
    ("L4", "Qwen2.5-0.5B + LoRA", "lora_pred.json", "lora_metrics.json"),
    ("L5", "分流 CLI", "router_pred.json", "router_metrics.json"),
]


def score_file(path: Path) -> dict:
    rows = json.loads(path.read_text(encoding="utf-8"))
    ys = [r["gold"] for r in rows]
    ps = [r["pred"] for r in rows]
    report = classification_report(ys, ps, num_classes=len(LABELS))
    report["n_pred_file"] = len(rows)
    return report


def main():
    results = []
    for level, name, pred_name, metrics_name in ROWS:
        pred_path = OUT / pred_name
        metrics_path = OUT / metrics_name
        if pred_path.exists():
            report = score_file(pred_path)
            if metrics_path.exists():
                old = json.loads(metrics_path.read_text(encoding="utf-8"))
                old.update({k: report[k] for k in ("micro_f1", "macro_f1", "acc", "acc_incl_unparsed", "unparsed") if k in report})
                metrics_path.write_text(json.dumps(old, ensure_ascii=False, indent=2), encoding="utf-8")
        elif metrics_path.exists():
            print(f"[fallback] {pred_name} missing, use {metrics_name}")
            report = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            print(f"[skip] missing {pred_name}")
            continue
        results.append((level, name, report))
    md = to_markdown_table(results)
    (OUT / "summary.md").write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
