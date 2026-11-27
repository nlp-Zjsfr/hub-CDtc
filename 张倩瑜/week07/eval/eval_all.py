# -*- coding: utf-8 -*-
"""汇总 L1–L5（及可选规则 L0）的 macro F1 对比表。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

METHODS = [
    ("L0", "规则词典+模板", "rule_metrics.json"),
    ("L1", "BiLSTM-CRF", "l1_bilstm_metrics.json"),
    ("L2", "BERT-Linear BIO", "l2_bert_metrics.json"),
    ("L3", "Qwen zero-shot", "l3_zero_metrics.json"),
    ("L4", "Qwen LoRA SFT", "l4_lora_metrics.json"),
    ("L5", "BERT+规则组合", "l5_service_metrics.json"),
]


def main():
    rows = []
    for level, name, fname in METHODS:
        path = OUT / fname
        if not path.exists():
            rows.append((level, name, None))
            continue
        m = json.loads(path.read_text(encoding="utf-8"))
        rows.append((level, name, m))

    lines = [
        "# Week07 Data Lineage IE — method comparison",
        "",
        "评测集：`data/val.jsonl`。统一口径：`common/evaluate.py`。",
        "",
        "主对比指标：**role_macro_f1**（按论元角色平均的 F1）与 **type_macro_f1**。",
        "",
        "| 层级 | 方法 | trigger F1 | argument F1 | event F1 | type_macro_f1 | role_macro_f1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for level, name, m in rows:
        if m is None:
            lines.append(f"| {level} | {name} | - | - | - | - | - |")
            continue
        lines.append(
            f"| {level} | {name} | {m['trigger']['f1']:.4f} | {m['argument']['f1']:.4f} | "
            f"{m['event_exact']['f1']:.4f} | {m['type_macro_f1']:.4f} | {m['role_macro_f1']:.4f} |"
        )

    # 只对 L1-L5 排序
    scored = [(level, name, m["role_macro_f1"]) for level, name, m in rows
              if m is not None and level != "L0"]
    if scored:
        scored.sort(key=lambda x: x[2], reverse=True)
        lines += ["", "## L1–L5 by role_macro_f1 (desc)", ""]
        for i, (level, name, f1) in enumerate(scored, 1):
            lines.append(f"{i}. {level} {name}: **{f1:.4f}**")

    text = "\n".join(lines) + "\n"
    (OUT / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
