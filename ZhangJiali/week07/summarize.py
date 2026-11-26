# -*- coding: utf-8 -*-
"""读取各方法评测 JSON，生成对比表与柱状图。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import METRIC_DIR, OUTPUT_DIR

METHODS = [
    ("rule", "规则基线"),
    ("bilstm_crf", "BiLSTM-CRF"),
    ("bert_linear", "BERT+Linear"),
]


def load_metric(name: str) -> dict | None:
    path = METRIC_DIR / f"{name}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    rows = []
    for name, label in METHODS:
        result = load_metric(name)
        if result is None:
            print(f"缺少指标文件：{name}.json")
            continue
        meta = result.get("meta") or {}
        rows.append({
            "name": name,
            "label": label,
            "trigger": result["trigger"]["f1"],
            "argument": result["argument"]["f1"],
            "event": result["event_exact"]["f1"],
            "type_macro": result.get("type_macro_f1", 0.0),
            "train_s": meta.get("train_s", "-"),
            "ms": meta.get("ms_per_sentence", "-"),
            "params_m": meta.get("params_m", "-"),
        })

    if not rows:
        raise SystemExit("未找到任何评测结果，请先运行三条实验脚本。")

    table_path = METRIC_DIR / "comparison_table.md"
    lines = [
        "# 方法对比",
        "",
        "| 方法 | 触发词 F1 | 论元 F1 | 事件级 F1 | 类型 macro F1 | 训练时间/s | 推理 ms/句 | 参数量/M |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['trigger']:.4f} | {row['argument']:.4f} | "
            f"{row['event']:.4f} | {row['type_macro']:.4f} | {row['train_s']} | "
            f"{row['ms']} | {row['params_m']} |"
        )
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("written:", table_path)

    fig_dir = OUTPUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    labels = [row["label"] for row in rows]
    x = range(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=150)
    ax.bar([i - width for i in x], [row["trigger"] for row in rows], width, label="触发词 F1")
    ax.bar(list(x), [row["argument"] for row in rows], width, label="论元 F1")
    ax.bar([i + width for i in x], [row["event"] for row in rows], width, label="事件级 F1")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1")
    ax.set_title("金融公告事件抽取方法对比")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig_path = fig_dir / "method_comparison.png"
    fig.savefig(fig_path)
    plt.close(fig)
    print("written:", fig_path)


if __name__ == "__main__":
    main()
