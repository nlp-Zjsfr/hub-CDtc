# -*- coding: utf-8 -*-
"""
画图：读 out/*_log.json，产出对比图 compare.png（PPT 直接用）。

用法：
    python plot.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent      # 脚本所在的 week7/ 目录
OUT = ROOT / "out"

try:
    import matplotlib
except ImportError:
    print("缺少 matplotlib：pip install matplotlib")
    sys.exit(1)
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib import font_manager                  # noqa: E402

# 中文字体（探测一下，避免一屏 findfont 警告）
_available = {f.name for f in font_manager.fontManager.ttflist}
for _f in ("SimHei", "Microsoft YaHei", "SimSun"):
    if _f in _available:
        matplotlib.rcParams["font.sans-serif"] = [_f]
        break
else:
    print("[warn] 没找到中文字体，图的中文会显示成方块")
matplotlib.rcParams["axes.unicode_minus"] = False


def main():
    logs = []
    for p in sorted(OUT.glob("*_log.json")):
        try:
            logs.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:      # noqa: BLE001
            continue
    logs = [x for x in logs if not x.get("is_mock")]
    if not logs:
        print("out/ 下没有可用的 *_log.json（或只有 mock 结果），先跑 extract.py。")
        return

    # 按数据集分组画：每组一张图（entity F1 + 幻觉率）
    groups: dict[str, list] = {}
    for x in logs:
        groups.setdefault(x.get("dataset", ""), []).append(x)

    fig, axes = plt.subplots(1, len(groups), figsize=(6.5 * len(groups), 4.8))
    if len(groups) == 1:
        axes = [axes]

    for ax, (ds, items) in zip(axes, groups.items()):
        items.sort(key=lambda x: -(x["metrics"]["entity"]["f1"] or 0))
        names = [f"{x['provider']}/{x['model']}\n{x['mode']}" for x in items]
        vals = [x["metrics"]["entity"]["f1"] for x in items]
        bars = ax.bar(range(len(vals)), vals, color="#4c72b0")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8)
        ax.set_ylabel("entity F1")
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{ds}（n={items[0]['n']}）")
        ax.grid(axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                    ha="center", fontsize=8)

    fig.suptitle("抽取效果对比（统一 entity F1 口径）", fontsize=13)
    fig.tight_layout()
    p = OUT / "compare.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    print(f"图 → {p}")


if __name__ == "__main__":
    main()
