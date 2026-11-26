# -*- coding: utf-8 -*-
"""
对比脚本：把 extract.py 跑出来的所有 log.json 汇总成一张表。

用法：
    python compare.py
输出：
    out/compare.md    对比表（Markdown，直接贴进报告）
    out/compare.csv   同一张表的 csv
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent      # 脚本所在的 week7/ 目录
OUT = ROOT / "out"


def load_logs():
    rows = []
    for p in sorted(OUT.glob("*_log.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:      # noqa: BLE001
            continue
        m = obj.get("metrics") or {}
        rows.append({
            "tag": obj.get("tag", p.stem.replace("_log", "")),
            "dataset": obj.get("dataset", ""),
            "provider": obj.get("provider", ""),
            "model": obj.get("model", ""),
            "mode": obj.get("mode", ""),
            "n": obj.get("n", 0),
            "entity_f1": (m.get("entity") or {}).get("f1"),
            "entity_p": (m.get("entity") or {}).get("p"),
            "entity_r": (m.get("entity") or {}).get("r"),
            "trigger_f1": (m.get("trigger") or {}).get("f1"),
            "event_f1": (m.get("event_exact") or {}).get("f1"),
            "hallucination": m.get("hallucination_rate"),
            "json_valid": obj.get("json_valid", 0),
            "parse_fail": obj.get("parse_fail", 0),
            "tokens": obj.get("total_tokens", 0),
            "elapsed_s": obj.get("elapsed_s", 0),
            "is_mock": obj.get("is_mock", False),
        })
    return rows


def fmt(v, nd=4):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main():
    rows = load_logs()
    if not rows:
        print("out/ 下还没有 *_log.json，先跑 extract.py。")
        return

    ds_list = sorted({r["dataset"] for r in rows})
    parts = ["# week7 抽取实验 · 结果对比\n"]
    parts.append("> 全部方案用同一套评测口径；entity F1 是跨数据集可比的主指标。\n")

    for ds in ds_list:
        sub = [r for r in rows if r["dataset"] == ds]
        sub.sort(key=lambda r: -(r["entity_f1"] or 0))
        has_trigger = any(r["trigger_f1"] is not None for r in sub)
        parts.append(f"\n## 数据集：{ds}\n")
        if has_trigger:
            head = ("| 方案 | n | entity F1 | P | R | trigger F1 | event F1 | "
                    "幻觉率 | JSON合法 | token | 耗时(s) |\n"
                    "|---|---|---|---|---|---|---|---|---|---|---|\n")
        else:
            head = ("| 方案 | n | entity F1 | P | R | 幻觉率 | JSON合法 | token | 耗时(s) |\n"
                    "|---|---|---|---|---|---|---|---|---|\n")
        body = ""
        for r in sub:
            name = f"{r['provider']}/{r['model']}·{r['mode']}"
            line = (f"| {name} | {r['n']} | {fmt(r['entity_f1'])} | {fmt(r['entity_p'])} | "
                    f"{fmt(r['entity_r'])} | ")
            if has_trigger:
                line += (f"{fmt(r['trigger_f1'])} | {fmt(r['event_f1'])} | ")
            line += (f"{fmt(r['hallucination'])} | {r['json_valid']}/{r['n']} | "
                     f"{r['tokens']} | {r['elapsed_s']} |")
            if r["is_mock"]:
                line += " ⚠mock"
            body += line + "\n"
        parts.append(head + body)

        real = [r for r in sub if not r["is_mock"]]
        if len(real) >= 2:
            best = real[0]
            gap = (real[0]["entity_f1"] or 0) - (real[1]["entity_f1"] or 0)
            parts.append(f"\n最佳：**{best['provider']}/{best['model']}·{best['mode']}**"
                         f"（entity F1 {best['entity_f1']:.4f}），"
                         f"比第二名高 {gap:.4f}。\n")

    md = OUT / "compare.md"
    md.write_text("\n".join(parts), encoding="utf-8")

    cols = ["tag", "dataset", "provider", "model", "mode", "n", "entity_p", "entity_r",
            "entity_f1", "trigger_f1", "event_f1", "hallucination", "json_valid",
            "parse_fail", "tokens", "elapsed_s", "is_mock"]
    with open(OUT / "compare.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})

    print("\n".join(parts))
    print(f"\n结果 → {md} / {OUT / 'compare.csv'}")


if __name__ == "__main__":
    main()
