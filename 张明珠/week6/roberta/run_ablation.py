"""
RoBERTa 训练技巧消融：一次只改一个旋钮，其余（数据 / seed / epoch / batch / lr / 池化）完全不动。

为什么必须固定其余变量：
    消融实验要回答「**是这个改动带来的差异吗**」。只要同时动了两处，差异就归因不了。
    所以这里所有组都用 seed=42、3 epoch、同一份 train/val，只加一个开关。

为什么每组都要重新算「干净集」指标：
    训练日志里的 val 是原始 val（含约 11% 与训练集重复，分数虚高），组间比较会被泄漏放大。
    统一到 8,901 条去重干净集上评，才和 README 里 0.5548 的口径一致。

为什么要配对检验：
    8,901 条上 Macro-F1 的标准误约 0.6 个点，所以 ±0.5 点以内一律算「没动」。
    这里对每一组都做「同批样本」的 McNemar + 配对 bootstrap，直接给出差值区间。

用法：
    python roberta/run_ablation.py                  # 全套 6 组（约 3 小时）
    python roberta/run_ablation.py --only noclass   # 只跑一组（可重复执行，已有产物会跳过）
    python roberta/run_ablation.py --smoke          # 每 epoch 2 步、只评 300 条，验证流水线（几分钟）

产物：
    outputs/ablation/<tag>/roberta_ckpt.pt   每组一个 ckpt（互不覆盖，主模型 outputs/roberta_ckpt.pt 不受影响）
    outputs/ablation/<tag>/pred_full.json    该组在 8,901 条干净集上的逐条预测
    outputs/ablation/<tag>/train.log         训练日志原文
    outputs/ablation/ablation.json           汇总（含 McNemar p、差值 CI、逐类 recall）
    outputs/ablation/ablation.md             可直接贴进文档的表格
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "eval"))

from compare_backbones import bootstrap_delta, load_preds, mcnemar_exact  # noqa: E402

# master 基线：就是用默认参数训出来的那个 ckpt（outputs/roberta_ckpt.pt），
# 不重训——同 seed 同配方重跑一遍只会得到几乎一样的数，浪费 26 分钟。
BASELINE_PRED = CODE_ROOT / "outputs" / "roberta_pred_full.json"

CONFIGS = [
    # tag,      额外参数,                                   中文说明
    ("baseline", [], "基线：cls 池化 + 类别加权 + CE"),
    ("noclass", ["--no_class_weight"], "去掉类别加权（看长尾类会不会塌）"),
    ("ls01", ["--label_smoothing", "0.1"], "Label Smoothing 0.1"),
    ("focal2", ["--focal_gamma", "2"], "Focal Loss γ=2"),
    ("fgm", ["--fgm"], "FGM 对抗训练 ε=1.0（耗时 ×1.5）"),
    ("mean", ["--pool", "mean"], "mean 池化"),
    ("max", ["--pool", "max"], "max 池化"),
]


def run(cmd, log_path):
    """跑一个子进程，stdout+stderr 全部落日志；失败时把日志尾部打出来。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    with log_path.open("w", encoding="utf-8") as f:
        p = subprocess.run(cmd, cwd=str(CODE_ROOT), stdout=f, stderr=subprocess.STDOUT, env=env)
    if p.returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-15:])
        raise SystemExit(f"[fail] {Path(cmd[1]).name} 退出码 {p.returncode}\n{tail}")


def clean_metrics(pred_json):
    """从预测 json 里取报告；preds 用于配对检验。"""
    d = json.loads(Path(pred_json).read_text(encoding="utf-8"))
    return d["report"], d["preds"]


def pair_vs_baseline(pred_json, boot=2000):
    """同一批样本上和基线做配对比较：McNemar + 差值 CI。"""
    A, _, _ = load_preds(BASELINE_PRED)
    B, _, _ = load_preds(pred_json)
    keys = [k for k in A if k in B]
    if not keys:
        return {}
    import numpy as np

    y = np.array([A[k][0] for k in keys])
    pa = np.array([A[k][1] for k in keys])
    pb = np.array([B[k][1] for k in keys])
    n01 = int(((pa != y) & (pb == y)).sum())      # 基线错、本组对
    n10 = int(((pa == y) & (pb != y)).sum())      # 基线对、本组错
    d = bootstrap_delta(y, pa, pb, times=boot) * 100
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    return {"n_paired": len(keys), "mcnemar": {"baseline_wrong_this_right": n01,
                                               "baseline_right_this_wrong": n10,
                                               "p_exact": mcnemar_exact(n01, n10)},
            "delta_f1_ci95_pt": [lo, hi], "delta_f1_mean_pt": float(d.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="只跑指定 tag（逗号分隔），默认全跑")
    ap.add_argument("--smoke", action="store_true", help="冒烟：1 epoch / 2 步 / 评 300 条")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--force", action="store_true", help="已有产物也重跑")
    args = ap.parse_args()

    tags = [t.strip() for t in args.only.split(",") if t.strip()]
    configs = [c for c in CONFIGS if not tags or c[0] in tags]
    if not configs:
        raise SystemExit(f"--only 里没有匹配的 tag，可选：{[c[0] for c in CONFIGS]}")
    if not BASELINE_PRED.exists():
        raise SystemExit(f"缺少基线预测 {BASELINE_PRED}，先跑 roberta/predict_roberta.py --n 0")

    root = CODE_ROOT / ("outputs/ablation_smoke" if args.smoke else "outputs/ablation")
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    t_all = time.time()

    for i, (tag, extra, desc) in enumerate(configs, 1):
        out = root / tag
        ckpt = out / "roberta_ckpt.pt"
        pred = out / "pred_full.json"
        # 断点续跑：训练产物和预测都在就跳过，长跑中断后不用从头再来
        if not args.force and ckpt.exists() and pred.exists():
            print(f"[{i}/{len(configs)}] {tag}: 已有产物，跳过", flush=True)
        else:
            cmd = [sys.executable, "roberta/train_roberta.py",
                   "--out_dir", str(out.relative_to(CODE_ROOT)),
                   "--epochs", "1" if args.smoke else "3", "--seed", "42", *extra]
            if args.smoke:
                cmd += ["--max_steps", "2", "--max_eval", "300"]
            print(f"\n[{i}/{len(configs)}] {tag}  {desc}\n  " + " ".join(cmd[1:]), flush=True)
            t0 = time.time()
            run(cmd, out / "train.log")
            print(f"  训练完成，用时 {(time.time() - t0) / 60:.1f} 分钟", flush=True)

            pcmd = [sys.executable, "roberta/predict_roberta.py", "--ckpt", str(ckpt),
                    "--n", "300" if args.smoke else "0",
                    "--out", str(pred.relative_to(CODE_ROOT))]
            run(pcmd, out / "predict.log")

        log = json.loads((out / "roberta_train_log.json").read_text(encoding="utf-8"))
        report, _ = clean_metrics(pred)
        pair = {} if tag == "baseline" else pair_vs_baseline(pred, boot=args.boot)
        rows.append({
            "tag": tag, "desc": desc, "args": extra,
            "train_min": round(sum(e["elapsed_s"] for e in log) / 60, 1),
            "val_macro_f1_raw": max(e["val_macro_f1"] for e in log),
            "val_epochs": [e["val_macro_f1"] for e in log],
            "clean_acc": report.get("acc_incl_unparsed", report["acc"]), "clean_macro_f1": report["macro_f1"],
            "clean_ece": report.get("ece"), "pair_vs_baseline": pair,
        })
        (root / "ablation.json").write_text(
            json.dumps({"baseline_pred": str(BASELINE_PRED.relative_to(CODE_ROOT)),
                        "smoke": args.smoke, "rows": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"  干净集 acc={report.get('acc_incl_unparsed', report['acc']):.4f} "
              f"macro-F1={report['macro_f1']:.4f}", flush=True)

    # ---------- 汇总表 ----------
    base = next((r for r in rows if r["tag"] == "baseline"), None)
    if base is None and BASELINE_PRED.exists():
        rep, _ = clean_metrics(BASELINE_PRED)
        base = {"tag": "baseline", "clean_macro_f1": rep["macro_f1"],
                "clean_acc": rep.get("acc_incl_unparsed", rep["acc"])}

    md = ["| 配置 | 干净集 acc | 干净集 Macro-F1 | 相对基线 | 原始 val（best） | 训练时长 |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        delta = "" if not base else f"{100 * (r['clean_macro_f1'] - base['clean_macro_f1']):+.2f} pt"
        if r["tag"] == "baseline":
            delta = "—"
        md.append(f"| {r['desc']} | {r['clean_acc']:.4f} | {r['clean_macro_f1']:.4f} | {delta} "
                  f"| {r['val_macro_f1_raw']:.4f} | {r['train_min']:.0f} 分钟 |")
    table = "\n".join(md)
    (root / "ablation.md").write_text(table + "\n", encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"消融汇总（干净集 n=8901，seed=42）总耗时 {(time.time() - t_all) / 60:.0f} 分钟")
    print(table)
    for r in rows:
        p = r.get("pair_vs_baseline") or {}
        if p:
            print(f"  {r['tag']}: 基线错它对 {p['mcnemar']['baseline_wrong_this_right']} / "
                  f"它对基线错 {p['mcnemar']['baseline_right_this_wrong']}，"
                  f"McNemar p={p['mcnemar']['p_exact']:.3f}，"
                  f"差值 95%CI [{p['delta_f1_ci95_pt'][0]:+.2f}, {p['delta_f1_ci95_pt'][1]:+.2f}] pt")
    print(f"→ {root / 'ablation.json'} / {root / 'ablation.md'}")


if __name__ == "__main__":
    main()
