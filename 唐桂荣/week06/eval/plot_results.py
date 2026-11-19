"""从验证和测试结果生成可用于作业总结的 PNG 图。"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
from matplotlib.lines import Line2D


OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"
MODELS = (
    ("bert-base-chinese", "BERT", "#2769A9", "o"),
    ("roberta_base", "RoBERTa", "#AD4E29", "D"),
)
METHODS = (
    ("TF-IDF + LinearSVC", "#19756D", "s"),
    *((label, color, marker) for _, label, color, marker in MODELS),
)
INK = "#172B3B"
MUTED = "#536474"
GRID = "#E4EAF0"


def configure_style(font_path):
    mac_font = Path("/System/Library/Fonts/STHeiti Medium.ttc")
    font_path = font_path or (mac_font if mac_font.is_file() else None)
    if font_path:
        font_manager.fontManager.addfont(font_path)
        family = font_manager.FontProperties(fname=font_path).get_name()
    else:
        preferred = ("Noto Sans CJK SC", "Source Han Sans SC", "Microsoft YaHei", "SimHei")
        installed = {entry.name for entry in font_manager.fontManager.ttflist}
        family = next((name for name in preferred if name in installed), None)
        if family is None:
            raise RuntimeError("请用 --font 指定一个支持中文的本地字体文件")
    plt.rcParams.update({
        "font.family": [family, "DejaVu Sans"],
        "font.size": 11,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "axes.unicode_minus": False,
        "savefig.dpi": 180,
    })


def add_header(fig, title, subtitle):
    top = 0.944
    fig.text(0.045, top, title, fontsize=24, va="top")
    fig.text(
        0.045, top - 0.44 / fig.get_figheight(), subtitle,
        fontsize=11.5, color=MUTED, va="top",
    )


def format_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=0.9)
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0, pad=8)


def save(fig, out, name):
    fig.savefig(out / name, facecolor="white")
    plt.close(fig)


def plot_ablation(validation, out):
    experiments = list(validation["experiments"].items())
    positions = list(range(len(experiments)))
    labels = [f"{exp_id}  {row['description']}" for exp_id, row in experiments]
    tfidf = validation["tfidf_macro_f1"]
    deltas = [
        point["change_from_exp1_pp"]
        for _, row in experiments for point in row["models"].values()
    ]
    fig, (scores, changes) = plt.subplots(
        1, 2, figsize=(12.2, 7.1), sharey=True, gridspec_kw={"width_ratios": [1.1, 1]},
    )
    fig.subplots_adjust(left=0.218, right=0.955, bottom=0.137, top=0.719, wspace=0.29)
    add_header(
        fig, "九组消融的验证集对比",
        f"{validation['n']:,} 条共用验证样本 · exp2–exp9 各自只改动 exp1 的一项设置",
    )
    for ax in (scores, changes):
        format_axis(ax)
        ax.set_ylim(len(positions) - 0.5, -0.5)
        ax.set_yticks(positions, labels)
        for position in positions[:-1]:
            ax.axhline(position + 0.5, color=GRID, linewidth=0.55, zorder=0)

    scores.axvline(tfidf, color=MUTED, linewidth=1.25, linestyle=(0, (4, 3)), zorder=2)
    changes.axvline(0, color=MUTED, linewidth=1.2, zorder=2)
    handles = []
    for (model, label, color, marker), offset in zip(MODELS, (-0.17, 0.17), strict=True):
        y = [pos + offset for pos in positions]
        model_rows = [row["models"][model] for _, row in experiments]
        absolute = [point["macro_f1"] for point in model_rows]
        delta = [point["change_from_exp1_pp"] for point in model_rows]
        scores.scatter(absolute, y, color=color, marker=marker, s=38, zorder=3)
        changes.barh(y, delta, color=color, height=0.22, zorder=3)
        for pos, value in zip(y, delta, strict=True):
            if value == 0:
                changes.scatter(0, pos, color=color, marker=marker, s=13, zorder=4)
                continue
            changes.annotate(
                f"{value:+.2f}", (value, pos), xytext=(5 if value > 0 else -5, 0),
                textcoords="offset points",
                ha="left" if value > 0 else "right", va="center", fontsize=9.5, color=color,
            )
        handles.append(Line2D([], [], linestyle="none", marker=marker, color=color, label=label))
    handles.append(Line2D([], [], linestyle="--", color=MUTED, label=f"TF-IDF 验证基线 {tfidf:.4f}"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.045, 0.832), ncol=3, frameon=False)

    scores.set_title("验证集 Macro-F1", loc="left", fontsize=13, pad=11)
    scores.set_xlim(0.961, 0.985)
    scores.set_xticks((0.962, 0.967, 0.972, 0.977, 0.982))
    scores.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    changes.set_title("相对各自 exp1 的变化", loc="left", fontsize=13, pad=11)
    changes.set_xlim(min(deltas) - 0.21, max(deltas) + 0.21)
    changes.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
    changes.xaxis.set_major_formatter(ticker.FuncFormatter(lambda value, _: f"{value:+.2f}" if value else "0"))
    changes.set_xlabel("Macro-F1 百分点")
    save(fig, out, "ablation_validation.png")


def plot_test_scores(test, out):
    labels = (
        METHODS[0][0],
        *(f"{method} · {test['methods'][method]['experiment']}" for method, _, _ in METHODS[1:]),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.35), sharey=True)
    fig.subplots_adjust(left=0.192, right=0.95, bottom=0.20, top=0.71, wspace=0.33)
    add_header(
        fig, "最终测试的三种方法对比",
        f"同一批 {test['n']:,} 条去重测试样本 · BERT 和 RoBERTa 各用验证集选定的配置",
    )
    for ax, title, metric in zip(axes, ("Macro-F1", "准确率"), ("macro_f1", "accuracy"), strict=True):
        format_axis(ax)
        ax.set_ylim(2.55, -0.55)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_title(title, loc="left", fontsize=14, pad=11)
        ax.set_xlim(0.965, 0.995)
        ax.set_xticks((0.965, 0.975, 0.985, 0.995))
        if metric == "accuracy":
            ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
        else:
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
        for position, (method, color, marker) in enumerate(METHODS):
            value = test["methods"][method]["metrics"][metric]
            ax.axhline(position, color=GRID, linewidth=0.8, zorder=0)
            ax.scatter(value, position, color=color, marker=marker, s=85, zorder=3)
            ax.annotate(
                f"{value:.2%}" if metric == "accuracy" else f"{value:.4f}",
                (value, position), xytext=(9, 0), textcoords="offset points",
                va="center", fontsize=11, color=color,
            )
    fig.text(0.045, 0.055, "横轴局部放大，便于读出接近的分数。", color=MUTED, fontsize=9.5)
    save(fig, out, "final_test_metrics.png")


def plot_training_time(validation, out):
    experiments = ("exp1", "exp6", "exp7")
    labels = ("exp1  基线（长度 64）", "exp6  最大长度 32", "exp7  FGM")
    fig, ax = plt.subplots(figsize=(10.3, 4.85))
    fig.subplots_adjust(left=0.217, right=0.95, bottom=0.175, top=0.726)
    add_header(
        fig, "缩短输入与加入 FGM 的训练耗时",
        "本机 MPS 上三轮训练的合计耗时 · 括号内是相对各自 exp1 的变化",
    )
    format_axis(ax)
    for (model, label, color, _), offset in zip(MODELS, (-0.17, 0.17), strict=True):
        values = [validation["experiments"][exp]["models"][model]["training_seconds"] / 60 for exp in experiments]
        y = [position + offset for position in range(len(labels))]
        ax.barh(y, values, height=0.27, color=color, label=label)
        for position, value, exp in zip(y, values, experiments, strict=True):
            suffix = "" if exp == "exp1" else f"（{value / values[0] - 1:+.1%}）"
            ax.annotate(
                f"{value:.1f}{suffix}", (value, position), xytext=(7, 0),
                textcoords="offset points", va="center", fontsize=10, color=color,
            )
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 42)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.set_xlabel("训练时间（分钟）")
    ax.legend(loc="upper right", frameon=False, ncol=2)
    save(fig, out, "training_time.png")


def plot_juice_confusions(test, out):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.75), sharey=True)
    fig.subplots_adjust(left=0.195, right=0.96, bottom=0.174, top=0.69, wspace=0.31)
    add_header(
        fig, "果汁误判集中在哪个方向",
        f"同一批 {test['n']:,} 条去重测试样本 · 比例以真实类别的样本数为分母",
    )
    cold, ambient = "冷藏即饮果汁", "非冷藏即饮果汁"
    directions = (
        (cold, ambient, "冷藏判为非冷藏"),
        (ambient, cold, "非冷藏判为冷藏"),
    )
    for ax, (actual, predicted, title) in zip(axes, directions, strict=True):
        actual_index = test["labels"].index(actual)
        predicted_index = test["labels"].index(predicted)
        support = test["methods"][METHODS[0][0]]["metrics"]["per_class"][actual]["support"]
        format_axis(ax)
        ax.set_title(f"{title}（{support} 条真实样本）", loc="left", fontsize=13, pad=12)
        for position, (method, color, _) in enumerate(METHODS):
            metrics = test["methods"][method]["metrics"]
            count = metrics["confusion_matrix"][actual_index][predicted_index]
            rate = count / support
            ax.barh(position, rate, height=0.48, color=color)
            ax.annotate(
                f"{rate:.2%}（{count} 条）", (rate, position), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=10, color=color,
            )
        ax.set_xlim(0, 0.192)
        ax.set_ylim(2.60, -0.60)
        ax.set_yticks(range(len(METHODS)), [method for method, _, _ in METHODS])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_xlabel("占真实类别的比例")
    save(fig, out, "juice_confusions.png")


def main():
    parser = argparse.ArgumentParser(description="绘制作业验证集与测试集结果")
    parser.add_argument("--font", type=Path, help="可选的中文字体文件")
    parser.add_argument("--out", type=Path, default=OUTPUTS / "figures")
    args = parser.parse_args()
    configure_style(args.font)
    validation = json.loads((OUTPUTS / "eval" / "validation.json").read_text(encoding="utf-8"))
    test = json.loads((OUTPUTS / "eval" / "test_comparison.json").read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    plot_ablation(validation, args.out)
    plot_test_scores(test, args.out)
    plot_training_time(validation, args.out)
    plot_juice_confusions(test, args.out)
    print(f"图表已保存到 {args.out}")


if __name__ == "__main__":
    main()
