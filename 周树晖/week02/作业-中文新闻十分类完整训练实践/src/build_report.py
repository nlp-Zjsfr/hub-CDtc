"""用已有实验结果生成根目录README和images中的实验图。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from .common import ROOT, read_json

LABELS = {"linear_adam": "单层分类器", "mlp_reference": "MLP基础设置", "mlp_lr_small": "学习率1e-4",
          "mlp_lr_large": "学习率1e-2", "mlp_batch64": "批大小64", "mlp_dropout": "Dropout 0.2",
          "mlp_weight_decay": "权重衰减", "mlp_scheduler": "学习率调度", "mlp_combined": "组合设置"}
TEAL, ORANGE = "#087f8c", "#d87832"


def pct(value):
    return f"{value * 100:.2f}%"


def plot_figures(audit, summaries, selection, per_class, cm):
    for font in [Path("C:/Windows/Fonts/msyh.ttc"), Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc")]:
        if font.exists():
            font_manager.fontManager.addfont(str(font))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
            break
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.unicode_minus": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    folder = ROOT / "images"
    folder.mkdir(parents=True, exist_ok=True)
    classes = audit["classes"]
    chosen = selection["main_model"]["name"]

    def save(fig, filename):
        fig.savefig(folder / filename, dpi=170, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    values = [audit["clean_class_counts"]["train"][str(i)] for i in range(10)]
    bars = ax.bar(classes, values, color=TEAL, width=.68)
    ax.bar_label(bars, [f"{v:,}" for v in values], padding=4, fontsize=9)
    ax.set(title="训练集类别分布", ylabel="标题数量", ylim=(0, max(values) * 1.13))
    save(fig, "01_class_distribution.png")

    history = read_json(ROOT / "outputs/experiments" / chosen / "history.json")
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    epochs = [row["epoch"] for row in history]
    ax.plot(epochs, [row["train_loss"] for row in history], "o-", color=TEAL, markersize=4, label="训练损失")
    ax.plot(epochs, [row["val_loss"] for row in history], "o-", color=ORANGE, markersize=4, label="验证损失")
    ax.axvline(selection["main_model"]["best_epoch"], color="#657580", linestyle=":", label="保留的模型轮次")
    ax.set(title="模型训练损失", xlabel="Epoch", ylabel="交叉熵损失")
    ax.legend()
    ax.grid(alpha=.15)
    save(fig, "08_final_model_loss.png")

    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    values = [item["best_val_macro_f1"] * 100 for item in summaries]
    colors = [TEAL if item["name"] == chosen else "#b2c3d1" for item in summaries]
    bars = ax.barh([LABELS[item["name"]] for item in summaries], values, color=colors)
    ax.bar_label(bars, [f"{v:.2f}%" for v in values], padding=5)
    ax.set(title="不同训练设置的验证结果", xlabel="验证集 Macro-F1 (%)",
           xlim=(min(values) - .45, max(values) + .42))
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.15)
    save(fig, "04_experiment_comparison.png")

    normalized = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 6.8), layout="constrained")
    im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(10), classes, rotation=40, ha="right")
    ax.set_yticks(range(10), classes)
    ax.set(title="测试集混淆矩阵", xlabel="预测类别", ylabel="真实类别")
    for i in range(10):
        for j in range(10):
            if normalized[i, j] >= .01 or i == j:
                ax.text(j, i, f"{normalized[i, j] * 100:.1f}%", ha="center", va="center", fontsize=8,
                        color="white" if normalized[i, j] > .55 else "#152f42")
    fig.colorbar(im, ax=ax, shrink=.8, label="比例")
    save(fig, "05_confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    x = np.arange(10)
    for shift, key, label, color in [(-.24, "precision", "Precision", "#b2c3d1"),
                                    (0, "recall", "Recall", TEAL), (.24, "f1-score", "F1", ORANGE)]:
        ax.bar(x + shift, [per_class[c][key] * 100 for c in classes], .23, label=label, color=color)
    ax.set_xticks(x, classes)
    ax.set(title="测试集各类别表现", ylabel="百分比 (%)", ylim=(0, 103))
    ax.legend(ncol=3, loc="lower right")
    save(fig, "06_per_class_metrics.png")


class Report:
    def __init__(self):
        self.blocks = []

    def heading(self, title):
        self.blocks.append(("h", title))

    def p(self, text):
        self.blocks.append(("p", text))

    def table(self, columns, rows):
        self.blocks.append(("table", (columns, [[str(x) for x in row] for row in rows])))

    def code(self, text):
        self.blocks.append(("code", text))

    def image(self, filename, caption):
        self.blocks.append(("image", (filename, caption)))

    def write(self, title):
        md = [f"# {title}\n"]
        for kind, data in self.blocks:
            if kind == "h":
                md.append(f"\n## {data}\n")
            elif kind == "p":
                md.append(f"\n{data}\n")
            elif kind == "code":
                md.append(f"\n```text\n{data}\n```\n")
            elif kind == "table":
                columns, rows = data
                md.append("\n| " + " | ".join(columns) + " |\n| " + " | ".join("---" for _ in columns) + " |\n" +
                          "\n".join("| " + " | ".join(row) + " |" for row in rows) + "\n")
            elif kind == "image":
                filename, caption = data
                md.append(f"\n![{caption}](images/{filename})\n")
        (ROOT / "README.md").write_text("\n".join(md), encoding="utf-8")


def build_report():
    audit = read_json(ROOT / "outputs/data_audit.json")
    summaries = read_json(ROOT / "outputs/experiment_summary.json")
    selection = read_json(ROOT / "outputs/selection.json")
    evaluation = read_json(ROOT / "outputs/evaluation/final_metrics.json")
    per_class = read_json(ROOT / "outputs/evaluation/classification_report.json")
    cm = np.asarray(read_json(ROOT / "outputs/evaluation/confusion_matrix.json"))
    selected = selection["main_model"]
    cfg = selected["config"]
    main = evaluation["metrics"]["selected_mlp"]
    plot_figures(audit, summaries, selection, per_class, cm)
    r = Report()

    r.heading("一、数据集与划分")
    r.p("实验使用[公开的THUCNews新闻标题子集](https://github.com/649453932/Chinese-Text-Classification-Pytorch)，"
        "共20万条标题，分为财经、房产、股票、教育、科技、社会、时政、体育、游戏和娱乐10类。每条标题对应一个类别，任务是根据标题判断所属栏目。")
    r.p("沿用数据集原有的训练、验证和测试划分，去除重复标题及标签冲突记录。词表和IDF只由训练集计算，模型及参数通过验证集选择，测试集用于最终评估。")
    r.table(["数据划分", "原始数量", "清洗后数量"], [[label, f"{audit['raw_sizes'][split]:,}", f"{audit['clean_sizes'][split]:,}"]
            for split, label in [("train", "训练集"), ("val", "验证集"), ("test", "测试集")]])
    r.image("01_class_distribution.png", "图1 训练集类别分布")

    r.heading("二、模型结构")
    r.p("先对标题做全半角、英文大小写和空白规范化，再以单字和相邻双字构建TF-IDF特征，将每条标题转换为5000维向量。分类模型采用两层隐藏层的多层感知机，隐藏层使用ReLU，输出10个类别分数。")
    r.code("TF-IDF（5000维） → 全连接层（128） → 全连接层（64） → 输出层（10）")

    r.heading("三、训练方法")
    r.p(f"训练采用交叉熵损失和Adam优化器，初始学习率为{cfg['lr']:g}，批大小为{cfg['batch_size']}，"
        f"每轮打乱训练样本，最多训练{cfg['max_epochs']}轮。隐藏层加入Dropout（{cfg['dropout']:g}），权重衰减设为{cfg['weight_decay']:g}。")
    r.p("验证集Macro-F1进入平台期后将学习率减半；连续4轮未显著改善时提前停止。实验比较了学习率、批大小、Dropout、权重衰减及学习率调度等9组设置，按验证集Macro-F1保存最佳模型。")

    r.heading("四、实验结果")
    r.p(f"组合设置取得了最高的验证集Macro-F1（{pct(selected['best_val_macro_f1'])}）。"
        f"模型实际训练{selected['epochs_ran']}轮，保留第{selected['best_epoch']}轮的权重。下图展示训练过程、参数对照和测试集表现。")
    r.image("08_final_model_loss.png", "图2 主模型训练与验证损失")
    r.image("04_experiment_comparison.png", "图3 不同训练设置的验证集Macro-F1（横轴局部放大）")
    r.image("05_confusion_matrix.png", "图4 测试集混淆矩阵，按真实类别归一化")
    r.image("06_per_class_metrics.png", "图5 测试集各类别的Precision、Recall和F1")

    r.heading("五、实验结论")
    r.p(f"本次实验采用TF-IDF和多层感知机完成中文新闻十分类。在{evaluation['n_test']:,}条测试标题中，"
        f"正确分类{main['correct']:,}条，准确率为{pct(main['accuracy'])}，Macro-F1为{main['macro_f1']:.4f}，"
        f"平均交叉熵损失为{evaluation['main_test_mean_cross_entropy_torch']:.4f}。"
        f"本次比较中，加入Dropout、权重衰减和学习率调度的组合设置表现最好，最终使用的是第{selected['best_epoch']}轮模型。")

    r.write("基于TF-IDF与多层感知机的十分类实验")
    print("实验报告已更新至README.md，图表保存在images目录。", flush=True)


if __name__ == "__main__":
    build_report()
