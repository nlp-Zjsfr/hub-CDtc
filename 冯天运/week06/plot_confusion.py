"""
作业 A 配套：混淆矩阵热力图（行=真实，列=预测，右侧标注每行 recall）

【这个脚本在干什么】
读评测结果里的 confusion_matrix，画两张图：
  1) 15 类矩阵（outputs/混淆矩阵.png）
  2) 业务分组 4 组矩阵（outputs/混淆矩阵_4组.png）
两张图一对照，就能说清「错在哪」以及「错得严不严重」。

用法（在作业目录下执行）：
  python plot_confusion.py                 # 15 类矩阵（读 code/outputs/bert_pred.json）→ outputs/混淆矩阵.png
  python plot_confusion.py --group         # 业务分组 4 组矩阵 → outputs/混淆矩阵_4组.png
  python plot_confusion.py --pred code/outputs/xxx.json --out outputs/混淆矩阵.png
"""
import argparse, json, sys                # 命令行参数 / 读 JSON / 模块搜索路径
from pathlib import Path                   # 路径处理

import matplotlib                         # 绘图库
matplotlib.use("Agg")                     # ★无界面后端：服务器/IDE 里也能出图（必须放在 pyplot 之前）
import matplotlib.pyplot as plt            # 绘图接口
import numpy as np                        # 数组运算

HERE = Path(__file__).resolve().parent    # 作业目录
CODE = HERE / "code"                      # 代码目录
sys.path.insert(0, str(CODE))             # 让 common.* 可导入
from common.labels import LABELS, GROUP_MAP, GROUPS   # 15 类名 + 组映射

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
# ↑ 指定中文字体，否则图上的中文会变成方框（Windows 上一般有微软雅黑）
plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示


def to_group_cm(cm):
    """把 15×15 的混淆矩阵按业务组聚合成 4×4（块求和）"""
    idx = {g: [i for i, l in enumerate(LABELS) if GROUP_MAP[l] == g] for g in GROUPS}
    # ↑ 每个组包含哪些 15 类 id
    M = [[sum(cm[i][k] for i in idx[a] for k in idx[b]) for b in GROUPS] for a in GROUPS]
    # ↑ 块求和：行组 a、列组 b 的所有格子加总
    return M, GROUPS


def draw(M, names, out, title):
    """画一张热力图：格子=样本数，右侧一列=每类 recall"""
    n = len(names)                         # 类别数（15 或 4）
    M = np.array(M, dtype=float)           # 转 numpy 方便算
    rec = [M[i][i] / M[i].sum() if M[i].sum() else 0.0 for i in range(n)]
    # ↑ recall = 对角线 / 该行总和（该类真值里有多少被判对）
    fig, ax = plt.subplots(figsize=(1.05 * n + 3.4, 0.95 * n + 2.6), dpi=160)
    im = ax.imshow(M, cmap="Blues")       # 蓝色热力图
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right")   # x 轴 = 预测（旋转防重叠）
    ax.set_yticks(range(n))
    ax.set_yticklabels(names)              # y 轴 = 真实
    ax.set_xlabel("预测")
    ax.set_ylabel("真实")
    ax.set_title(title)
    thr = M.max() * 0.55 if M.max() else 0  # 深色格子用白字、浅色用深字，保证可读
    for i in range(n):
        for k in range(n):
            if M[i][k] > 0:                # 0 不标注（标了全是 0，画面很脏）
                ax.text(k, i, int(M[i][k]), ha="center", va="center",
                        color="white" if M[i][k] > thr else "#333333", fontsize=8)
    for i in range(n):
        # 右侧一列标注该行 recall（红色 = 重点关注）
        ax.text(n - 0.3, i, "recall %.2f" % rec[i], va="center", ha="left", fontsize=8, color="#b03030")
    ax.set_xlim(-0.5, n + 1.0)            # 右侧留出写 recall 的空间
    fig.colorbar(im, ax=ax, shrink=0.8, label="样本数")   # 色条
    fig.tight_layout()                     # 自动排版，避免标签被裁掉
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")  # 存 PNG
    print("已保存", out)
    print("各类 recall:", {names[i]: round(rec[i], 4) for i in range(n)})   # 终端也打一份，方便复制


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(CODE / "outputs" / "bert_pred.json"))  # 输入结果 JSON
    ap.add_argument("--out", default=None, help="输出 PNG 路径，默认 outputs/混淆矩阵[_4组].png")
    ap.add_argument("--group", action="store_true", help="画业务分组（4 组）矩阵")
    args = ap.parse_args()
    if args.out is None:
        args.out = str(HERE / "outputs" / ("混淆矩阵_4组.png" if args.group else "混淆矩阵.png"))
    j = json.loads(Path(args.pred).read_text(encoding="utf-8"))
    cm = j["report"]["confusion_matrix"]   # 15×15 混淆矩阵（行=真实，列=预测）
    title = "BERT 微调：15 类混淆矩阵（去重验证集）"
    if args.group:
        cm, names = to_group_cm(cm)        # 聚合成 4×4
        title = "BERT 微调：业务分组（4 组）混淆矩阵"
    else:
        names = LABELS
    draw(cm, names, args.out, title)       # 画图并落盘


if __name__ == "__main__":
    main()
