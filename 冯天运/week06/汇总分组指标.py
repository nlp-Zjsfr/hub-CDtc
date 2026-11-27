"""
实验 C（更换任务口径）的可复现实现：
把同一份 15 类预测（code/outputs/bert_pred.json）重算成
  1) 15 类口径
  2) 4 组业务口径（金融/产业/宏观/噪音）
  3) 相关性二分类（金融+产业 = 相关）
并统计误差结构：错分里有多少是「跨业务组」、多少是「同业务组」。

【这个脚本在干什么】
训练一次，得到 15 个类别的预测；换业务口径时**不需要再训练**，
只要把已经预测好的标签按「业务组映射表」重新归并，再算一遍指标即可。
这就是它零训练成本的原因，也是它能跟实验 A / B 共用同一份预测文件的原因。

【为什么口径要统一】
15 类 acc 0.53 → 4 组 acc 0.68 → 相关性 acc 0.85，
数字变好不是因为模型变强，而是因为「颗粒度变粗、容错变多」。
所以横向对比必须注明口径，否则 0.85 和别的方法的 0.53 比是没有意义的。

【误差结构为什么要看】
错分分两种：
  跨组错分（把金融新闻说成宏观）→ 业务上是硬错误，必须修
  同组错分（把「基金」说成「理财」，仍在金融组）→ 业务上可接受
两者比例决定了这套系统该往哪个方向优化。

用法（在作业目录下执行）：
  python 汇总分组指标.py                       # 默认读 code/outputs/bert_pred.json（全量干净集）
  python 汇总分组指标.py --pred code/outputs/xxx.json
输出：outputs/group_metrics.json + 终端 Markdown 表
"""
import argparse                                   # 命令行参数
import json                                       # 读预测文件、写结果文件
import sys                                        # 需要把 code/ 加进 sys.path 才能 import common.*
from collections import Counter                   # 计数：统计错分对、每类样本数
from pathlib import Path                         # 路径处理（跨平台）

HERE = Path(__file__).resolve().parent            # 本脚本所在目录（作业目录）
CODE = HERE / "code"                             # 代码目录
sys.path.insert(0, str(CODE))                    # ★把 code/ 插到模块搜索路径最前面，
                                                #   这样下面 import common.labels 才找得到

from common.labels import LABELS, GROUP_MAP, GROUPS, RELEVANT_GROUPS  # noqa: E402
# LABELS        : 15 个类名（顺序 = 标签 id 顺序，全局约定，改了就全乱）
# GROUP_MAP    : 类名 → 业务组（金融/产业/宏观/噪音） 的映射表
# GROUPS       : 4 个业务组的名字，顺序 = 组 id
# RELEVANT_GROUPS: 哪几个组算「相关」（金融+产业），用于二分类口径


def macro_f1(y, p, n):
    """宏平均 F1：先算每一类的 F1，再对 n 个类求平均（类别不均衡时比 acc 更能反映真实水平）"""
    from sklearn.metrics import f1_score
    # labels=range(n)：即使某类在预测里一次都没出现，也照样给它算一列 F1（zero_division=0 → 记 0 分）
    #   如果不指定 labels，sklearn 会自动只统计"出现过"的类，类数会变，跨口径就不可比了
    return round(float(f1_score(y, p, average="macro", labels=list(range(n)), zero_division=0)), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(CODE / "outputs" / "bert_pred.json"))  # 输入：预测明细 JSON
    ap.add_argument("--out", default=str(HERE / "outputs" / "group_metrics.json"))  # 输出：三口径指标 JSON
    ap.add_argument("--top", type=int, default=15)   # 列出前 N 个最易混的对
    args = ap.parse_args()

    # ---- 0) 读入预测明细（不重新跑模型，这是"零训练成本"的体现）
    d = json.loads(Path(args.pred).read_text(encoding="utf-8"))
    items = d["preds"]                    # 逐条明细：[{"text","true","pred"}, ...]
    y_true = [it["true"] for it in items]  # 真值标签（整数 id）
    y_pred = [it["pred"] for it in items]  # 预测标签（整数 id）
    n = len(items)                        # 样本总数（干净集 = 去重后的验证集）

    # ---- 1) 15 类口径（原始任务，最细）
    from sklearn.metrics import accuracy_score
    m15 = {"acc": round(float(accuracy_score(y_true, y_pred)), 4),   # 准确率 = 对的比例
           "macro_f1": macro_f1(y_true, y_pred, len(LABELS))}        # 15 类宏平均 F1

    # ---- 2) 4 组业务口径（把 15 类按 GROUP_MAP 归并成 4 组）
    g2i = {g: i for i, g in enumerate(GROUPS)}                       # 组名 → 组 id
    gy_true = [g2i[GROUP_MAP[LABELS[t]]] for t in y_true]           # 真值：15 类 id → 组 id
    gy_pred = [g2i[GROUP_MAP[LABELS[p]]] for p in y_pred]           # 预测：同样归并
    m4 = {"acc": round(float(accuracy_score(gy_true, gy_pred)), 4),
          "macro_f1": macro_f1(gy_true, gy_pred, len(GROUPS))}
    # 每组 recall = 该组里被判对的条数 / 该组真实条数
    m4["per_group_recall"] = {
        GROUPS[i]: round(float(sum(1 for a, b in zip(gy_true, gy_pred) if a == i and b == i)
                               / max(1, sum(1 for a in gy_true if a == i))), 4)
        for i in range(len(GROUPS))}

    # ---- 3) 相关性二分类（金融+产业 = 相关=1，其它 = 不相关=0）
    ry_true = [1 if GROUP_MAP[LABELS[t]] in RELEVANT_GROUPS else 0 for t in y_true]
    ry_pred = [1 if GROUP_MAP[LABELS[p]] in RELEVANT_GROUPS else 0 for p in y_pred]
    from sklearn.metrics import f1_score
    mrel = {"acc": round(float(accuracy_score(ry_true, ry_pred)), 4),
            "f1": round(float(f1_score(ry_true, ry_pred, average="binary", zero_division=0)), 4),
            "n_positive": sum(ry_true),                 # 正样本条数
            "positive_rate": round(sum(ry_true) / max(1, n), 4)}  # 正样本占比（正负不均衡时 F1 更可信）

    # ---- 4) 误差结构：错分里有几个是跨组、几个是同组
    wrong = [(t, p) for t, p in zip(y_true, y_pred) if t != p]     # 所有错分的 (真值, 预测) 对
    cross = sum(1 for t, p in wrong if GROUP_MAP[LABELS[t]] != GROUP_MAP[LABELS[p]])  # 跨业务组
    same = len(wrong) - cross                                   # 同业务组（错得更"轻"）
    err = {"n_total": n, "n_wrong": len(wrong),
           "n_cross_group": cross, "n_same_group": same,
           "cross_rate": round(cross / max(1, len(wrong)), 4),   # 跨组错分占比
           "same_rate": round(same / max(1, len(wrong)), 4)}     # 同组错分占比

    # ---- 5) 最易混的对（真实 -> 预测）
    c = Counter(wrong)                                    # 每个 (真值,预测) 组合错了多少条
    total_per_class = Counter(y_true)                    # 每个真实类一共有多少条
    wrong_per_class = Counter(t for t, _ in wrong)       # 每个真实类一共有多少条被错分
    # ★注意：这里统计的是"该类全部错分条数"，而不是"只错到某一个类的条数"。
    #   早期版本写成后者，会让 recall 被高估（一个类错到 3 个类，只算 1 个）。
    pairs = []
    for (t, p), cnt in c.most_common(args.top):          # 取错得最多的前 N 对
        pairs.append({"true": LABELS[t], "pred": LABELS[p], "n": cnt,
                      "true_class_total": total_per_class[t],      # 该类总样本数
                      "true_class_wrong": wrong_per_class[t],      # 该类错分数
                      "true_class_recall": round((total_per_class[t] - wrong_per_class[t])
                                                 / max(1, total_per_class[t]), 4),
                      "cross_group": GROUP_MAP[LABELS[t]] != GROUP_MAP[LABELS[p]]})  # 是否跨组

    # ---- 6) 汇总落盘
    out = {"source": str(args.pred), "n": n,
           "m15": m15, "m4group": m4, "relevance": mrel,
           "error_structure": err, "top_confusions": pairs}
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)          # 确保输出目录存在
    op.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 7) 终端输出（Markdown 表，可直接贴进报告）
    print("| 口径 | 样本数 | Accuracy | Macro-F1 |")
    print("|---|---|---|---|")
    print("| 15 类 | %d | %s | %s |" % (n, m15["acc"], m15["macro_f1"]))
    print("| 4 组 | %d | %s | %s |" % (n, m4["acc"], m4["macro_f1"]))
    print("| 相关性二分类 | %d | %s | F1=%s |" % (n, mrel["acc"], mrel["f1"]))
    print()
    print("| 混淆对 | 次数 | 真实类样本 | 真实类 recall | 跨业务组 |")
    print("|---|---|---|---|---|")
    for p in pairs:
        print("| %s -> %s | %d | %d | %s | %s |" % (
            p["true"], p["pred"], p["n"], p["true_class_total"],
            p["true_class_recall"], "是" if p["cross_group"] else "否"))
    print()
    print("总错分 %d：跨组 %d (%.1f%%)，同组 %d (%.1f%%)" % (
        err["n_wrong"], err["n_cross_group"], err["cross_rate"] * 100,
        err["n_same_group"], err["same_rate"] * 100))
    print("4 组各组 recall:", m4["per_group_recall"])
    print("→ 写入", op)


if __name__ == "__main__":
    main()
