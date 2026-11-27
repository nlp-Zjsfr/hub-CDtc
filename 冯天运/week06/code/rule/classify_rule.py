"""
方法一：规则 / 关键词（L0 基线）

【这个脚本在干什么】
用最笨的办法分类：标题里出现「财经」「体育」这类类别名，就判成那一类；
一个都没出现就猜最多的那一类。作为整条实验线的**下限基线**。

教学重点：
  1. 这是**最便宜的**方法：不训练、不加载模型，10 行代码。
     先跑它，是为了给后面所有方法一个「下限」——连它都没有的东西不配谈提升。
  2. 它的天花板由「命中率」决定：实测只有 6% 的标题里出现了类别名，
     所以 Macro-F1 只有 0.069（15 类随机基线是 0.067）。
     这不是代码写得差，是这类任务天生不适合规则。
  3. 规则并非一无是处：在**格式固定的公告标题**上（"XX公司：2023 年年度业绩预告"），
     一条正则就能覆盖大部分，生产里常用它做「快筛」挡掉模型调用。

用法：
  python rule/classify_rule.py            # 全量干净验证集
  python rule/classify_rule.py --n 500    # 抽样 500 条
"""

import argparse                                # 命令行参数
import json                                   # 写结果 JSON
import sys                                    # 把 code/ 加进模块搜索路径
from pathlib import Path                     # 路径处理

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ★脚本在 code/rule/ 下，common/ 在 code/ 下 → 往上退一级加进 sys.path

from common.data import load_clean_val, load_splits   # 统一的数据入口（去重 + 抽样）
from common.labels import LABELS                       # 15 类（顺序 = 标签 id）
from common.metrics import classification_report, pred_distribution, to_markdown_table


def classify(text: str, fallback: int) -> int:
    """
    最简单的规则：标题里出现哪个类别名，就判成哪一类。

    为什么按 LABELS 的顺序取第一个命中：
      这里故意不做打分/优先级——规则法就是要暴露「它有多脆弱」，
      加上优先级表就变成人工调参了，反而掩盖 L0 的真实上限。
    """
    for i, name in enumerate(LABELS):   # 按固定顺序遍历 15 个类别名
        if name in text:                # 子串包含判断（Python 的 in 就是这个意思）
            return i                    # 命中即返回该类 id
    return fallback                     # 一个类别名都没有 → 退化成「猜最多的那一类」


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="抽样条数，0=用全部干净样本")
    ap.add_argument("--out", default="outputs/rule_pred.json")   # 结果写到哪里
    args = ap.parse_args()

    samples, meta = load_clean_val(args.n or None)   # 与其它方法完全相同的评测集
    train, _ = load_splits()                         # 只用训练集算「最多类」（不碰标签答案）

    # 兜底类：训练集里样本最多的类别（这叫 majority baseline，是所有方法的下限）
    from collections import Counter
    fallback = Counter(d["label"] for d in train).most_common(1)[0][0]   # 众数类 id

    y_true = [d["label"] for d in samples]            # 真值
    y_pred = [classify(d["sentence"], fallback) for d in samples]   # 逐条规则预测

    report = classification_report(y_true, y_pred)    # 统一口径指标
    report["method"] = "rule_keyword"                 # 方法名
    report["pred_distribution"] = pred_distribution(y_pred)   # 预测都落到哪几类（几乎全在众数类）

    out = Path(__file__).resolve().parent.parent / args.out   # 相对 code/ 目录
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "report": report}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"样本 {meta['n_eval']} 条（干净集，重复已剔除 {meta['n_dup']} 条）")
    # 命中率：标题里真的出现过类别名的比例 —— 这个数字直接决定规则法的天花板
    print(f"命中率 {sum(1 for d in samples if any(n in d['sentence'] for n in LABELS)) / len(samples):.2%}")
    print(to_markdown_table({"规则/关键词": report}))
    print(f"→ 结果写入 {out}")


if __name__ == "__main__":
    main()
