"""
数据读取与「干净子集」构建（全项目共用）

【这个文件在干什么】
本项目所有方法（规则 / TF-IDF / BERT / LLM）读的都是同一份数据、同一份评测子集，
这是「横向对比」成立的前提。

教学重点：
  1. 这份 TNEWS 切分里，验证集有 1099/10000 条与训练集**完全重复**（11%）。
     不做处理的话，靠词面匹配的 TF-IDF 会「背答案」，指标被显著虚高
     （实测：全量 val 上 F1 0.541，去重后只有 0.509）。
  2. 所以本项目所有方法统一在 clean 子集上评测——**同一批样本**是横向比较的前提。
  3. 重复样本不是这份数据独有的问题：真实业务里同一条公告被多家媒体转载、
     同一个问句被不同用户重复提交，都会造成「训练/验证泄漏」。

用法：
  from common.data import load_splits, clean_val, sample_subset
"""

import json                                    # 读 json 数据文件
import random                                 # 固定 seed 的随机抽样
from pathlib import Path                       # 路径处理

CODE_ROOT = Path(__file__).resolve().parent.parent   # code/ 根目录（本文件在 code/common/ 下）
DATA_DIR = CODE_ROOT / "data"                  # 数据目录：train.json / val.json


def _load(name: str):
    """按文件名读一个 json 文件（统一编码为 utf-8，避免 Windows 下 GBK 报错）"""
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def load_splits():
    """返回 (train, val)，每条是 {"idx", "sentence", "label"}"""
    return _load("train.json"), _load("val.json")     # 训练集 / 验证集（原始，未去重）


def clean_val(train=None, val=None):
    """
    去掉与训练集句子完全重复的验证样本，返回 (clean_list, n_dup)。

    为什么按「整句完全相同」判定：
      这里要暴露的是最朴素的泄漏形态——同一条数据被切到了两边。
      更隐蔽的近似重复（改标题、换标点）需要 SimHash / 向量去重，属于第 8 周的内容。
    """
    if train is None or val is None:      # 允许外部传入（避免重复读盘），否则自己读
        train, val = load_splits()
    seen = {d["sentence"] for d in train} # 训练集句子集合（set 查询 O(1)）
    clean = [d for d in val if d["sentence"] not in seen]   # 丢掉训练里出现过的
    return clean, len(val) - len(clean)  # 第二个返回值 = 被丢掉的条数（写进 meta 做审计）


def sample_subset(items, n, seed=42):
    """
    固定 seed 抽样：四次实验（规则/TF-IDF/BERT/LLM）必须用同一批样本，
    否则「差 2 个点」可能只是抽样波动。n 超过总数时直接返回全部。
    """
    if n >= len(items):
        return list(items)                 # 要的条数 ≥ 总量 → 全给（不做无意义抽样）
    rng = random.Random(seed)              # ★固定随机种子：同 seed 永远抽到同一批
    return rng.sample(items, n)            # 不放回抽样


def load_clean_val(n=None, seed=42):
    """最常用入口：读数据 → 去重 →（可选）抽样，返回 (samples, meta)"""
    train, val = load_splits()             # 原始切分
    clean, n_dup = clean_val(train, val)   # 去掉与训练集重复的验证样本
    samples = sample_subset(clean, n if n else len(clean), seed)  # n=None → 全量 clean
    meta = {                               # 评测元信息：写进每个结果 JSON，便于事后核对口径
        "n_train": len(train),             # 训练集条数
        "n_val": len(val),                 # 原始验证集条数
        "n_dup": n_dup,                    # 被判定为重复（泄漏）而剔除的条数
        "dup_rate": round(n_dup / len(val), 4) if val else 0.0,   # 泄漏比例（0.1099）
        "n_clean": len(clean),             # 去重后可用条数（8901）
        "n_eval": len(samples),            # 本次实际评测条数
        "seed": seed,                      # 抽样种子（写清楚便于复现）
    }
    return samples, meta
