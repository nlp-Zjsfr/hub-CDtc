"""
本地预训练模型定位（离线课堂必备）

【这个文件在干什么】
决定「用哪个预训练模型、从哪个本地目录加载」。
这是作业 B（换模型对照实验）的**唯一开关**：改环境变量，不改代码。

为什么不能写死路径：
  同一个项目会存在于两套目录树里（母版 `项目材料准备/` 与交付副本 `八斗课程/weekN/`），
  脚本所在层级不同，`../pretrain_models` 在其中一套里必然失效。
  做法：从脚本所在目录**向上逐级查找** `pretrain_models/<模型名>`，
  并且要求目录里真的有权重文件——否则会拿到半下载的空目录（离线课堂最常见的坑）。

用法：
  from common.model_paths import qwen3_path, bert_path
"""

import os                                  # 读环境变量
from pathlib import Path                    # 路径处理

QWEN_DIR = "Qwen3-0.6B"
# 换预训练模型：设环境变量即可，不用改代码
#   set WEEK6_BERT_DIR=chinese-roberta-wwm-ext
BERT_DIR = os.getenv("WEEK6_BERT_DIR", "bert-base-chinese")   # ★默认 bert-base-chinese
#   注意：这是「模块导入时」读一次的值，所以必须在启动进程前设置环境变量，
#   换模型对照实验.py 就是通过子进程 env 传进来的。
PRETRAIN = "pretrain_models"                # 存放预训练模型的目录名


def _has_weights(p: Path) -> bool:
    """判断一个目录里是不是真的有模型权重（防止把半下载的空目录当模型用）"""
    if not p.is_dir():
        return False                         # 目录都不存在
    if any(p.glob("*.safetensors")):        # 新格式权重
        return True
    return (p / "pytorch_model.bin").exists()   # 老格式权重


def resolve(dir_name: str, extra_roots=None) -> Path:
    """
    从当前文件向上逐级找 pretrain_models/<dir_name>。
    找不到就抛错并列出尝试过的路径——**不静默回退到联网下载**，保证断网也能用。
    """
    here = Path(__file__).resolve()           # 本文件绝对路径
    candidates = []
    for root in (extra_roots or []):         # 额外指定的根目录（优先尝试）
        root = Path(root)
        candidates.append(root if root.name == dir_name else root / dir_name)
    for parent in here.parents:              # 从 code/common/ 一路往上：code/ → 作业/ → week6/ → 主课程/ ...
        candidates.append(parent / PRETRAIN / dir_name)
    for c in candidates:
        if _has_weights(c):
            return c                         # 找到第一个「真有权重」的目录
    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"未找到本地模型 {dir_name}，尝试过：\n  {tried}")


def qwen3_path() -> Path:
    """LLM（Qwen3-0.6B）本地路径"""
    return resolve(QWEN_DIR)


def bert_path() -> Path:
    """BERT 系列本地路径（当前用哪个由环境变量 WEEK6_BERT_DIR 决定）"""
    return resolve(BERT_DIR)


if __name__ == "__main__":
    # 自检：课堂排查「模型找不到」时先跑这个
    print(f"{QWEN_DIR:<20} -> {qwen3_path()}")
    print(f"{BERT_DIR:<20} -> {bert_path()}")
