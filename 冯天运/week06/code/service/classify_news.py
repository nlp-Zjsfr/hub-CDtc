"""
分流服务：把单点模型拼成可被业务调用的接口（L5 系统组合）

【这个脚本在干什么】
把「规则快筛 → BERT 分类 → 置信度闸门 → 队列路由」四段串成一个可调用的接口，
对外只吐一个 JSON 契约。

教学重点：
  1. 真实系统从来不是「一个模型」，而是四段串起来：
       规则快筛 → 模型分类 → 置信度闸门 → 队列路由
     每一层只负责自己最擅长的那一段。
  2. 接口契约比准确率更重要：下游要的是 {topic, relevant, conf, queue}，
     有了 conf 才能做「低置信转人工」；只有 label 的话下游没法决策。
  3. 降级要显式：BERT 权重不存在时退回规则法并打标记，
     工作台不会崩，但调用方能从 model 字段看出「这次是降级结果」。

被谁调用：
  第 9 周会把这个函数包成 Agent 的第一个工具（Function Calling / MCP / CLI 三接法）。

用法：
  python service/classify_news.py --text "贵州茅台：2023 年年度业绩预告"
  python service/classify_news.py --file inputs.txt      # 每行一条
"""

import argparse                          # 命令行参数
import json                             # 输出 JSON 契约
import sys                              # 模块搜索路径
from pathlib import Path                 # 路径处理

CODE_ROOT = Path(__file__).resolve().parent.parent   # code/ 根目录
sys.path.insert(0, str(CODE_ROOT))      # 让 common.* 可导入

from common.labels import LABELS, is_relevant, queue_name, to_group   # 标签与分流规则

MODEL_NAME = "bert-cls-v1"              # 模型名（写进输出，便于追溯是哪版模型出的结果）
MODEL_VERSION = "2026-11"               # 版本号

# 规则快筛：公告标题有很强的固定套路，一条正则就能挡掉大量调用。
# 命中即直接给出结果，不再过模型 —— 这是生产里最常见的「省钱」手段。
FAST_RULES = [
    ("业绩预告", "财经"), ("年度报告", "财经"), ("季度报告", "财经"),
    ("回购", "证券"), ("减持", "证券"), ("增持", "证券"), ("涨停", "证券"),
]
# ↑ 键 = 关键词，值 = 类别名；命中就跳过后面的模型


class ShuntService:
    """分流服务：加载一次模型，之后每次调用复用"""

    def __init__(self, ckpt_path=None, conf_threshold=0.8):
        self.conf_threshold = conf_threshold          # 置信度闸门阈值（<它 → review）
        self.ckpt_path = Path(ckpt_path) if ckpt_path else CODE_ROOT / "outputs" / "bert_ckpt.pt"
        self.model, self.tok = None, None              # 模型与分词器
        self._try_load()                               # 初始化时就把模型加载好（避免每次调用重载）

    def _try_load(self):
        """加载 BERT；权重缺失时**显式降级**，而不是静默出错"""
        if not self.ckpt_path.exists():
            print(f"[warn] 未找到 {self.ckpt_path}，降级为规则法（先跑 bert/train_bert.py）", flush=True)
            return                                    # 没有权重 → 保持 model=None，走降级路径
        try:
            import torch
            from transformers import AutoTokenizer
            sys.path.insert(0, str(CODE_ROOT / "bert"))   # 让 from train_bert import ... 能找到
            from train_bert import BertClassifier           # 复用训练时的模型结构定义
            from common.model_paths import bert_path         # 定位本地预训练模型目录

            ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)  # 读权重
            self.tok = AutoTokenizer.from_pretrained(str(bert_path()))                 # 分词器
            self.model = BertClassifier(bert_path(), num_labels=len(LABELS), pool=ckpt.get("pool", "cls"))
            self.model.load_state_dict(ckpt["state_dict"])   # 灌参数
            self.model.eval()                              # 推理模式（关 dropout）
        except Exception as e:                      # 模型坏了也不能让服务整体挂掉
            print(f"[warn] BERT 加载失败（{e}），降级为规则法", flush=True)
            self.model = None                         # 显式置空 → 后续自动走降级分支

    def _rule_hit(self, text):
        """规则快筛：命中关键词直接给类别 id，没命中返回 None"""
        for kw, label in FAST_RULES:
            if kw in text:
                return LABELS.index(label)            # 类别名 → id
        return None                                   # 没命中 → 交给模型

    def _model_predict(self, text):
        """模型分类：返回 (类别 id, 置信度)"""
        import torch
        enc = self.tok(text, truncation=True, max_length=64, padding="max_length", return_tensors="pt")
        with torch.no_grad():                          # 不建计算图
            logits = self.model(enc["input_ids"], enc["attention_mask"])
        prob = torch.softmax(logits.float(), -1)[0]    # softmax → 取第 0 条的概率分布
        idx = int(prob.argmax())                      # 最大概率的类别
        return idx, float(prob[idx])                  # 置信度 = 最大概率

    def classify(self, title: str) -> dict:
        """
        返回字段说明（下游按这个契约消费）：
          topic     4 组主题，给人看的
          sub_topic 15 类细标签，给统计用的
          relevant  是否与投研相关
          conf      置信度；< threshold 时 queue=review
          queue     research / review / drop
          model     用了哪条路径（rule / bert / fallback）
        """
        label_idx = self._rule_hit(title)             # 第 1 段：规则快筛
        if label_idx is not None:
            conf, model_used = 0.99, "rule"        # 规则命中的置信度是人为设定的，不是模型输出
        elif self.model is not None:
            label_idx, conf = self._model_predict(title)   # 第 2 段：模型分类
            model_used = "bert"
        else:
            label_idx, conf, model_used = 0, 0.0, "fallback"   # 什么都没有 → 全部进 review

        return {
            "title": title,
            "topic": to_group(label_idx),                    # 第 3 段：15 类 → 4 组主题
            "sub_topic": LABELS[label_idx],                 # 细标签
            "relevant": is_relevant(label_idx),              # 是否投研相关
            "conf": round(conf, 4),                         # 置信度（下游据此决定要不要转人工）
            "queue": queue_name(label_idx, conf, self.conf_threshold),   # 第 4 段：队列路由
            "model": model_used,                            # 哪条路径出的结果（能看出是不是降级）
            "version": MODEL_VERSION,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", help="单条标题")
    ap.add_argument("--file", help="每行一条标题的文件；传 - 表示从标准输入读")
    ap.add_argument("--ckpt", default=None)               # 指定权重文件（默认 code/outputs/bert_ckpt.pt）
    ap.add_argument("--conf_threshold", type=float, default=0.8)   # 置信度闸门
    args = ap.parse_args()

    svc = ShuntService(args.ckpt, args.conf_threshold)   # 服务只加载一次模型

    def emit(title):
        print(json.dumps(svc.classify(title.strip()), ensure_ascii=False))   # 每行一个 JSON

    if args.text:
        emit(args.text)
        return
    if args.file:
        if args.file == "-":
            texts = sys.stdin.read().splitlines()        # 从标准输入读（管道友好）
        else:
            p = Path(args.file)
            if not p.is_file():
                ap.error(f"找不到输入文件：{p}（请给绝对路径，或先 cd 到 code/ 目录）")
            texts = p.read_text(encoding="utf-8").splitlines()
        for t in texts:
            if t.strip():
                emit(t)                                # 跳过空行
        return

    # 没给 --text / --file 时进入交互模式，避免 Path(None) 直接崩
    print(f"分流服务已就绪：置信度阈值={svc.conf_threshold}。输入一条标题回车看结果；q / exit 退出。")
    print("也可以直接带参数运行：python service/classify_news.py --text \"贵州茅台：2023 年年度业绩预告\"")
    while True:
        try:
            title = input("\n标题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见")
            break
        if title.lower() in ("q", "quit", "exit"):
            print("再见")
            break
        if not title:
            continue                                 # 空输入不处理
        emit(title)


if __name__ == "__main__":
    main()
