# coding: utf-8
"""
Week06 作业：换预训练模型做 TNEWS 15 类新闻标题分类（最简版）

老师的要求（下午课）：换数据集 / 换预训练模型 / 换 API 都行，最简单的做法是
    下载一个 RoBERTa，用它把课件里的 BERT 实验复现一遍，所有数字都会变。
这里只换「预训练模型」这一个变量，其余全部和课件 BERT 微调保持一致：
    · 数据：data/train.json、data/val.json（TNEWS，和课件同一份）
    · 模型结构：预训练模型 + CLS 向量 + Linear(768 → 15)
    · 训练：3 轮、batch 64、max_length 64、分层学习率（BERT 层 2e-5，分类头 ×5）、前 10% 步 warmup 再线性衰减、类别加权
    · 评测：去掉与训练集重复的验证样本后（8,901 条），算全量和课件同款 500 条抽样（seed=42）两个口径

运行（在本文件所在目录）：
    python classify_tnews.py --model models/chinese-roberta-wwm-ext --name roberta
    python classify_tnews.py --model "D:/BaiduNetdiskDownload/八斗精品班/第四周 语言模型/bert-base-chinese/bert-base-chinese" --name bert
    python classify_tnews.py --model models/chinese-roberta-wwm-ext --name smoke --max_steps 3 --max_eval 200    # 冒烟

代码结构（C# 的职责拆分方式）：
    Config           设置
    NewsRepository   读数据、去重、抽样
    NewsDataset      一条数据 → 编号
    NewsClassifier   模型：预训练模型 + CLS + 分类头
    Trainer          训练一轮 / 评测
    Metrics          准确率、Macro-F1
    Program          入口
"""

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_LABELS = 15


# ═════════════════════════════ 设置 ═════════════════════════════
@dataclass(frozen=True)
class Config:
    model_path: str
    name: str
    epochs: int = 3
    batch_size: int = 64
    max_length: int = 64
    learning_rate: float = 2e-5         # 预训练模型层的学习率
    head_lr_mult: float = 5.0           # 分类头是随机初始化的，学得快一点：学习率 ×5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    seed: int = 42
    eval_n: int = 500                   # 课件对比用的抽样条数
    max_steps: int = 0                  # >0 时每轮只跑这么多步（冒烟）
    max_eval: int = 0                   # >0 时只评这么多条（冒烟）


# ═════════════════════════════ 数据 ═════════════════════════════
class NewsRepository:
    """<summary>读 train/val，去掉验证集里和训练集句子完全相同的样本（课件口径：这 11% 会让指标虚高）。</summary>"""

    @staticmethod
    def load(file_name: str) -> List[Dict]:
        with open(os.path.join(SCRIPT_DIR, "data", file_name), encoding="utf-8") as reader:
            return json.load(reader)          # 每条：{"idx", "sentence", "label"}

    @staticmethod
    def remove_duplicates(train: List[Dict], val: List[Dict]) -> List[Dict]:
        seen = set()
        for item in train:
            seen.add(item["sentence"])
        clean = []
        for item in val:
            if item["sentence"] not in seen:
                clean.append(item)
        return clean

    @staticmethod
    def sample(items: List[Dict], count: int, seed: int) -> List[Dict]:
        """<summary>和课件 common/data.py 的 sample_subset 完全一样：random.Random(seed).sample。</summary>"""
        if count >= len(items):
            return list(items)
        return random.Random(seed).sample(items, count)


class NewsDataset(Dataset):
    """<summary>第 i 条 → (句子, 标签)；分词放在 DataLoader 的 collate 里批量做。</summary>"""

    def __init__(self, items: List[Dict]) -> None:
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> Tuple[str, int]:
        item = self._items[index]
        return item["sentence"], item["label"]


class BatchCollator:
    """<summary>把一批 (句子, 标签) 变成模型要的张量：[CLS] 句子 [SEP]，补齐到本批最长，超过 max_length 截断。</summary>"""

    def __init__(self, tokenizer, max_length: int) -> None:
        self._tokenizer = tokenizer
        self._max_length = max_length

    def __call__(self, batch: List[Tuple[str, int]]) -> Dict[str, torch.Tensor]:
        sentences = []
        labels = []
        for sentence, label in batch:
            sentences.append(sentence)
            labels.append(label)
        encoded = self._tokenizer(sentences, padding=True, truncation=True,
                                  max_length=self._max_length, return_tensors="pt")
        encoded["labels"] = torch.tensor(labels, dtype=torch.long)
        return encoded


# ═════════════════════════════ 模型 ═════════════════════════════
class NewsClassifier(nn.Module):
    """
    <summary>预训练模型 + 取 [CLS] 位置的向量 + Linear(768 → 15)。</summary>
    换模型只换 model_path：BERT 和 RoBERTa-wwm 的结构一样（12 层、768 维），代码一行都不用改。
    """

    def __init__(self, model_path: str) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, NUM_LABELS)

    def forward(self, input_ids, attention_mask, token_type_ids=None) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                              token_type_ids=token_type_ids, return_dict=True).last_hidden_state   # (B, L, 768)
        cls_vector = hidden[:, 0, :]                                             # 第 0 个位置就是 [CLS]
        return self.classifier(self.dropout(cls_vector))                         # (B, 15)


# ═════════════════════════════ 指标 ═════════════════════════════
class Metrics:
    @staticmethod
    def accuracy(true_labels: List[int], predictions: List[int]) -> float:
        correct = 0
        for t, p in zip(true_labels, predictions):
            if t == p:
                correct += 1
        return correct / len(true_labels)

    @staticmethod
    def macro_f1(true_labels: List[int], predictions: List[int]) -> float:
        """<summary>每类先算 F1 再取平均；某类没出现过（真实和预测都没有）就不参与平均。</summary>"""
        f1_sum = 0.0
        class_count = 0
        for label in range(NUM_LABELS):
            tp = fp = fn = 0
            for t, p in zip(true_labels, predictions):
                if p == label and t == label:
                    tp += 1
                elif p == label:
                    fp += 1
                elif t == label:
                    fn += 1
            if tp + fp + fn == 0:
                continue
            f1_sum += 2 * tp / (2 * tp + fp + fn)
            class_count += 1
        return f1_sum / class_count


# ═════════════════════════════ 训练 ═════════════════════════════
class Trainer:
    def __init__(self, model: NewsClassifier, config: Config, class_weights: torch.Tensor,
                 total_steps: int, device: torch.device) -> None:
        self._model = model
        self._config = config
        self._device = device
        # 类别加权：头部类上万条、尾部类几十条，不加权尾部 Recall 趋近 0
        self._criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        # 分层学习率：预训练层小步，分类头大步
        head_parameters = list(model.classifier.parameters())
        body_parameters = list(model.encoder.parameters())
        self._optimizer = torch.optim.AdamW(
            [{"params": body_parameters, "lr": config.learning_rate},
             {"params": head_parameters, "lr": config.learning_rate * config.head_lr_mult}],
            weight_decay=config.weight_decay)
        self._scheduler = get_linear_schedule_with_warmup(
            self._optimizer, num_warmup_steps=int(config.warmup_ratio * total_steps),
            num_training_steps=total_steps)
        self._use_bf16 = device.type == "cuda"

    def _forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.autocast(device_type=self._device.type, dtype=torch.bfloat16, enabled=self._use_bf16):
            logits = self._model(batch["input_ids"].to(self._device),
                                 batch["attention_mask"].to(self._device),
                                 batch["token_type_ids"].to(self._device))
        return logits.float()

    def train_epoch(self, loader: DataLoader) -> float:
        self._model.train()
        total_loss = 0.0
        steps = 0
        for batch in loader:
            if self._config.max_steps and steps >= self._config.max_steps:
                break
            logits = self._forward(batch)
            loss = self._criterion(logits, batch["labels"].to(self._device))
            self._optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._config.grad_clip)
            self._optimizer.step()
            self._scheduler.step()
            total_loss += loss.item()
            steps += 1
        return total_loss / steps

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Tuple[List[int], List[int]]:
        """<returns>(真实标签列表, 预测标签列表)</returns>"""
        self._model.eval()
        true_labels: List[int] = []
        predictions: List[int] = []
        for batch in loader:
            logits = self._forward(batch)
            predictions.extend(logits.argmax(dim=-1).tolist())
            true_labels.extend(batch["labels"].tolist())
        return true_labels, predictions


# ═════════════════════════════ 入口 ═════════════════════════════
class Program:
    @staticmethod
    def parse_args() -> Config:
        parser = argparse.ArgumentParser(description="换预训练模型做 TNEWS 分类（最简版）")
        parser.add_argument("--model", required=True, help="预训练模型目录（含 config.json、权重、vocab.txt）")
        parser.add_argument("--name", required=True, help="这次实验的名字，用于输出文件")
        parser.add_argument("--epochs", type=int, default=Config.epochs)
        parser.add_argument("--lr", type=float, default=Config.learning_rate)
        parser.add_argument("--max_steps", type=int, default=0)
        parser.add_argument("--max_eval", type=int, default=0)
        args = parser.parse_args()
        return Config(model_path=args.model, name=args.name, epochs=args.epochs,
                      learning_rate=args.lr, max_steps=args.max_steps, max_eval=args.max_eval)

    @staticmethod
    def class_weights(train: List[Dict]) -> torch.Tensor:
        """<summary>balanced：权重 = 总数 / (类别数 × 该类条数)，少的类权重大。</summary>"""
        counts = [0] * NUM_LABELS
        for item in train:
            counts[item["label"]] += 1
        weights = []
        for count in counts:
            weights.append(len(train) / (NUM_LABELS * max(count, 1)))
        return torch.tensor(weights, dtype=torch.float)

    @staticmethod
    def main() -> None:
        config = Program.parse_args()
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train = NewsRepository.load("train.json")
        val = NewsRepository.load("val.json")
        clean_val = NewsRepository.remove_duplicates(train, val)
        subset_500 = NewsRepository.sample(clean_val, config.eval_n, config.seed)
        if config.max_eval:
            clean_val = clean_val[:config.max_eval]
            subset_500 = subset_500[:config.max_eval]
        print(f"device: {device}  model: {config.model_path}")
        print(f"训练 {len(train):,} 条 / 验证 {len(val):,} 条，去重后 {len(clean_val):,} 条，抽样 {len(subset_500)} 条")

        tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        collator = BatchCollator(tokenizer, config.max_length)
        train_loader = DataLoader(NewsDataset(train), batch_size=config.batch_size, shuffle=True, collate_fn=collator)
        clean_loader = DataLoader(NewsDataset(clean_val), batch_size=128, shuffle=False, collate_fn=collator)
        subset_loader = DataLoader(NewsDataset(subset_500), batch_size=128, shuffle=False, collate_fn=collator)

        model = NewsClassifier(config.model_path).to(device)
        parameter_count = sum(p.numel() for p in model.parameters())
        print(f"参数量: {parameter_count:,}")
        steps_per_epoch = config.max_steps if config.max_steps else len(train_loader)
        trainer = Trainer(model, config, Program.class_weights(train), config.epochs * steps_per_epoch, device)

        print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Acc(去重全量)':>14}  {'Macro-F1(去重全量)':>18}  {'用时':>6}")
        print("-" * 66)
        history = []
        for epoch in range(1, config.epochs + 1):
            start = time.time()
            train_loss = trainer.train_epoch(train_loader)
            true_labels, predictions = trainer.predict(clean_loader)
            accuracy = Metrics.accuracy(true_labels, predictions)
            macro_f1 = Metrics.macro_f1(true_labels, predictions)
            history.append({"epoch": epoch, "train_loss": train_loss, "acc": accuracy, "macro_f1": macro_f1})
            print(f"{epoch:>6}  {train_loss:>10.4f}  {accuracy:>14.4f}  {macro_f1:>18.4f}  {time.time() - start:>5.0f}s")

        # 最终成绩：训练 3 轮后的模型（不挑最好的一轮，避免拿验证集选模型）
        true_labels, predictions = trainer.predict(subset_loader)
        result = {
            "name": config.name, "model": config.model_path, "params": parameter_count,
            "full_clean": {"n": len(clean_val), "acc": history[-1]["acc"], "macro_f1": history[-1]["macro_f1"]},
            "subset_500": {"n": len(subset_500),
                           "acc": Metrics.accuracy(true_labels, predictions),
                           "macro_f1": Metrics.macro_f1(true_labels, predictions)},
            "history": history,
        }
        os.makedirs(os.path.join(SCRIPT_DIR, "logs"), exist_ok=True)
        out_path = os.path.join(SCRIPT_DIR, "logs", f"{config.name}.json")
        with open(out_path, "w", encoding="utf-8") as writer:
            json.dump(result, writer, ensure_ascii=False, indent=2)
        print(f"\n课件同款 500 条抽样：Acc {result['subset_500']['acc']:.4f}  Macro-F1 {result['subset_500']['macro_f1']:.4f}")
        print("课件参考（500 条）：TF-IDF 0.472 / BERT 0.544 / Qwen3 LoRA 0.484 / API zero-shot 0.471")
        print(f"已保存 {out_path}")


if __name__ == "__main__":
    Program.main()
