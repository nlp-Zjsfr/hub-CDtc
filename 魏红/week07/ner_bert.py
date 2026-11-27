# coding: utf-8
"""
Week07 作业：换一个 BIO 数据集做抽取——人民日报 NER（BERT + 线性头）

老师的要求：选一个「不是课上 DuEE 格式」的数据集（BIO / BIOES / span 字典 / CoNLL 均可），
    写清「原始格式 → 模型输入」的转换，用 BERT 训一版，报 entity-level 的 P/R/F1，
    交完整训练日志，并回答「和 DuEE 差在哪、转换丢了什么」。

数据：data/{train,validation,test}.json，每条 {"tokens": [字...], "ner_tags": [BIO...]}，没有 text、没有 start/end。
标签：data/label_names.json 的 7 个标签（O + PER/ORG/LOC 各 B/I）。

流程（对应 README 里的 ①②③④）：
    ① 读 JSON，校验长度和标签
    ② tokenizer(is_split_into_words=True) → word_ids，非首子词 / 特殊符 = -100
    ③ BERT + Linear(768 → 7)，逐 token 交叉熵
    ④ argmax → BIO → span → 实体级 P/R/F1

运行（本文件所在目录）：
    python ner_bert.py                         # 3 轮，约 10 分钟
    python ner_bert.py --epochs 1 --max_steps 5 --max_eval 50 --name smoke      # 冒烟

代码结构（C# 的职责拆分方式，普通 for 循环，不用 LINQ 式写法）：
    Config           设置
    BioCodec         BIO ↔ span，非法序列计数，span → BIO（会记录重叠冲突）
    NerDataset       一条数据 → input_ids / labels / word_ids（含截断统计）
    NerModel         BERT + 线性头
    EntityMetrics    实体级 P/R/F1（总体 + 逐类型）
    Trainer          训练一轮 / 预测
    Program          入口
"""

import argparse
import json
import os
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
DEFAULT_BERT = r"D:\BaiduNetdiskDownload\八斗精品班\第四周 语言模型\bert-base-chinese\bert-base-chinese"

Span = Tuple[int, int, str]          # (start, end, 类型)，左闭右开


# ═════════════════════════════ 设置 ═════════════════════════════
@dataclass(frozen=True)
class Config:
    model_path: str = DEFAULT_BERT
    name: str = "bert_bio_128"
    epochs: int = 3
    batch_size: int = 32
    max_length: int = 128               # 含 [CLS]/[SEP]，所以每句最多放 126 个字
    learning_rate: float = 2e-5
    head_lr_mult: float = 5.0           # 分类头随机初始化，学习率 ×5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    log_every: int = 50
    seed: int = 42
    max_steps: int = 0                  # >0：每轮只跑这么多步（冒烟）
    max_eval: int = 0                   # >0：每个评估集只评这么多句（冒烟）


# ═════════════════════════════ BIO ↔ span ═════════════════════════════
class BioCodec:
    """<summary>BIO 没有 offset：实体的起止是从标签「数」出来的。</summary>"""

    @staticmethod
    def bio_to_spans(tags: List[str]) -> Tuple[List[Span], int]:
        """
        <returns>(spans, 非法转移次数)</returns>
        O 关掉当前实体；B-X 关掉上一个并开新实体；I-X 如果当前实体也是 X 就继续，否则按非法序列修成新实体。
        非法转移：I-X 开头、O → I-X、B-X → I-Y。要单独统计，不能修完就算。
        """
        spans: List[Span] = []
        illegal = 0
        start = -1
        kind = ""
        for i, tag in enumerate(tags):
            if tag == "O":
                if start != -1:
                    spans.append((start, i, kind))
                    start, kind = -1, ""
            elif tag.startswith("B-"):
                if start != -1:
                    spans.append((start, i, kind))
                start, kind = i, tag[2:]
            else:                                   # I-X
                this_kind = tag[2:]
                if start != -1 and this_kind == kind:
                    continue                        # 合法延续
                illegal += 1
                if start != -1:
                    spans.append((start, i, kind))
                start, kind = i, this_kind          # 修成新实体
        if start != -1:
            spans.append((start, len(tags), kind))
        return spans, illegal

    @staticmethod
    def spans_to_bio(length: int, spans: List[Span]) -> Tuple[List[str], List[Tuple[Span, Span]]]:
        """
        <summary>start/end → BIO（换数据集时最常写）。不是无损的：重叠的 span，后出现的整个被丢掉（先到先得）。</summary>
        <returns>(tags, 冲突列表 [(被覆盖的, 覆盖它的)])</returns>
        """
        tags = ["O"] * length
        owner: List[Span] = [None] * length          # 记录每个位置现在属于哪个 span
        conflicts: List[Tuple[Span, Span]] = []
        for span in spans:
            start, end, kind = span
            blocker = None
            for i in range(start, end):
                if owner[i] is not None:
                    blocker = owner[i]
                    break
            if blocker is not None:
                conflicts.append((blocker, span))      # 和已写入的 span 重叠：整个丢掉，不写出混合标签
                continue
            for i in range(start, end):
                owner[i] = span
                tags[i] = ("B-" if i == start else "I-") + kind
        return tags, conflicts


# ═════════════════════════════ 数据 ═════════════════════════════
class NerDataset(Dataset):
    """
    <summary>第 i 条 → input_ids / attention_mask / labels（非首子词、特殊符 = -100）。</summary>
    启动时先校验：len(tokens) == len(ner_tags)，每个标签都在 label_names 里（出错立刻报，不要训到一半才发现）。
    """

    def __init__(self, path: str, tokenizer, label2id: Dict[str, int], max_length: int) -> None:
        with open(path, encoding="utf-8") as reader:
            self.items: List[Dict] = json.load(reader)
        self._tokenizer = tokenizer
        self._label2id = label2id
        self._max_length = max_length
        for index, item in enumerate(self.items):
            if len(item["tokens"]) != len(item["ner_tags"]):
                raise ValueError("第 %d 条 tokens 和 ner_tags 长度不等" % index)
            for tag in item["ner_tags"]:
                if tag not in label2id:
                    raise ValueError("第 %d 条出现未知标签 %s" % (index, tag))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict:
        item = self.items[index]
        chars = item["tokens"]
        encoded = self._tokenizer(chars, is_split_into_words=True, truncation=True,
                                  max_length=self._max_length, padding="max_length", return_tensors="pt")
        word_ids = encoded.word_ids(0)               # [None, 0, 1, ..., None, None(PAD)...]
        labels: List[int] = []
        previous = None
        for word_id in word_ids:
            if word_id is None:                      # [CLS] / [SEP] / [PAD]
                labels.append(-100)
            elif word_id != previous:                # 该字的第一个子词，背标签
                labels.append(self._label2id[item["ner_tags"][word_id]])
            else:                                    # 同一个字的后续子词
                labels.append(-100)
            previous = word_id
        return {"input_ids": encoded["input_ids"][0], "attention_mask": encoded["attention_mask"][0],
                "token_type_ids": encoded["token_type_ids"][0], "labels": torch.tensor(labels),
                "index": index}

    def truncated_count(self) -> Tuple[int, int]:
        """<returns>(被截断的句数, 因截断而掉的尾部字数)。每句最多放 max_length - 2 个字。</summary>"""
        limit = self._max_length - 2
        sentences = 0
        lost_chars = 0
        for item in self.items:
            if len(item["tokens"]) > limit:
                sentences += 1
                lost_chars += len(item["tokens"]) - limit
        return sentences, lost_chars


# ═════════════════════════════ 模型 ═════════════════════════════
class NerModel(nn.Module):
    """<summary>BERT + Linear(768 → 7)：每个 token 一个向量，逐 token 7 分类。不池化。</summary>"""

    def __init__(self, model_path: str, num_labels: int) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                              token_type_ids=token_type_ids, return_dict=True).last_hidden_state   # (B, L, 768)
        return self.classifier(self.dropout(hidden))                                                # (B, L, 7)


# ═════════════════════════════ 指标 ═════════════════════════════
class EntityMetrics:
    """<summary>实体级 P/R/F1：类型 + 起止完全相同才算 TP。不是 token accuracy（全预测 O 也有 ~89%）。</summary>"""

    @staticmethod
    def compute(gold_spans: List[List[Span]], pred_spans: List[List[Span]]) -> Dict:
        tp: Dict[str, int] = {}
        gold_count: Dict[str, int] = {}
        pred_count: Dict[str, int] = {}
        for gold, pred in zip(gold_spans, pred_spans):
            for span in gold:
                gold_count[span[2]] = gold_count.get(span[2], 0) + 1
            for span in pred:
                pred_count[span[2]] = pred_count.get(span[2], 0) + 1
                if span in gold:
                    tp[span[2]] = tp.get(span[2], 0) + 1
        result = {}
        total_tp = total_gold = total_pred = 0
        for kind in sorted(set(gold_count) | set(pred_count)):
            result[kind] = EntityMetrics._prf(tp.get(kind, 0), pred_count.get(kind, 0), gold_count.get(kind, 0))
            total_tp += tp.get(kind, 0)
            total_gold += gold_count.get(kind, 0)
            total_pred += pred_count.get(kind, 0)
        result["ALL"] = EntityMetrics._prf(total_tp, total_pred, total_gold)
        return result

    @staticmethod
    def _prf(tp: int, pred: int, gold: int) -> Dict:
        precision = tp / pred if pred else 0.0
        recall = tp / gold if gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"precision": precision, "recall": recall, "f1": f1, "support": gold}

    @staticmethod
    def error_kinds(gold: List[Span], pred: List[Span]) -> Dict[str, int]:
        """<summary>错误分类：miss 金标有预测没有；spurious 预测有金标没有；type 边界对类型错；boundary 类型对边界错。</summary>"""
        counts = {"miss": 0, "spurious": 0, "type": 0, "boundary": 0}
        for g in gold:
            if g in pred:
                continue
            same_box = False
            overlap_same_type = False
            for p in pred:
                if p[0] == g[0] and p[1] == g[1]:
                    same_box = True
                elif p[2] == g[2] and p[0] < g[1] and g[0] < p[1]:
                    overlap_same_type = True
            if same_box:
                counts["type"] += 1
            elif overlap_same_type:
                counts["boundary"] += 1
            else:
                counts["miss"] += 1
        for p in pred:
            if p in gold:
                continue
            touches_gold = False
            for g in gold:
                if (g[0] == p[0] and g[1] == p[1]) or (g[2] == p[2] and g[0] < p[1] and p[0] < g[1]):
                    touches_gold = True
            if not touches_gold:
                counts["spurious"] += 1
        return counts


# ═════════════════════════════ 训练 ═════════════════════════════
class Trainer:
    def __init__(self, model: NerModel, config: Config, total_steps: int, device: torch.device) -> None:
        self._model = model
        self._config = config
        self._device = device
        self._criterion = nn.CrossEntropyLoss(ignore_index=-100)      # -100 的位置不参与
        self._optimizer = torch.optim.AdamW(
            [{"params": list(model.encoder.parameters()), "lr": config.learning_rate},
             {"params": list(model.classifier.parameters()), "lr": config.learning_rate * config.head_lr_mult}],
            weight_decay=config.weight_decay)
        self._scheduler = get_linear_schedule_with_warmup(
            self._optimizer, int(config.warmup_ratio * total_steps), total_steps)
        self._use_bf16 = device.type == "cuda"

    def _forward(self, batch: Dict) -> torch.Tensor:
        with torch.autocast(device_type=self._device.type, dtype=torch.bfloat16, enabled=self._use_bf16):
            logits = self._model(batch["input_ids"].to(self._device), batch["attention_mask"].to(self._device),
                                 batch["token_type_ids"].to(self._device))
        return logits.float()

    def train_epoch(self, loader: DataLoader, epoch: int, step_log, global_step: int, start_time: float) -> Tuple[float, int]:
        self._model.train()
        total_loss = 0.0
        steps = 0
        for batch in loader:
            if self._config.max_steps and steps >= self._config.max_steps:
                break
            logits = self._forward(batch)
            labels = batch["labels"].to(self._device)
            loss = self._criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            self._optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._config.grad_clip)
            self._optimizer.step()
            self._scheduler.step()
            total_loss += loss.item()
            steps += 1
            global_step += 1
            if global_step % self._config.log_every == 0:
                step_log.write(json.dumps({"epoch": epoch, "step": global_step, "loss": round(loss.item(), 5),
                                           "lr": round(self._scheduler.get_last_lr()[0], 10),
                                           "elapsed_s": round(time.time() - start_time, 1)}) + "\n")
                step_log.flush()
        return total_loss / steps, global_step

    @torch.no_grad()
    def predict(self, dataset: NerDataset, id2label: List[str], batch_size: int = 64, limit: int = 0) -> Dict:
        """<returns>每句：gold_spans、pred_spans、pred_tags、是否截断、非法转移数、val loss 用的总 loss 和 token 数</returns>"""
        self._model.eval()
        items = dataset.items[:limit] if limit else dataset.items
        sub = torch.utils.data.Subset(dataset, list(range(len(items))))
        loader = DataLoader(sub, batch_size=batch_size, shuffle=False)
        rows = []
        loss_sum = 0.0
        token_count = 0
        max_chars = dataset._max_length - 2
        for batch in loader:
            logits = self._forward(batch)
            labels = batch["labels"].to(self._device)
            loss_sum += nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                                                    ignore_index=-100, reduction="sum").item()
            token_count += int((labels != -100).sum())
            predictions = logits.argmax(dim=-1).cpu()
            for b in range(predictions.size(0)):
                item = items[int(batch["index"][b])]
                # 子词标签 → 字符标签：只取每个字的第一个子词（labels != -100 的位置）
                chars = min(len(item["tokens"]), max_chars)
                char_tags: List[str] = []
                for position in range(predictions.size(1)):
                    if batch["labels"][b, position] != -100:
                        char_tags.append(id2label[int(predictions[b, position])])
                char_tags = char_tags[:chars]
                pred_spans, illegal = BioCodec.bio_to_spans(char_tags)
                # 金标只取模型看得到的前 chars 个字，截断掉的尾部实体另算（见 README 的截断说明）
                full_gold, _ = BioCodec.bio_to_spans(item["ner_tags"])
                rows.append({"text": "".join(item["tokens"]), "truncated": len(item["tokens"]) > max_chars,
                             "gold_spans": full_gold, "pred_spans": pred_spans, "illegal": illegal,
                             "kept_chars": chars})
        return {"rows": rows, "val_loss": loss_sum / max(token_count, 1)}


# ═════════════════════════════ 入口 ═════════════════════════════
class Program:
    @staticmethod
    def parse_args() -> Config:
        parser = argparse.ArgumentParser(description="人民日报 NER：BERT + 线性头")
        parser.add_argument("--model", default=DEFAULT_BERT)
        parser.add_argument("--name", default=Config.name)
        parser.add_argument("--epochs", type=int, default=Config.epochs)
        parser.add_argument("--max_length", type=int, default=Config.max_length)
        parser.add_argument("--max_steps", type=int, default=0)
        parser.add_argument("--max_eval", type=int, default=0)
        args = parser.parse_args()
        return Config(model_path=args.model, name=args.name, epochs=args.epochs, max_length=args.max_length,
                      max_steps=args.max_steps, max_eval=args.max_eval)

    @staticmethod
    def evaluate(rows: List[Dict]) -> Dict:
        """<summary>只在模型看得到的范围内评估：金标里落在截断区外的实体算作「掉尾」，单独报告。</summary>"""
        gold_all: List[List[Span]] = []
        pred_all: List[List[Span]] = []
        lost = 0
        crossing = 0
        illegal_sentences = 0
        illegal_transitions = 0
        errors = {"miss": 0, "spurious": 0, "type": 0, "boundary": 0}
        exact = 0
        exact_total = 0
        for row in rows:
            kept = row["kept_chars"]
            gold_kept: List[Span] = []
            for g in row["gold_spans"]:
                if g[1] <= kept:
                    gold_kept.append(g)
                elif g[0] >= kept:
                    lost += 1                       # 完全掉在截断区外
                else:
                    crossing += 1                   # 跨截断边界，既不算金标也不算漏检
            gold_all.append(gold_kept)
            pred_all.append(row["pred_spans"])
            if row["illegal"] > 0:
                illegal_sentences += 1
                illegal_transitions += row["illegal"]
            if not row["truncated"]:
                exact_total += 1
                if gold_kept == row["pred_spans"]:
                    exact += 1
                for key, value in EntityMetrics.error_kinds(gold_kept, row["pred_spans"]).items():
                    errors[key] += value
        metrics = EntityMetrics.compute(gold_all, pred_all)
        return {"metrics": metrics, "lost_tail_entities": lost, "crossing_entities": crossing,
                "truncated_sentences": sum(1 for r in rows if r["truncated"]),
                "illegal_sentences": illegal_sentences, "illegal_transitions": illegal_transitions,
                "sentence_exact_match": [exact, exact_total], "errors": errors}

    @staticmethod
    def main() -> None:
        config = Program.parse_args()
        random_seed = config.seed
        torch.manual_seed(random_seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logs = os.path.join(SCRIPT_DIR, "logs")
        os.makedirs(logs, exist_ok=True)

        with open(os.path.join(SCRIPT_DIR, "data", "label_names.json"), encoding="utf-8") as reader:
            id2label: List[str] = json.load(reader)          # 训练和评估必须共用这一份顺序
        label2id = {label: i for i, label in enumerate(id2label)}
        tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        data = os.path.join(SCRIPT_DIR, "data")
        train_set = NerDataset(os.path.join(data, "train.json"), tokenizer, label2id, config.max_length)
        val_set = NerDataset(os.path.join(data, "validation.json"), tokenizer, label2id, config.max_length)
        test_set = NerDataset(os.path.join(data, "test.json"), tokenizer, label2id, config.max_length)
        print(f"device: {device}  训练 {len(train_set):,} / 验证 {len(val_set):,} / 测试 {len(test_set):,}  标签 {id2label}")
        for name, dataset in (("train", train_set), ("validation", val_set), ("test", test_set)):
            sentences, lost = dataset.truncated_count()
            print(f"  {name}: 截断 {sentences} 句（{sentences / len(dataset):.2%}），掉尾 {lost} 个字")

        train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
        model = NerModel(config.model_path, len(id2label)).to(device)
        parameters = sum(p.numel() for p in model.parameters())
        steps_per_epoch = config.max_steps if config.max_steps else len(train_loader)
        trainer = Trainer(model, config, config.epochs * steps_per_epoch, device)
        print(f"参数量: {parameters:,}  每轮 {steps_per_epoch} 步")

        start = time.time()
        summary = {"config": config.__dict__, "device": str(device), "params": parameters, "epochs": []}
        print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>9}  {'Val P':>7}  {'Val R':>7}  {'Val F1':>7}  {'非法句':>6}  {'用时':>6}")
        print("-" * 72)
        global_step = 0
        with open(os.path.join(logs, config.name + "_steps.jsonl"), "w", encoding="utf-8") as step_log:
            for epoch in range(1, config.epochs + 1):
                epoch_start = time.time()
                train_loss, global_step = trainer.train_epoch(train_loader, epoch, step_log, global_step, start)
                output = trainer.predict(val_set, id2label, limit=config.max_eval)
                report = Program.evaluate(output["rows"])["metrics"]["ALL"]
                illegal = Program.evaluate(output["rows"])["illegal_sentences"]
                summary["epochs"].append({"epoch": epoch, "train_loss": train_loss, "val_loss": output["val_loss"],
                                          "val": report, "illegal_sentences": illegal,
                                          "elapsed_s": round(time.time() - epoch_start, 1)})
                print(f"{epoch:>6}  {train_loss:>10.4f}  {output['val_loss']:>9.4f}  {report['precision']:>7.4f}"
                      f"  {report['recall']:>7.4f}  {report['f1']:>7.4f}  {illegal:>6}  {time.time() - epoch_start:>5.0f}s")
        summary["total_s"] = round(time.time() - start, 1)
        summary["peak_mem_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20) if device.type == "cuda" else 0

        # 最终成绩：训练 3 轮后的模型，同时报 validation 和 test（不用 test 选模型）
        final = {}
        for name, dataset in (("validation", val_set), ("test", test_set)):
            output = trainer.predict(dataset, id2label, limit=config.max_eval)
            final[name] = Program.evaluate(output["rows"])
            with open(os.path.join(logs, f"{config.name}_{name}_predictions.jsonl"), "w", encoding="utf-8") as writer:
                for row in output["rows"]:
                    writer.write(json.dumps(row, ensure_ascii=False) + "\n")
            all_scores = final[name]["metrics"]["ALL"]
            print(f"\n[{name}] P {all_scores['precision']:.4f}  R {all_scores['recall']:.4f}  F1 {all_scores['f1']:.4f}"
                  f"  | 实体 {all_scores['support']}  掉尾 {final[name]['lost_tail_entities']}"
                  f"  非法句 {final[name]['illegal_sentences']}  错误 {final[name]['errors']}")
            for kind in ("PER", "ORG", "LOC"):
                if kind in final[name]["metrics"]:
                    s = final[name]["metrics"][kind]
                    print(f"    {kind}: P {s['precision']:.4f} R {s['recall']:.4f} F1 {s['f1']:.4f} (n={s['support']})")
        summary["final"] = final
        with open(os.path.join(logs, config.name + "_train.json"), "w", encoding="utf-8") as writer:
            json.dump(summary, writer, ensure_ascii=False, indent=2)
        print(f"\n总耗时 {summary['total_s']}s  峰值显存 {summary['peak_mem_mb']} MB  日志在 {logs}")


if __name__ == "__main__":
    Program.main()
