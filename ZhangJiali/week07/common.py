# -*- coding: utf-8 -*-
"""事件抽取作业的共享模块：数据读取、联合 BIO 标签、序列标注数据集与统一评测。

评测约定：
    预测与金标均按 (id, sent_id) 对齐。DuEE-Fin 原始字段 id 是文档编号，
    同一公告切出的多句共用该字段，仅按 id 对齐会导致样本互相覆盖。
    解析失败或未抽出事件时，预测行仍须保留，events 置为空列表，不得从分母中删除。
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
PRED_DIR = OUTPUT_DIR / "preds"
METRIC_DIR = OUTPUT_DIR / "metrics"
LOG_DIR = OUTPUT_DIR / "logs"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
PRETRAINED_DIR = ROOT / "pretrained_model"

TRIGGER_PREFIX = "TRG:"
ARG_PREFIX = "ARG:"
IGNORE_INDEX = -100
NEG_INF = -1e4


def set_seed(seed: int = 42) -> None:
    """固定 Python、NumPy 与 PyTorch 的随机状态。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs() -> None:
    for path in (PRED_DIR, METRIC_DIR, LOG_DIR, CKPT_DIR, OUTPUT_DIR / "figures"):
        path.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path | str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows: list[dict], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_schema(path: Path | None = None) -> dict:
    """读取事件类型到角色列表的映射，忽略以双下划线开头的元数据键。"""
    path = path or (DATA_DIR / "event_schema.json")
    with open(path, "r", encoding="utf-8") as file:
        raw = json.load(file)
    return {key: list(value) for key, value in raw.items() if not key.startswith("__")}


def load_enums(path: Path | None = None) -> dict:
    path = path or (DATA_DIR / "event_schema.json")
    with open(path, "r", encoding="utf-8") as file:
        raw = json.load(file)
    return raw.get("__enums__", {}) or {}


def load_extract_split(split: str, data_dir: Path | None = None) -> list[dict]:
    directory = Path(data_dir) if data_dir else DATA_DIR
    return load_jsonl(directory / f"{split}_extract.jsonl")


def row_event(row: dict) -> dict | None:
    if row.get("event"):
        return row["event"]
    events = row.get("events") or []
    return events[0] if events else None


def make_pred_row(row: dict, events: list[dict]) -> dict:
    return {
        "id": row.get("id", ""),
        "sent_id": row.get("sent_id", ""),
        "events": events,
    }


def inner_split(rows: list[dict], ratio: float = 0.1, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """从训练集切出内层验证集，用于选择 checkpoint；正式评测集不参与选点。"""
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    n_inner = max(1, int(len(rows) * ratio))
    inner = set(indices[:n_inner])
    fit = [row for i, row in enumerate(rows) if i not in inner]
    val = [row for i, row in enumerate(rows) if i in inner]
    return fit, val


def union_roles(schema: dict) -> list[str]:
    return sorted({role for roles in schema.values() for role in roles})


def build_joint_schema(schema: dict) -> dict:
    """构造联合 BIO 标签表：触发词按事件类型区分，论元角色跨类型共享。"""
    entity_tags = [f"{TRIGGER_PREFIX}{event_type}" for event_type in sorted(schema)]
    entity_tags += [f"{ARG_PREFIX}{role}" for role in union_roles(schema)]
    labels = ["O"]
    for tag in entity_tags:
        labels.append(f"B-{tag}")
        labels.append(f"I-{tag}")
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}
    return {
        "labels": labels,
        "label2id": label2id,
        "id2label": id2label,
        "entity_tags": entity_tags,
    }


def classify_entity_tag(tag: str) -> tuple[str, str]:
    if tag.startswith(TRIGGER_PREFIX):
        return "trigger", tag[len(TRIGGER_PREFIX):]
    if tag.startswith(ARG_PREFIX):
        return "arg", tag[len(ARG_PREFIX):]
    return "arg", tag


def span_to_char_tags(text: str, event: dict, label_scheme: dict) -> list[int]:
    """将触发词与论元的字符区间转换为逐字 BIO 标签编号。重叠时触发词优先。"""
    n = len(text)
    tags = ["O"] * n
    label2id = label_scheme["label2id"]

    def paint(start: int, end: int, entity_tag: str) -> None:
        start = max(0, int(start))
        end = min(n, int(end))
        if start >= end or f"B-{entity_tag}" not in label2id:
            return
        tags[start] = f"B-{entity_tag}"
        for index in range(start + 1, end):
            tags[index] = f"I-{entity_tag}"

    for arg in event.get("args") or []:
        entity_tag = f"{ARG_PREFIX}{arg.get('role', '')}"
        paint(arg.get("start"), arg.get("end"), entity_tag)
    trigger = event.get("trigger") or {}
    if trigger:
        entity_tag = f"{TRIGGER_PREFIX}{event.get('type', '')}"
        paint(trigger.get("start"), trigger.get("end"), entity_tag)
    return [label2id[tag] for tag in tags]


def decode_bio_spans(tags: list[str]) -> list[dict]:
    """将 BIO 序列还原为半开区间 span。非法 I- 开头按新实体处理，避免直接丢弃。"""
    spans: list[dict] = []
    current = None
    for index, tag in enumerate(tags):
        if tag == "O":
            if current is not None:
                spans.append(current)
                current = None
            continue
        if tag.startswith("B-"):
            if current is not None:
                spans.append(current)
            current = {"tag": tag[2:], "start": index, "end": index + 1}
            continue
        entity = tag[2:]
        if current is not None and current["tag"] == entity:
            current["end"] = index + 1
        else:
            if current is not None:
                spans.append(current)
            current = {"tag": entity, "start": index, "end": index + 1}
    if current is not None:
        spans.append(current)
    return spans


def spans_to_event(spans: list[dict], text: str, enums: dict | None = None) -> dict | None:
    """将 span 列表组装为评测用事件。无触发词或无法判定类型时返回 None。"""
    trigger_text, trigger_span, event_type, args = "", None, "", []
    for span in spans:
        kind, payload = classify_entity_tag(span["tag"])
        piece = text[span["start"]:span["end"]]
        if not piece:
            continue
        if kind == "trigger":
            if trigger_span is None:
                trigger_span = span
                trigger_text = piece
                if payload:
                    event_type = payload
        else:
            args.append({
                "role": payload,
                "text": piece,
                "start": span["start"],
                "end": span["end"],
            })
    if not trigger_text or not event_type:
        return None
    if enums:
        for arg in args:
            allowed = enums.get(f"{event_type}.{arg['role']}")
            if allowed:
                arg["enum_ok"] = arg["text"] in allowed
    seen, unique = set(), []
    for arg in args:
        key = (arg["role"], arg["text"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(arg)
    return {
        "type": event_type,
        "trigger": {
            "text": trigger_text,
            "start": trigger_span["start"],
            "end": trigger_span["end"],
        },
        "args": unique,
    }


class BioDataset(Dataset):
    """BERT 序列标注数据：按字切分后做子词对齐，仅首子词携带标签。"""

    def __init__(self, rows: list[dict], tokenizer, label_scheme: dict, max_length: int = 256):
        self.rows = rows
        self.tokenizer = tokenizer
        self.label_scheme = label_scheme
        self.max_length = max_length
        self.n_truncated = sum(1 for row in rows if len(row["text"]) > max_length - 2)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        text = row["text"]
        chars = list(text)
        encoded = self.tokenizer(
            chars,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        word_ids = encoded.word_ids()
        event = row_event(row)
        if event is None:
            char_labels = [self.label_scheme["label2id"]["O"]] * len(text)
        else:
            char_labels = span_to_char_tags(text, event, self.label_scheme)

        aligned = []
        prev_wid = None
        for wid in word_ids:
            if wid is None:
                aligned.append(IGNORE_INDEX)
            elif wid != prev_wid:
                aligned.append(char_labels[wid] if wid < len(char_labels) else IGNORE_INDEX)
                prev_wid = wid
            else:
                aligned.append(IGNORE_INDEX)

        item = {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "word_ids": torch.tensor([-1 if wid is None else wid for wid in word_ids], dtype=torch.long),
            "labels": torch.tensor(aligned, dtype=torch.long),
            "row_idx": torch.tensor(idx, dtype=torch.long),
        }
        if encoded.get("token_type_ids") is not None:
            item["token_type_ids"] = torch.tensor(encoded["token_type_ids"], dtype=torch.long)
        return item


def collate_bio(batch: list[dict]) -> dict:
    """按当前 batch 最长序列动态补齐，减少短句上的无效计算。"""
    max_len = max(int(item["input_ids"].numel()) for item in batch)
    size = len(batch)
    out = {
        "input_ids": torch.zeros(size, max_len, dtype=torch.long),
        "attention_mask": torch.zeros(size, max_len, dtype=torch.long),
        "token_type_ids": torch.zeros(size, max_len, dtype=torch.long),
        "labels": torch.full((size, max_len), IGNORE_INDEX, dtype=torch.long),
        "word_ids": torch.full((size, max_len), -1, dtype=torch.long),
        "row_idx": torch.stack([item["row_idx"] for item in batch]),
    }
    for i, item in enumerate(batch):
        n = int(item["input_ids"].numel())
        out["input_ids"][i, :n] = item["input_ids"]
        out["attention_mask"][i, :n] = item["attention_mask"]
        out["labels"][i, :n] = item["labels"]
        out["word_ids"][i, :n] = item["word_ids"]
        if "token_type_ids" in item:
            out["token_type_ids"][i, :n] = item["token_type_ids"]
    return out


class CharDataset(Dataset):
    """字符级序列标注数据，供 BiLSTM-CRF 使用。"""

    def __init__(self, rows: list[dict], vocab: dict, label_scheme: dict, max_length: int = 256):
        self.rows = rows
        self.vocab = vocab
        self.label_scheme = label_scheme
        self.max_length = max_length
        self.unk_id = vocab.get("<UNK>", 1)
        self.n_truncated = sum(1 for row in rows if len(row["text"]) > max_length)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        text = row["text"][:self.max_length]
        event = row_event(row)
        if event is None:
            labels = [self.label_scheme["label2id"]["O"]] * len(text)
        else:
            labels = span_to_char_tags(text, event, self.label_scheme)
        ids = [self.vocab.get(char, self.unk_id) for char in text]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "row_idx": torch.tensor(idx, dtype=torch.long),
        }


def make_vocab(rows: list[dict], min_freq: int = 1) -> dict:
    """仅使用训练集构建字表，避免验证集字符泄漏。"""
    counts = Counter()
    for row in rows:
        counts.update(row["text"])
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for char, freq in counts.most_common():
        if freq >= min_freq:
            vocab[char] = len(vocab)
    return vocab


def collate_char(batch: list[dict]) -> dict:
    max_len = max(int(item["input_ids"].numel()) for item in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
    mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    row_idx = torch.zeros(len(batch), dtype=torch.long)
    for i, item in enumerate(batch):
        n = int(item["input_ids"].numel())
        input_ids[i, :n] = item["input_ids"]
        labels[i, :n] = item["labels"]
        mask[i, :n] = 1
        row_idx[i] = item["row_idx"]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": mask,
        "row_idx": row_idx,
    }


class CRF(nn.Module):
    """线性链 CRF：前向算法计算配分函数，维特比算法解码最优标签路径。"""

    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def _gold_score(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        emit = emissions.gather(2, tags.unsqueeze(-1)).squeeze(-1)
        score = self.start_transitions[tags[:, 0]] + emit[:, 0]
        length = emissions.size(1)
        for i in range(1, length):
            step = self.transitions[tags[:, i - 1], tags[:, i]] + emit[:, i]
            score = score + step * mask[:, i].to(score.dtype)
        seq_ends = mask.long().sum(dim=1) - 1
        last_tags = tags.gather(1, seq_ends.unsqueeze(1)).squeeze(1)
        return score + self.end_transitions[last_tags]

    def _forward_algorithm(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        alpha = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        length = emissions.size(1)
        for i in range(1, length):
            scores = alpha.unsqueeze(2) + self.transitions.unsqueeze(0) + emissions[:, i].unsqueeze(1)
            new_alpha = torch.logsumexp(scores, dim=1)
            keep = mask[:, i].unsqueeze(1).bool()
            alpha = torch.where(keep, new_alpha, alpha)
        return torch.logsumexp(alpha + self.end_transitions.unsqueeze(0), dim=1)

    def neg_log_likelihood(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.bool()
        log_z = self._forward_algorithm(emissions, mask)
        gold = self._gold_score(emissions, tags, mask)
        return (log_z - gold).mean()

    @torch.no_grad()
    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        mask = mask.bool()
        batch, length, _ = emissions.shape
        alpha = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        backpointers = torch.zeros(batch, length, self.num_tags, dtype=torch.long, device=emissions.device)
        for i in range(1, length):
            scores = alpha.unsqueeze(2) + self.transitions.unsqueeze(0)
            best_score, best_prev = scores.max(dim=1)
            new_alpha = best_score + emissions[:, i]
            keep = mask[:, i].unsqueeze(1).bool()
            alpha = torch.where(keep, new_alpha, alpha)
            backpointers[:, i] = best_prev
        best_last = (alpha + self.end_transitions.unsqueeze(0)).argmax(dim=1)
        lengths = mask.long().sum(dim=1)
        paths = []
        for b in range(batch):
            seq_len = int(lengths[b].item())
            tag = int(best_last[b].item())
            path = [tag]
            for i in range(seq_len - 1, 0, -1):
                tag = int(backpointers[b, i, tag].item())
                path.append(tag)
            paths.append(path[::-1])
        return paths


class SequenceTagger(nn.Module):
    """编码器加线性分类头，可选 CRF 解码。"""

    def __init__(self, encoder: nn.Module, hidden_size: int, num_tags: int,
                 dropout: float = 0.1, use_crf: bool = True):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_tags)
        self.num_tags = num_tags
        self.use_crf = use_crf
        self.crf = CRF(num_tags) if use_crf else None

    def emissions(self, input_ids, attention_mask, token_type_ids=None):
        hidden = self.encoder(input_ids, attention_mask, token_type_ids)
        return self.classifier(self.dropout(hidden))

    def forward(self, input_ids, attention_mask, labels=None, token_type_ids=None):
        emit = self.emissions(input_ids, attention_mask, token_type_ids)
        loss = None
        if labels is not None:
            if self.use_crf:
                tags = labels.clone()
                tags[tags == IGNORE_INDEX] = 0
                loss = self.crf.neg_log_likelihood(emit, tags, attention_mask)
            else:
                loss = nn.functional.cross_entropy(
                    emit.view(-1, self.num_tags), labels.view(-1), ignore_index=IGNORE_INDEX)
        return emit, loss

    @torch.no_grad()
    def predict(self, input_ids, attention_mask, token_type_ids=None) -> list[list[int]]:
        emit = self.emissions(input_ids, attention_mask, token_type_ids)
        if self.use_crf:
            return self.crf.decode(emit, attention_mask)
        pred = emit.argmax(dim=-1)
        paths = []
        for b in range(pred.shape[0]):
            n = int(attention_mask[b].sum().item())
            paths.append(pred[b, :n].tolist())
        return paths


def row_key(row: dict) -> str:
    if row.get("uid"):
        return str(row["uid"])
    if row.get("sent_id"):
        return f'{row.get("id", "")}|{row["sent_id"]}'
    return str(row.get("id", ""))


def as_event_list(row: dict) -> list[dict]:
    if "events" in row and row["events"] is not None:
        events = row["events"]
    elif row.get("event"):
        events = [row["event"]]
    else:
        events = []
    out = []
    for event in events:
        if not isinstance(event, dict):
            continue
        trigger = event.get("trigger") or {}
        trigger_text = (trigger.get("text") if isinstance(trigger, dict) else str(trigger)) or ""
        args = []
        for arg in event.get("args") or event.get("arguments") or []:
            if not isinstance(arg, dict):
                continue
            text = arg.get("text", arg.get("argument", "")) or ""
            role = arg.get("role", "") or ""
            if text:
                args.append((str(role).strip(), str(text).strip()))
        out.append({
            "type": str(event.get("type", event.get("event_type", ""))).strip(),
            "trigger": str(trigger_text).strip(),
            "args": args,
        })
    return out


def prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def compare_events(gold_events: list[dict], pred_events: list[dict]):
    """同一类型下按出现顺序一对一匹配触发词、论元多重集合与整事件。"""
    gold_by_type, pred_by_type = defaultdict(list), defaultdict(list)
    for event in gold_events:
        gold_by_type[event["type"]].append(event)
    for event in pred_events:
        pred_by_type[event["type"]].append(event)

    tr = [0, 0, 0]
    ar = [0, 0, 0]
    ev = [0, 0, 0]
    for event_type in set(gold_by_type) | set(pred_by_type):
        golds = gold_by_type.get(event_type, [])
        preds = pred_by_type.get(event_type, [])
        n = max(len(golds), len(preds))
        for i in range(n):
            gold = golds[i] if i < len(golds) else None
            pred = preds[i] if i < len(preds) else None
            if gold is None:
                ev[1] += 1
                tr[1] += 1 if pred["trigger"] else 0
                ar[1] += len(pred["args"])
                continue
            if pred is None:
                ev[2] += 1
                tr[2] += 1 if gold["trigger"] else 0
                ar[2] += len(gold["args"])
                continue
            if pred["trigger"] and pred["trigger"] == gold["trigger"]:
                tr[0] += 1
            else:
                tr[1] += 1 if pred["trigger"] else 0
                tr[2] += 1 if gold["trigger"] else 0
            gold_counter, pred_counter = Counter(gold["args"]), Counter(pred["args"])
            inter = sum((gold_counter & pred_counter).values())
            ar[0] += inter
            ar[1] += sum(pred_counter.values()) - inter
            ar[2] += sum(gold_counter.values()) - inter
            if pred["trigger"] == gold["trigger"] and gold_counter == pred_counter:
                ev[0] += 1
            else:
                ev[1] += 1
                ev[2] += 1
    return tuple(tr), tuple(ar), tuple(ev)


def type_macro_f1(gold_rows: list[dict], pred_rows: list[dict]) -> float:
    gold_map, pred_map = {}, {}
    for row in gold_rows:
        gold_map[row_key(row)] = Counter(event["type"] for event in as_event_list(row))
    for row in pred_rows:
        pred_map[row_key(row)] = Counter(event["type"] for event in as_event_list(row))
    types = set(t for counter in gold_map.values() for t in counter)
    types |= set(t for counter in pred_map.values() for t in counter)
    if not types:
        return 0.0
    scores = []
    for event_type in types:
        tp = sum(min(gold_map[i].get(event_type, 0), pred_map.get(i, Counter()).get(event_type, 0))
                 for i in gold_map)
        gp = sum(gold_map[i].get(event_type, 0) for i in gold_map)
        pp = sum(pred_map.get(i, Counter()).get(event_type, 0) for i in gold_map)
        scores.append(prf(tp, pp - tp, gp - tp)["f1"])
    return round(sum(scores) / len(scores), 4)


def evaluate(gold_rows: list[dict], pred_rows: list[dict], verbose: bool = True) -> dict:
    """计算触发词、论元、事件级精确匹配三级 P/R/F1。缺预测按全部漏检计。"""
    pred_map = {row_key(row): row for row in pred_rows}
    missing = 0
    tr = [0, 0, 0]
    ar = [0, 0, 0]
    ev = [0, 0, 0]
    for gold in gold_rows:
        gold_events = as_event_list(gold)
        pred = pred_map.get(row_key(gold))
        if pred is None:
            missing += 1
            pred_events = []
        else:
            pred_events = as_event_list(pred)
        t, a, e = compare_events(gold_events, pred_events)
        for i in range(3):
            tr[i] += t[i]
            ar[i] += a[i]
            ev[i] += e[i]
    result = {
        "num_samples": len(gold_rows),
        "missing_predictions": missing,
        "trigger": prf(*tr),
        "argument": prf(*ar),
        "event_exact": prf(*ev),
        "type_macro_f1": type_macro_f1(gold_rows, pred_rows),
    }
    if verbose:
        print(f"样本数 {result['num_samples']}（缺预测 {missing} 条）")
        for key in ("trigger", "argument", "event_exact"):
            metric = result[key]
            print(f"  {key:12s} P={metric['precision']:.4f} R={metric['recall']:.4f} "
                  f"F1={metric['f1']:.4f}")
        print(f"  type_macro_f1 = {result['type_macro_f1']:.4f}")
    return result


def write_metrics(name: str, result: dict, meta: dict | None = None) -> Path:
    ensure_dirs()
    payload = dict(result)
    if meta:
        payload["meta"] = meta
    path = METRIC_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print("written:", path)
    return path


def save_predictions(name: str, rows: list[dict]) -> Path:
    ensure_dirs()
    path = PRED_DIR / f"pred_{name}.jsonl"
    save_jsonl(rows, path)
    return path


def span_f1(rows: list[dict], events: list[dict | None]) -> float:
    """训练过程中用于选择 checkpoint 的 span 级 F1，非正式评测指标。"""
    tp = fp = fn = 0
    for row, event in zip(rows, events):
        gold_event = row_event(row) or {}
        gold = {(arg["role"], arg["text"]) for arg in gold_event.get("args") or []}
        trigger = (gold_event.get("trigger") or {}).get("text", "")
        if trigger:
            gold.add(("__trigger__", trigger))
        pred = {(arg["role"], arg["text"]) for arg in (event or {}).get("args", [])}
        pred_trigger = (event or {}).get("trigger", {}).get("text", "")
        if pred_trigger:
            pred.add(("__trigger__", pred_trigger))
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def peak_gpu_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)
