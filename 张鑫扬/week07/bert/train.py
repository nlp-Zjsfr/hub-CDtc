# -*- coding: utf-8 -*-
"""
Week7 homework：BERT 微调做 NER（conllpp，英文，只预测第 4 列 NER 标签）

任务定义
  - 输入：conllpp 的 4 列格式（token / POS / chunk / NER），本脚本**只用第 1 列和第 4 列**；
    第 2、3 列是 CoNLL-2003 附带的句法标注，不是本任务的预测目标。
  - 输出：每个 word 一个 BIO 标签（9 类：O + PER/LOC/ORG/MISC 的 B-/I-）。
  - 标签对齐：数据集已按词切好，用 `is_split_into_words=True` 交给 BERT 做子词化；
    一个词的标签只贴在**首个子词**上，其余子词与 [CLS]/[SEP]/[PAD] 一律 -100。
    推理时同样只取每个词首个子词的预测，还原成 word 级 BIO，再解码成 span。

消融实验（4 组，用 --ablation 一键切换）
  full       : BERT + dropout + CRF 转移矩阵          （默认主力）
  no_crf     : 取消 CRF，改线性头（每 token 独立 softmax）
  crf_hard   : CRF + 硬编码约束（非法转移分数直接置 -1e4，BIO 非法序列严格归零）
  len_half   : max_len 减半（192 → 96），看截断对 F1 的影响

用法
  python bert/train.py                                  # 全量：BERT+CRF，max_len=192
  python bert/train.py --ablation no_crf
  python bert/train.py --ablation crf_hard
  python bert/train.py --ablation len_half
  python bert/train.py --all_ablations                   # 依次跑完 4 组并打印对比表
  python bert/train.py --epochs 1 --batch_size 64        # 快速跑

产物（默认写到 homework/output/bert/）
  model/<run_name>.pt           checkpoint（含 label 体系与超参）
  log/metrics_<run_name>.json   dev/test 指标 + 非法 BIO 统计 + 延迟
  log/log_<run_name>.json       逐 epoch 训练曲线

依赖：torch / transformers（参考 week7/code/requirements.txt）
  - 预训练模型默认用 bert-base-cased：优先找本地 pretrain_models/bert-base-cased，
    找不到就回退到 HuggingFace 名字（需要联网）。必须用 cased —— 大小写是 PER/LOC/ORG 的强信号。
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import copy
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertModel, get_linear_schedule_with_warmup

HERE = Path(__file__).resolve().parent          # homework/bert
DATA_DIR = HERE.parent / "data" / "conllpp"     # homework/data/conllpp
MODEL_DIR = HERE.parent / "output" / "bert" / "model"   # checkpoint；运行时会被 --model_dir 覆盖
LOG_DIR = HERE.parent / "output" / "bert" / "log"       # 训练日志与指标；运行时会被 --log_dir 覆盖
IGNORE_INDEX = -100
NEG_INF = -1e4                                  # 用大负数代替 -inf，避免反传出 NaN

LABELS = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC"]
LABEL2ID = {lb: i for i, lb in enumerate(LABELS)}
ID2LABEL = {i: lb for lb, i in LABEL2ID.items()}


# ─────────────────────────── 数据读取 ───────────────────────────

def load_conll(path: Path) -> list[dict]:
    """
    读 conll 格式，返回 [{"words": [...], "tags": [...]}, ...]。

    两个必须处理的点：
      1. 按空行切句（原始文件句间恰好一个空行）；
      2. 丢掉 `-DOCSTART- -X- -X- O` 文档边界行 —— train 有 945 行、dev 215、test 230，
         不过滤的话会多出一堆「单 token 句子」，句子数和 README 对不上。
    """
    sentences, words, tags = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if words:
                    sentences.append({"words": words, "tags": tags})
                    words, tags = [], []
                continue
            parts = line.split()
            if len(parts) != 4:                     # 防御：格式不完整就跳过
                continue
            if parts[0] == "-DOCSTART-":
                continue
            if parts[3] not in LABEL2ID:            # 防御：未知标签直接报错，别静默变 O
                raise ValueError(f"发现未知 NER 标签 {parts[3]!r}（{path}）")
            words.append(parts[0])
            tags.append(parts[3])
    if words:
        sentences.append({"words": words, "tags": tags})
    return sentences


class NERDataset(Dataset):
    """词级样本 → BERT 子词输入 + 对齐后的 token 级标签。"""

    def __init__(self, rows: list[dict], tokenizer, max_len: int, with_labels: bool = True):
        self.rows = rows
        self.tok = tokenizer
        self.max_len = max_len
        self.with_labels = with_labels
        # 截断句数在初始化时算一次（不能放在 __getitem__ 里自增，否则每轮都会重复计数）
        self.n_truncated = 0
        if with_labels:
            for r in rows:
                n = len(self.tok(r["words"], is_split_into_words=True)["input_ids"])
                if n > max_len:
                    self.n_truncated += 1

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        words = row["words"]
        enc = self.tok(
            words,
            is_split_into_words=True,   # 关键：词边界由数据集给定，tokenizer 只做子词化
            max_length=self.max_len,
            truncation=True,
            padding=False,              # 动态 padding 交给 collate_fn（省算力）
            return_tensors=None,
        )
        word_ids = enc.word_ids()       # token → 第几个词；[CLS]/[SEP] 为 None

        item = {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            # -1 表示特殊符；推理时要把子词预测还原回词级，必须留着这个映射
            "word_ids": torch.tensor([-1 if w is None else w for w in word_ids], dtype=torch.long),
            "n_words": torch.tensor(len(words), dtype=torch.long),
        }
        if "token_type_ids" in enc and enc["token_type_ids"] is not None:
            item["token_type_ids"] = torch.tensor(enc["token_type_ids"], dtype=torch.long)

        if self.with_labels:
            # labels   ：线性头用，只有词的首片背标签，续片/特殊符 = -100（CE 会忽略）
            # crf_tags ：CRF 用，**续片沿用同一个词的标签**、特殊符 = O。
            #   为什么不能把续片直接填 O：多词实体中间若夹着一个被切成多片的词，
            #   就会出现 B-PER → O → I-PER 这种「金标路径本身非法」的情况。
            #   软约束下只是噪声，硬约束下每一处都会被罚 -1e4，loss 直接爆到几千、F1 崩塌。
            aligned, crf_tags, prev = [], [], None
            for wid in word_ids:
                if wid is None:
                    aligned.append(IGNORE_INDEX)
                    crf_tags.append(LABEL2ID["O"])
                else:
                    lab = LABEL2ID[row["tags"][wid]]
                    if wid != prev:
                        aligned.append(lab)          # 词的第一个子片：背标签
                        prev = wid
                    else:
                        aligned.append(IGNORE_INDEX)  # 续片：不参与 CE
                    crf_tags.append(lab)             # CRF 路径上续片保持同标签，链路才合法
            item["labels"] = torch.tensor(aligned, dtype=torch.long)
            item["crf_tags"] = torch.tensor(crf_tags, dtype=torch.long)
        return item


def collate_fn(batch: list[dict]) -> dict:
    """动态 padding：只补到 batch 内最长，不按全局 max_len 补。"""
    max_len = max(int(b["input_ids"].numel()) for b in batch)
    out = {
        "input_ids": torch.zeros(len(batch), max_len, dtype=torch.long),
        "attention_mask": torch.zeros(len(batch), max_len, dtype=torch.long),
        "word_ids": torch.full((len(batch), max_len), -1, dtype=torch.long),
        "n_words": torch.stack([b["n_words"] for b in batch]),
    }
    has_tt = all("token_type_ids" in b for b in batch)
    if has_tt:
        out["token_type_ids"] = torch.zeros(len(batch), max_len, dtype=torch.long)
    if "labels" in batch[0]:
        out["labels"] = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
        out["crf_tags"] = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, b in enumerate(batch):
        n = int(b["input_ids"].numel())
        out["input_ids"][i, :n] = b["input_ids"]
        out["attention_mask"][i, :n] = b["attention_mask"]
        out["word_ids"][i, :n] = b["word_ids"]
        if has_tt:
            out["token_type_ids"][i, :n] = b["token_type_ids"]
        if "labels" in b:
            out["labels"][i, :n] = b["labels"]
            out["crf_tags"][i, :n] = b["crf_tags"]
    return out


# ─────────────────────────── CRF 层 ───────────────────────────

class CRF(nn.Module):
    """
    线性链 CRF：发射分数 + 可学习转移矩阵 + 前向算法（logZ）+ 维特比解码。

    为什么 NER 要 CRF：线性头对每个 token 独立 softmax，不知道 BIO 的结构约束
    （`I-PER` 前面必须是 `B-PER`/`I-PER`），会喷出非法序列。CRF 用一条全局路径打分，
    非法路径天然分低；打开 constraint 后可以把非法转移直接置 -1e4，严格归零。
    """

    def __init__(self, num_tags: int, constraint: torch.Tensor | None = None,
                 start_constraint: torch.Tensor | None = None):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        if constraint is None:
            constraint = torch.ones(num_tags, num_tags, dtype=torch.bool)
        if start_constraint is None:
            start_constraint = torch.ones(num_tags, dtype=torch.bool)
        self.register_buffer("constraint", constraint.bool())
        self.register_buffer("start_constraint", start_constraint.bool())

    def _trans(self) -> torch.Tensor:
        if self.constraint.all():
            return self.transitions
        return self.transitions.masked_fill(~self.constraint, NEG_INF)

    def _start(self) -> torch.Tensor:
        if self.start_constraint.all():
            return self.start_transitions
        return self.start_transitions.masked_fill(~self.start_constraint, NEG_INF)

    def _gold_score(self, emissions, tags, mask) -> torch.Tensor:
        trans = self._trans()
        emit = emissions.gather(2, tags.unsqueeze(-1)).squeeze(-1)
        score = self._start()[tags[:, 0]] + emit[:, 0]
        for i in range(1, emissions.size(1)):
            step = trans[tags[:, i - 1], tags[:, i]] + emit[:, i]
            score = score + step * mask[:, i].to(score.dtype)
        seq_ends = mask.long().sum(dim=1) - 1
        last_tags = tags.gather(1, seq_ends.unsqueeze(1)).squeeze(1)
        return score + self.end_transitions[last_tags]

    def _log_z(self, emissions, mask) -> torch.Tensor:
        trans = self._trans()
        alpha = self._start().unsqueeze(0) + emissions[:, 0]
        for i in range(1, emissions.size(1)):
            scores = alpha.unsqueeze(2) + trans.unsqueeze(0) + emissions[:, i].unsqueeze(1)
            new_alpha = torch.logsumexp(scores, dim=1)
            alpha = torch.where(mask[:, i].unsqueeze(1).bool(), new_alpha, alpha)
        return torch.logsumexp(alpha + self.end_transitions.unsqueeze(0), dim=1)

    def neg_log_likelihood(self, emissions, tags, mask) -> torch.Tensor:
        mask = mask.bool()
        return (self._log_z(emissions, mask) - self._gold_score(emissions, tags, mask)).mean()

    @torch.no_grad()
    def decode(self, emissions, mask) -> list[list[int]]:
        mask = mask.bool()
        B, L, _ = emissions.shape
        trans = self._trans()
        alpha = self._start().unsqueeze(0) + emissions[:, 0]
        backpointers = torch.zeros(B, L, self.num_tags, dtype=torch.long, device=emissions.device)
        for i in range(1, L):
            scores = alpha.unsqueeze(2) + trans.unsqueeze(0)
            best_score, best_prev = scores.max(dim=1)
            new_alpha = best_score + emissions[:, i]
            alpha = torch.where(mask[:, i].unsqueeze(1).bool(), new_alpha, alpha)
            backpointers[:, i] = best_prev
        best_last = (alpha + self.end_transitions.unsqueeze(0)).argmax(dim=1)
        lengths = mask.long().sum(dim=1)
        paths = []
        for b in range(B):
            Lb = int(lengths[b].item())
            tag = int(best_last[b].item())
            path = [tag]
            for i in range(Lb - 1, 0, -1):
                tag = int(backpointers[b, i, tag].item())
                path.append(tag)
            paths.append(path[::-1])
        return paths


def build_constraints() -> tuple[torch.Tensor, torch.Tensor]:
    """
    BIO 硬约束：True=允许。
      - 转移：`I-X` 只能从 `B-X` 或 `I-X` 转移而来（O→I-X、B-Y→I-X 都非法）
      - 开头：只允许 O 或 B-X（转移矩阵管不到第一个 token）
    """
    C = len(LABELS)
    trans = torch.ones(C, C, dtype=torch.bool)
    for i, prev in enumerate(LABELS):
        for j, cur in enumerate(LABELS):
            if cur.startswith("I-") and (prev == "O" or prev[2:] != cur[2:]):
                trans[i, j] = False
    start = torch.tensor([not lb.startswith("I-") for lb in LABELS], dtype=torch.bool)
    return trans, start


# ─────────────────────────── 模型 ───────────────────────────

class BertNER(nn.Module):
    """BERT + dropout + (Linear | CRF)。两种解码器共用同一份 encoder / 标签体系，指标才可比。"""

    def __init__(self, bert_path: str, num_tags: int, dropout: float = 0.1,
                 use_crf: bool = True, hard_constraint: bool = False, bert_layers: int = 0):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        if bert_layers and bert_layers > 0:
            # 消融用：保留预训练权重，只截前 N 层（不是重新随机初始化 N 层）
            self.bert.encoder.layer = nn.ModuleList(list(self.bert.encoder.layer)[:bert_layers])
            self.bert.config.num_hidden_layers = bert_layers
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_tags)
        self.num_tags = num_tags
        self.use_crf = use_crf
        if use_crf:
            trans_c, start_c = build_constraints() if hard_constraint else (None, None)
            self.crf = CRF(num_tags, constraint=trans_c, start_constraint=start_c)
        else:
            self.crf = None

    def emissions(self, input_ids, attention_mask, token_type_ids=None) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids, return_dict=True)
        return self.classifier(self.dropout(out.last_hidden_state))   # (B, L, C)

    def forward(self, input_ids, attention_mask, labels=None, token_type_ids=None, crf_tags=None):
        emit = self.emissions(input_ids, attention_mask, token_type_ids)
        loss = None
        if labels is not None:
            if self.use_crf:
                # crf_tags：续片沿用同词标签、特殊符为 O，金标路径天然合法（见 NERDataset）
                if crf_tags is None:
                    crf_tags = labels.clone()
                    crf_tags[crf_tags == IGNORE_INDEX] = 0
                loss = self.crf.neg_log_likelihood(emit, crf_tags, attention_mask)
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
        return [pred[b, :int(attention_mask[b].sum().item())].tolist()
                for b in range(pred.size(0))]


# ─────────────────────────── 解码 / 指标 ───────────────────────────

def to_word_tags(paths, word_ids, n_words, id2label) -> list[list[str]]:
    """子词级预测 → 词级 BIO：每个词只认它首个子词的预测（与训练标签对齐规则一致）。"""
    all_tags = []
    for path, wids, nw in zip(paths, word_ids.tolist(), n_words.tolist()):
        tags = ["O"] * nw
        prev = None
        for j, wid in enumerate(wids):
            if wid < 0 or j >= len(path):
                continue
            if wid != prev and wid < nw:
                tags[wid] = id2label[path[j]]
            prev = wid
        all_tags.append(tags)
    return all_tags


def spans_from_bio(tags: list[str]) -> set[tuple]:
    """BIO → 实体集合 {(start, end, type)}，end 左闭右开。非法序列做兜底（保住实体）。"""
    spans, cur = set(), None
    for i, tag in enumerate(tags):
        if tag == "O":
            if cur:
                spans.add(cur)
                cur = None
            continue
        prefix, etype = tag[0], tag[2:]
        if prefix == "B":
            if cur:
                spans.add(cur)
            cur = (i, i + 1, etype)
        else:  # I-
            if cur and cur[2] == etype:
                cur = (cur[0], i + 1, etype)
            else:  # 非法：I 开头 或 B-x→I-y，当作新实体起点
                if cur:
                    spans.add(cur)
                cur = (i, i + 1, etype)
    if cur:
        spans.add(cur)
    return spans


def illegal_stats(tags_list: list[list[str]]) -> dict:
    """统计模型原始输出里的非法 BIO（linear vs CRF 的核心对比点）。"""
    stat = {"total_seqs": len(tags_list), "illegal_seqs": 0,
            "illegal_start": 0, "illegal_transition": 0}
    for seq in tags_list:
        bad = 0
        if seq and seq[0].startswith("I-"):
            stat["illegal_start"] += 1
            bad += 1
        for i in range(1, len(seq)):
            if not seq[i].startswith("I-"):
                continue
            prev, cur = seq[i - 1], seq[i]
            if prev == "O" or prev[2:] != cur[2:]:
                stat["illegal_transition"] += 1
                bad += 1
        if bad:
            stat["illegal_seqs"] += 1
    return stat


def compute_metrics(gold_list: list[list[str]], pred_list: list[list[str]]) -> dict:
    """实体级（span exact match）micro P/R/F1 + 每种类型的分项。"""
    types = ["PER", "LOC", "ORG", "MISC"]
    tp = {t: 0 for t in types}
    fp = {t: 0 for t in types}
    fn = {t: 0 for t in types}
    for gold, pred in zip(gold_list, pred_list):
        g = spans_from_bio(gold)
        p = spans_from_bio(pred)
        for s in p:
            (tp if s in g else fp)[s[2]] += 1
        for s in g - p:
            fn[s[2]] += 1

    def prf(a, b, c):
        prec = a / (a + b) if a + b else 0.0
        rec = a / (a + c) if a + c else 0.0
        return {"precision": round(prec, 4), "recall": round(rec, 4),
                "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0}

    per_type = {t: prf(tp[t], fp[t], fn[t]) for t in types}
    overall = prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    overall["per_type"] = per_type
    return overall


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(requested: str = "") -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def batch_to_device(batch: dict, device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def evaluate(model, loader, device) -> tuple[float, dict, list[list[str]]]:
    """返回 (平均 loss, 指标, 词级预测标签)。

    指标里同时报两份非法 BIO 统计：
      - illegal       词级（把子词聚合回词之后）—— 这是最终评测口径
      - illegal_token 子词级（CRF 真正解码的序列）—— 硬约束在 token 级把非法转移压到 0，
        但词级仍可能残留少量：连续子词被丢弃后，相邻两个词的首片在词级看起来可能非法。
    """
    model.eval()
    total, n = 0.0, 0
    preds, tok_preds = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            _, loss = model(batch["input_ids"], batch["attention_mask"],
                            batch["labels"], batch.get("token_type_ids"),
                            crf_tags=batch.get("crf_tags"))
            total += float(loss.item())
            n += 1
            paths = model.predict(batch["input_ids"], batch["attention_mask"],
                                  batch.get("token_type_ids"))
            for path, wids in zip(paths, batch["word_ids"].tolist()):
                tok_preds.append([ID2LABEL[t] for t, w in zip(path, wids) if w >= 0])
            preds.extend(to_word_tags(paths, batch["word_ids"], batch["n_words"], ID2LABEL))
    golds = [r["tags"] for r in loader.dataset.rows]
    metrics = compute_metrics(golds, preds)
    metrics["illegal"] = illegal_stats(preds)
    metrics["illegal_token"] = illegal_stats(tok_preds)
    return total / max(n, 1), metrics, preds


def run(args, loaders: dict) -> dict:
    """跑一组配置（训练 → 按 dev F1 选最优 → 在 dev/test 上出最终指标）。"""
    device = get_device(args.device)
    set_seed(args.seed)

    model = BertNER(
        bert_path=args.bert_path,
        num_tags=len(LABELS),
        dropout=args.dropout,
        use_crf=(args.decoder == "crf"),
        hard_constraint=args.hard_constraint,
        bert_layers=args.bert_layers,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    enc_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (enc_params if name.startswith("bert.") else head_params).append(p)
    groups = [{"params": enc_params, "lr": args.lr}]
    if head_params:
        groups.append({"params": head_params, "lr": args.lr * args.head_lr_mult})
    optimizer = AdamW(groups, weight_decay=args.weight_decay)

    total_steps = max(1, len(loaders["train"]) * args.epochs // max(args.grad_accum, 1))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps)

    print(f"\n{'=' * 70}\n配置：{args.run_name} | decoder={args.decoder} | "
          f"max_len={args.max_len} | hard_constraint={args.hard_constraint} | "
          f"dropout={args.dropout} | 设备={device} | 参数={n_params / 1e6:.1f}M\n{'=' * 70}")

    best_f1, best_epoch, log = -1.0, 0, []
    t_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        optimizer.zero_grad()
        epoch_t = time.perf_counter()
        for step, batch in enumerate(loaders["train"]):
            batch = batch_to_device(batch, device)
            _, loss = model(batch["input_ids"], batch["attention_mask"],
                            batch["labels"], batch.get("token_type_ids"),
                            crf_tags=batch.get("crf_tags"))
            (loss / args.grad_accum).backward()
            total += float(loss.item())
            n += 1
            if (step + 1) % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        if len(loaders["train"]) % args.grad_accum:
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        dev_loss, dev_m, _ = evaluate(model, loaders["dev"], device)
        dur = time.perf_counter() - epoch_t
        print(f"epoch {epoch}/{args.epochs} | train_loss={total / max(n, 1):.4f} | "
              f"dev_loss={dev_loss:.4f} | dev_P={dev_m['precision']:.4f} "
              f"R={dev_m['recall']:.4f} F1={dev_m['f1']:.4f} | "
              f"非法序列={dev_m['illegal']['illegal_seqs']} | {dur:.0f}s", flush=True)
        log.append({"epoch": epoch, "train_loss": round(total / max(n, 1), 6),
                    "dev_loss": round(dev_loss, 6), "dev_metrics": dev_m,
                    "elapsed_s": round(dur, 1)})
        if dev_m["f1"] > best_f1:
            best_f1, best_epoch = dev_m["f1"], epoch
            torch.save({"state_dict": model.state_dict(), "labels": LABELS,
                        "args": vars(args), "epoch": epoch, "dev_f1": best_f1},
                       MODEL_DIR / f"{args.run_name}.pt")
            print(f"  ★ dev F1 创新高 → 保存 {args.run_name}.pt", flush=True)

    train_s = time.perf_counter() - t_start

    ckpt = torch.load(MODEL_DIR / f"{args.run_name}.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    print(f"载入最优 checkpoint（epoch {ckpt['epoch']}，dev F1={best_f1:.4f}），"
          f"开始出最终指标……")

    t_inf = time.perf_counter()
    _, dev_metrics, dev_preds = evaluate(model, loaders["dev"], device)
    dev_ms = (time.perf_counter() - t_inf) * 1000 / max(len(loaders["dev"].dataset), 1)

    result = {
        "run_name": args.run_name,
        "config": {"decoder": args.decoder, "max_len": args.max_len,
                   "hard_constraint": args.hard_constraint, "dropout": args.dropout,
                   "bert_layers": args.bert_layers, "epochs": args.epochs,
                   "batch_size": args.batch_size, "lr": args.lr,
                   "head_lr_mult": args.head_lr_mult, "seed": args.seed},
        "best_epoch": best_epoch,
        "params_m": round(n_params / 1e6, 1),
        "train_s": round(train_s, 1),
        "ms_per_sentence": round(dev_ms, 2),
        "n_truncated_dev": loaders["dev"].dataset.n_truncated,
        "dev": dev_metrics,
    }

    if not args.skip_test:
        _, test_metrics, _ = evaluate(model, loaders["test"], device)
        result["test"] = test_metrics
        result["n_truncated_test"] = loaders["test"].dataset.n_truncated
        print(f"\n【test 最终指标（只跑一次，不参与任何选择）】"
              f"P={test_metrics['precision']:.4f} R={test_metrics['recall']:.4f} "
              f"F1={test_metrics['f1']:.4f} | "
              f"非法序列 词级={test_metrics['illegal']['illegal_seqs']} "
              f"token级={test_metrics['illegal_token']['illegal_seqs']}")

    with open(LOG_DIR / f"log_{args.run_name}.json", "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    with open(LOG_DIR / f"metrics_{args.run_name}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n【{args.run_name} 汇总】dev F1={dev_metrics['f1']:.4f} | "
          f"非法序列 词级={dev_metrics['illegal']['illegal_seqs']}/"
          f"{dev_metrics['illegal']['total_seqs']} "
          f"token级={dev_metrics['illegal_token']['illegal_seqs']} | "
          f"{dev_ms:.2f} ms/句 | 截断 {loaders['dev'].dataset.n_truncated} 句")
    return result


# ─────────────────────────── 入口 ───────────────────────────

ABLATIONS = {
    "full":     {"decoder": "crf",    "hard_constraint": False, "len_half": False},
    "no_crf":   {"decoder": "linear", "hard_constraint": False, "len_half": False},
    "crf_hard": {"decoder": "crf",    "hard_constraint": True,  "len_half": False},
    "len_half": {"decoder": "crf",    "hard_constraint": False, "len_half": True},
}


def resolve_bert_path(given: str) -> str:
    """优先本地 pretrain_models/bert-base-cased（离线可跑），否则回退 HF 名字（需联网）。"""
    if given:
        return given
    for parent in HERE.parents:
        local = parent / "pretrain_models" / "bert-base-cased"
        if local.is_dir() and (any(local.glob("*.safetensors")) or (local / "pytorch_model.bin").exists()):
            return str(local)
        # 兼容 week7/code/pretrain_models 这种多一层的情况
        local = parent / "code" / "pretrain_models" / "bert-base-cased"
        if local.is_dir() and (any(local.glob("*.safetensors")) or (local / "pytorch_model.bin").exists()):
            return str(local)
    print("⚠ 本地没找到 bert-base-cased，回退到 HuggingFace（首次运行需要联网下载 ~400MB）")
    return "bert-base-cased"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="BERT 微调做 conllpp NER（含 4 组消融）")
    ap.add_argument("--ablation", choices=list(ABLATIONS), default=None,
                    help="一键切换消融配置：full / no_crf / crf_hard / len_half")
    ap.add_argument("--all_ablations", action="store_true", help="依次跑完 4 组并打印对比表")
    ap.add_argument("--data_dir", default=str(DATA_DIR))
    ap.add_argument("--model_dir", default=str(MODEL_DIR), help="checkpoint 存放目录")
    ap.add_argument("--log_dir", default=str(LOG_DIR), help="训练日志/指标存放目录")
    ap.add_argument("--bert_path", default="", help="本地模型目录或 HF 名字；留空自动解析")
    ap.add_argument("--bert_layers", type=int, default=0, help="0=完整 12 层；N=只保留前 N 层")
    ap.add_argument("--decoder", choices=["linear", "crf"], default="crf")
    ap.add_argument("--hard_constraint", action="store_true",
                    help="CRF 硬约束：非法 BIO 转移分数置 -1e4")
    ap.add_argument("--max_len", type=int, default=192,
                    help="默认 192：实测最长句 183 个子词（含特殊符 185），128 会截断并丢实体")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head_lr_mult", type=float, default=5.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="", help="cuda / mps / cpu；留空自动选")
    ap.add_argument("--run_name", default="")
    ap.add_argument("--skip_test", action="store_true", help="不跑 test（调试时用）")
    args = ap.parse_args()

    if args.ablation:
        cfg = ABLATIONS[args.ablation]
        args.decoder = cfg["decoder"]
        args.hard_constraint = cfg["hard_constraint"]
        if cfg["len_half"]:
            args.max_len = max(16, args.max_len // 2)
    if not args.run_name:
        tag = f"bert{args.bert_layers}L" if args.bert_layers else "bert"
        extra = "_hard" if args.hard_constraint else ""
        args.run_name = f"{tag}_{args.decoder}_{args.max_len}{extra}"
    return args


def build_loaders(args, tokenizer) -> dict:
    data_dir = Path(args.data_dir)
    splits = {name: load_conll(data_dir / f"{name}.txt")
              for name in ("train", "dev", "test")}
    print(f"数据：train {len(splits['train'])} 句 / dev {len(splits['dev'])} 句 / "
          f"test {len(splits['test'])} 句（已过滤 -DOCSTART-）")
    return {
        name: DataLoader(
            NERDataset(rows, tokenizer, args.max_len, with_labels=True),
            batch_size=args.batch_size, shuffle=(name == "train"),
            num_workers=0, collate_fn=collate_fn)
        for name, rows in splits.items()
    }


def main():
    global MODEL_DIR, LOG_DIR
    args = parse_args()
    args.bert_path = resolve_bert_path(args.bert_path)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_path, use_fast=True)
    print(f"tokenizer：{args.bert_path}（fast={tokenizer.is_fast}）")
    MODEL_DIR = Path(args.model_dir)
    LOG_DIR = Path(args.log_dir)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not args.all_ablations:
        loaders = build_loaders(args, tokenizer)
        run(args, loaders)
        return

    results = {}
    for name in ABLATIONS:
        cfg_args = copy.deepcopy(args)
        cfg_args.ablation = name
        cfg_args.all_ablations = False
        cfg_args.max_len = args.max_len          # 先复原，避免 len_half 被反复减半
        preset = ABLATIONS[name]
        cfg_args.decoder = preset["decoder"]
        cfg_args.hard_constraint = preset["hard_constraint"]
        if preset["len_half"]:
            cfg_args.max_len = max(16, args.max_len // 2)
        cfg_args.run_name = f"abl_{name}"
        loaders = build_loaders(cfg_args, tokenizer)
        results[name] = run(cfg_args, loaders)

    print("\n" + "=" * 92)
    print(f"{'消融':<12}{'dev F1':>9}{'test F1':>10}{'非法(词)':>10}{'非法(token)':>12}"
          f"{'ms/句':>9}{'截断':>7}")
    print("-" * 92)
    for name, r in results.items():
        print(f"{name:<12}{r['dev']['f1']:>9.4f}{r.get('test', {}).get('f1', float('nan')):>10.4f}"
              f"{r['dev']['illegal']['illegal_seqs']:>10}"
              f"{r['dev']['illegal_token']['illegal_seqs']:>12}"
              f"{r['ms_per_sentence']:>9.2f}{r['n_truncated_dev']:>7}")
    print("=" * 92)
    with open(LOG_DIR / "ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
