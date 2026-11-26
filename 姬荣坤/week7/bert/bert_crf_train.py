# -*- coding: utf-8 -*-
"""
BERT + CRF 事件抽取训练脚本

在 bert_ori_train.py（标准 BIO 版的 BERT + 线性分类头）基础上，把最后的
「逐 token 独立 softmax」换成「CRF 层」，让标签之间不再独立预测：

    BERT 编码  ->  Linear(num_labels) 得到发射分数 emissions
               ->  CRF 建模相邻标签的转移，解码时做 Viterbi

CRF 带来的好处：
  - 解码时天然满足标签序列约束（不会出现 O -> I-质押方 这类非法序列）
  - 实体边界更连贯，通常提升 Precision（减少碎片化实体）

实现说明（重要）：
  1. 用 torchcrf.CRF（与 bilstm_train.py 同一个库）。
  2. torchcrf 要求 tags >= 0，所以进入 CRF 前把 -100（padding）替换成 0。
  3. torchcrf 的 mask 必须是「前缀掩码」（True...True,False...False），
     因此本脚本把 [CLS]/[SEP] 也标成 O 参与计算（不能再标 -100），
     只有 padding 是 -100。-CLS/SEP 的 offset 为 (0,0)，解码时会被自动跳过。
  4. 优化器分两组：BERT 用 2e-5，CRF/分类头（随机初始化）用 5e-4。
  5. 每个 epoch 在 dev 上算 Trigger F1，按 dev F1 保存最优模型 + 早停。

输出：output/bert_crf/（bert_crf_valid.py 从这里加载）
"""

import os
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torchcrf import CRF
from transformers import BertTokenizerFast, BertModel
from tqdm import tqdm

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ================= 配置 =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 512
BATCH_SIZE = 16
EPOCHS = 15
LR_BERT = 2e-5
LR_HEAD = 5e-4
WEIGHT_DECAY = 0.01
PATIENCE = 4
SEED = 42
DROPOUT = 0.1


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ================= 标签体系 =================
def build_tag_dicts(train_path):
    """O + (B-/I-) × 每个角色。与 bert_ori_train.py 完全一致的顺序。"""
    roles = set()
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            event = item.get("event")
            if not event:
                continue
            if event.get("trigger", {}).get("text"):
                roles.add("Trigger")
            for arg in event.get("args", []):
                if arg.get("role"):
                    roles.add(arg["role"])

    ordered = ["Trigger"] + sorted(r for r in roles if r != "Trigger")
    tag2id = {"O": 0}
    for role in ordered:
        tag2id[f"B-{role}"] = len(tag2id)
        tag2id[f"I-{role}"] = len(tag2id)
    id2tag = {v: k for k, v in tag2id.items()}
    return tag2id, id2tag


def fill_label(labels, offset_mapping, start, end, role, tag2id):
    """实体覆盖的第 1 个 token -> B-role，其余 -> I-role。"""
    first = True
    for i, (tok_start, tok_end) in enumerate(offset_mapping):
        if tok_end <= tok_start:
            continue
        if tok_start < end and tok_end > start:
            if first:
                labels[i] = tag2id[f"B-{role}"]
                first = False
            else:
                labels[i] = tag2id[f"I-{role}"]


# ================= 数据集 =================
class EventDataset(Dataset):
    def __init__(self, path, tokenizer, tag2id):
        self.items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                text = item.get("text", "")
                if not text:
                    continue

                enc = tokenizer(
                    text, max_length=MAX_LEN, truncation=True,
                    return_offsets_mapping=True,
                )
                input_ids = enc["input_ids"]
                offset_mapping = enc["offset_mapping"]

                # CRF 需要连续的时间步，[CLS]/[SEP] 也标 O（padding 在 collate 里给 -100）
                labels = [tag2id["O"]] * len(input_ids)

                event = item.get("event")
                if event:
                    trig = event.get("trigger")
                    if trig and trig.get("start") is not None:
                        fill_label(labels, offset_mapping, trig["start"], trig["end"], "Trigger", tag2id)
                    for arg in event.get("args", []):
                        role = arg.get("role")
                        if role and arg.get("start") is not None:
                            fill_label(labels, offset_mapping, arg["start"], arg["end"], role, tag2id)

                self.items.append({
                    "input_ids": input_ids,
                    "labels": labels,
                    "text": text,
                    "sent_id": item.get("sent_id"),
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn(batch):
    """动态 padding 到 batch 内最长；padding 的 label 用 -100。"""
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, attention_mask, labels = [], [], []
    for x in batch:
        pad = max_len - len(x["input_ids"])
        input_ids.append(x["input_ids"] + [0] * pad)
        attention_mask.append([1] * len(x["input_ids"]) + [0] * pad)
        labels.append(x["labels"] + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


# ================= 模型：BERT + CRF =================
class BertCrfForTokenClassification(nn.Module):
    def __init__(self, bert: BertModel, num_labels: int, dropout: float = DROPOUT):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.num_labels = num_labels

    def emissions(self, input_ids, attention_mask):
        seq_out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.classifier(self.dropout(seq_out))

    def forward(self, input_ids, attention_mask, labels=None):
        emissions = self.emissions(input_ids, attention_mask)
        mask = attention_mask.bool()
        if labels is not None:
            tags = labels.clone()
            tags[tags < 0] = 0          # torchcrf 要求 tags >= 0；被 mask 掉的位置不影响结果
            # 按「有效 token 数」归一化：
            # CRF 的 NLL 是整条序列求和，若按 batch 求平均，回传到 BERT 的梯度会被
            # 放大 seq_len 倍（相当于把学习率放大两个数量级）。除以 token 数后，
            # 梯度尺度与 bert_ori_train.py 的 token-mean 交叉熵一致，可直接用同样的 lr。
            log_likelihood = self.crf(emissions, tags, mask=mask, reduction="sum")
            n_tokens = mask.sum().clamp(min=1)
            return -log_likelihood / n_tokens
        # 推理：Viterbi 解码，返回每个样本「非 padding 长度」的标签 id 列表
        return self.crf.decode(emissions, mask=mask)


# ================= 解码 & 评估 =================
def decode_entities(text, preds, offsets, id2tag):
    """标准 BIO 解码（与 bert_ori_valid.py 相同）。"""
    ents, cur = [], None
    for i, pid in enumerate(preds):
        s, e = int(offsets[i][0]), int(offsets[i][1])
        if e <= s:
            if cur:
                ents.append(cur)
                cur = None
            continue
        tag = id2tag.get(int(pid), "O")
        if tag.startswith("B-"):
            if cur:
                ents.append(cur)
            cur = {"type": tag[2:], "start": s, "end": e}
        elif tag.startswith("I-"):
            if cur is not None and cur["type"] == tag[2:]:
                cur["end"] = e
            else:
                if cur:
                    ents.append(cur)
                cur = {"type": tag[2:], "start": s, "end": e}
        else:
            if cur:
                ents.append(cur)
                cur = None
    if cur:
        ents.append(cur)
    for x in ents:
        x["text"] = text[x["start"]:x["end"]]
    return ents


@torch.no_grad()
def predict_entities(model, tokenizer, id2tag, texts, batch_size=32):
    """批量推理，返回每条文本的实体列表。"""
    model.eval()
    results = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tokenizer(
            chunk, max_length=MAX_LEN, truncation=True,
            padding=True, return_offsets_mapping=True, return_tensors="pt",
        )
        paths = model(
            enc["input_ids"].to(DEVICE),
            enc["attention_mask"].to(DEVICE),
        )
        offsets = enc["offset_mapping"].numpy()
        for k, text in enumerate(chunk):
            preds = paths[k]                       # 长度 = 该样本真实 token 数
            offs = offsets[k][:len(preds)]
            results.append(decode_entities(text, preds, offs, id2tag))
    return results


def evaluate(model, tokenizer, id2tag, rows, batch_size=32):
    """dev 上的 Trigger (text, start) 严格匹配统计量 (tp, fp, fn)。"""
    ents_list = predict_entities(model, tokenizer, id2tag, [r["text"] for r in rows], batch_size)
    tp = fp = fn = 0
    for r, ents in zip(rows, ents_list):
        pred_trig = {(e["text"], e["start"]) for e in ents if e["type"] == "Trigger"}
        gt = r.get("event")
        gt_trig = {(gt["trigger"]["text"], gt["trigger"]["start"])} if gt else set()
        tp += len(pred_trig & gt_trig)
        fp += len(pred_trig - gt_trig)
        fn += len(gt_trig - pred_trig)
    return tp, fp, fn


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


# ================= 主流程 =================
def main():
    set_seed()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)

    train_path = os.path.join(root_dir, "data", "processed", "train_extract.jsonl")
    dev_path = os.path.join(root_dir, "data", "processed", "dev_extract_clean.jsonl")
    model_path = os.path.join(root_dir, "bert", "bert-base-chinese")
    save_dir = os.path.join(root_dir, "output", "bert_crf")
    os.makedirs(save_dir, exist_ok=True)

    print(f"✅ 当前使用的设备: {DEVICE}")
    print("1. 初始化字典与数据集...")
    tag2id, id2tag = build_tag_dicts(train_path)
    print(f"   标签数: {len(tag2id)}  (O + B/I × {len(tag2id)//2} 个角色)")

    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    dataset = EventDataset(train_path, tokenizer, tag2id)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    print(f"   训练样本数: {len(dataset)}")

    dev_rows = []
    with open(dev_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dev_rows.append(json.loads(line))
    print(f"   验证样本数: {len(dev_rows)}")

    print("2. 加载预训练 BERT 并接上 CRF...")
    bert = BertModel.from_pretrained(model_path)
    model = BertCrfForTokenClassification(bert, num_labels=len(tag2id)).to(DEVICE)
    n_bert = sum(p.numel() for p in model.bert.parameters())
    n_head = sum(p.numel() for p in model.classifier.parameters()) + sum(p.numel() for p in model.crf.parameters())
    print(f"   BERT 参数: {n_bert/1e6:.1f}M | 分类头+CRF 参数: {n_head/1e6:.3f}M")

    optimizer = AdamW([
        {"params": model.bert.parameters(), "lr": LR_BERT, "weight_decay": WEIGHT_DECAY},
        {"params": list(model.classifier.parameters()) + list(model.crf.parameters()),
         "lr": LR_HEAD, "weight_decay": 0.0},
    ])

    best_f1 = -1.0
    bad_epochs = 0

    print("3. 开始训练...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for batch in pbar:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            loss = model(input_ids, attention_mask, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = total_loss / len(dataloader)

        tp, fp, fn = evaluate(model, tokenizer, id2tag, dev_rows)
        p, r, f1 = prf(tp, fp, fn)
        print(f" [Epoch {epoch + 1}] 训练Loss: {epoch_loss:.4f} | "
              f"Dev Trigger  P={p:.4f} R={r:.4f} F1={f1:.4f}  (tp={tp} fp={fp} fn={fn})")

        if f1 > best_f1:
            best_f1 = f1
            bad_epochs = 0
            print(f" 🌟 Dev F1 提升到 {best_f1:.4f}，保存模型...")
            torch.save({
                "model_state": model.state_dict(),
                "bert_path": model_path,
                "num_labels": len(tag2id),
            }, os.path.join(save_dir, "bert_crf_model.pt"))
            tokenizer.save_pretrained(save_dir)
            with open(os.path.join(save_dir, "tag_dict.json"), "w", encoding="utf-8") as f:
                json.dump({"tag2id": tag2id, "id2tag": id2tag}, f, ensure_ascii=False, indent=2)
        else:
            bad_epochs += 1
            print(f" Dev F1 未提升（已 {bad_epochs}/{PATIENCE} 轮）")
            if bad_epochs >= PATIENCE:
                print(f" ⏹ 早停：连续 {PATIENCE} 轮没有提升。")
                break

    print(f"✅ 训练完成！最优模型已保存至 {save_dir}  (best Dev Trigger F1 = {best_f1:.4f})")


if __name__ == "__main__":
    main()
