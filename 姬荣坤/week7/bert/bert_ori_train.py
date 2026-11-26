# -*- coding: utf-8 -*-
"""
BERT 事件抽取（BERT + 线性分类头，无 CRF）—— 标准 BIO 标注版

与旧版 bert_ori_train.py 的关键区别：
  1. 【核心修复】实体内部的 token 全部参与监督：
       第 1 个 token  -> B-<role>
       第 2..n 个 token -> I-<role>
     旧版只把首 token 标成 B-，其余标 -100，导致模型学不会实体边界，
     loss 很低（因为几乎所有 token 都是 O 或被忽略）但 F1 接近 0。
  2. 【速度】动态 padding：每个 batch 只补齐到 batch 内最长，不再统一补到 512。
  3. 【选模型】每个 epoch 在 dev 上算 Trigger F1，按 dev F1 保存最优模型
     （旧版按「训练 loss」保存，训练 loss 继续下降不代表泛化变好）。
  4. 标签顺序固定（sorted），多次运行结果可复现。

输出目录仍是 output/bert_ori，bert_ori_valid.py 无需改路径。
"""

import os
import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizerFast, BertForTokenClassification
from tqdm import tqdm

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ================= 配置 =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 512
BATCH_SIZE = 16
EPOCHS = 10
LR = 2e-5
WEIGHT_DECAY = 0.01
PATIENCE = 3          # dev F1 连续多少轮不提升就早停
SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ================= 标签体系 =================
def build_tag_dicts(train_path):
    """O + (B-/I-) × 每个角色，包括 Trigger。"""
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

    # 固定顺序，保证可复现
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
        if tok_end <= tok_start:          # [CLS] / [SEP] 等特殊 token
            continue
        if tok_start < end and tok_end > start:   # 与实体区间有重叠
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

                labels = [-100] * len(input_ids)
                for i, (s, e) in enumerate(offset_mapping):
                    if e > s:                     # 非特殊 token 默认 O
                        labels[i] = tag2id["O"]

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
    """动态 padding 到 batch 内最长。"""
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


# ================= 解码 & 评估 =================
def decode_entities(text, preds, offsets, id2tag):
    """标准 BIO 解码：B- 开启实体，I- 延长实体，O/异类 I 结束实体。"""
    ents, cur = [], None
    for i, pid in enumerate(preds):
        s, e = int(offsets[i][0]), int(offsets[i][1])
        if e <= s:                       # 特殊 token
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
            else:                        # 孤儿 I-，当作 B- 处理
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
def evaluate(model, tokenizer, id2tag, rows, batch_size=32):
    """返回 (tp, fp, fn)，针对 Trigger 的 (text, start) 严格匹配。"""
    model.eval()
    tp = fp = fn = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        enc = tokenizer(
            [r["text"] for r in batch], max_length=MAX_LEN, truncation=True,
            padding=True, return_offsets_mapping=True, return_tensors="pt",
        )
        logits = model(
            input_ids=enc["input_ids"].to(DEVICE),
            attention_mask=enc["attention_mask"].to(DEVICE),
        ).logits
        preds = logits.argmax(-1).cpu().numpy()
        offsets = enc["offset_mapping"].numpy()
        for k, r in enumerate(batch):
            ents = decode_entities(r["text"], preds[k], offsets[k], id2tag)
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
    save_dir = os.path.join(root_dir, "output", "bert_ori")
    os.makedirs(save_dir, exist_ok=True)

    print(f"✅ 当前使用的设备: {DEVICE}")
    print("1. 初始化字典与数据集...")
    tag2id, id2tag = build_tag_dicts(train_path)
    print(f"   标签数: {len(tag2id)}  (O + B/I × {len(tag2id)//2} 个角色)")

    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    dataset = EventDataset(train_path, tokenizer, tag2id)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0, drop_last=False,
    )
    print(f"   训练样本数: {len(dataset)}")

    dev_rows = []
    with open(dev_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dev_rows.append(json.loads(line))
    print(f"   验证样本数: {len(dev_rows)}")

    print("2. 加载预训练模型...")
    model = BertForTokenClassification.from_pretrained(
        model_path,
        num_labels=len(tag2id),
        id2label=id2tag,
        label2id=tag2id,
    ).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

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
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
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
            model.save_pretrained(save_dir)
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
