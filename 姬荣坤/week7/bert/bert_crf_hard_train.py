# -*- coding: utf-8 -*-
"""
BERT + CRF + 硬约束（hard constraints）事件抽取训练脚本

在 bert_crf_train.py（BERT + CRF）基础上，把 CRF 的转移矩阵从「完全自由学习」
改成「只在合法转移里学习」，即把 BIO 标注体系的语法规则硬编码进 CRF：

    1. 序列不能以 I-X 开头（起始转移：I-* 全部置 -1e4）
    2. O 后面不能紧跟 I-X          （O -> I-X 非法）
    3. B-X 后面只能跟 I-X 或别的标签，不能跟 I-Y (Y≠X)
    4. I-X 后面只能跟 I-X，不能跟 I-Y (Y≠X)

    B-X -> I-X      允许
    I-X -> I-X      允许
    B-X -> B-Y      允许
    I-X -> B-Y      允许
    *   -> O        允许
    O   -> B-X      允许

为什么用 -1e4 而不是 -inf：
    -inf 参与 logsumexp / Viterbi 累加时容易产生 NaN（-inf + inf）。
    -1e4 已经远大于任何发射/转移分数的量级（通常 ±10），等价于概率 0，
    且全程是有限数，数值稳定。

关键实现细节（容易写错的地方）：
    硬约束不能只加一次。这些值本身是 nn.Parameter，optimizer.step() 会把它
    从 -1e4 往梯度方向挪。所以必须在「每次 optimizer.step() 之后」重新把它们
    钉回 -1e4，否则约束会逐渐失效（这就是很多实现里“加了约束但没生效”的原因）。

输出：output/bert_crf_hard/（bert_crf_hard_valid.py 从这里加载）
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
NEG = -1e4          # 非法转移/起始的分数（等价于概率 0，但数值稳定）


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ================= 标签体系 =================
def build_tag_dicts(train_path):
    """O + (B-/I-) × 每个角色。与 bert_crf_train.py 完全一致的顺序。"""
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


# ================= 硬约束：从标签体系推导非法转移 =================
def build_constraints(id2tag):
    """返回 (起始非法标签集合, 非法转移 (from, to) 列表)。

    规则：
      - 起始：不能以 I-* 开头
      - 转移到 I-X：前一个标签必须是 B-X 或 I-X
        （于是 O -> I-X、B-Y -> I-X(Y≠X)、I-Y -> I-X(Y≠X) 都被禁掉）
    """
    start_forbidden = [j for j, t in id2tag.items() if t.startswith("I-")]

    trans_forbidden = []
    for i, from_tag in id2tag.items():
        for j, to_tag in id2tag.items():
            if not to_tag.startswith("I-"):
                continue
            role = to_tag[2:]
            if from_tag not in (f"B-{role}", f"I-{role}"):
                trans_forbidden.append((i, j))
    return start_forbidden, trans_forbidden


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

                # CRF 的时间步必须连续：[CLS]/[SEP] 也标 O，只有 padding 是 -100
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


# ================= 模型：BERT + 硬约束 CRF =================
class BertCrfHardForTokenClassification(nn.Module):
    def __init__(self, bert: BertModel, id2tag, dropout: float = DROPOUT):
        super().__init__()
        num_labels = len(id2tag)
        self.bert = bert
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.num_labels = num_labels

        # 一次性推导出非法项
        self.start_forbidden, self.trans_forbidden = build_constraints(id2tag)
        self.apply_constraints()      # 初始化后立刻钉一次

    @torch.no_grad()
    def apply_constraints(self):
        """把非法起点 / 非法转移钉回 NEG。每次 optimizer.step() 之后都要调一次。"""
        dev = self.crf.transitions.device

        if self.start_forbidden:
            idx = torch.as_tensor(self.start_forbidden, dtype=torch.long, device=dev)
            self.crf.start_transitions[idx] = NEG

        if self.trans_forbidden:
            rows = torch.as_tensor([a for a, _ in self.trans_forbidden], dtype=torch.long, device=dev)
            cols = torch.as_tensor([b for _, b in self.trans_forbidden], dtype=torch.long, device=dev)
            self.crf.transitions[rows, cols] = NEG

    @torch.no_grad()
    def check_constraints(self):
        """校验约束确实生效；返回 (起点合法数, 转移合法数, 是否全部生效)。"""
        start_ok = (self.crf.start_transitions.cpu() > NEG / 2).sum().item()
        trans_ok = (self.crf.transitions.cpu() > NEG / 2).sum().item()
        total = self.num_labels * self.num_labels
        ok = (start_ok == self.num_labels - len(self.start_forbidden)
              and trans_ok == total - len(self.trans_forbidden))
        return start_ok, trans_ok, ok

    def emissions(self, input_ids, attention_mask):
        seq_out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.classifier(self.dropout(seq_out))

    def forward(self, input_ids, attention_mask, labels=None):
        emissions = self.emissions(input_ids, attention_mask)
        mask = attention_mask.bool()
        if labels is not None:
            tags = labels.clone()
            tags[tags < 0] = 0                  # torchcrf 要求 tags >= 0
            log_likelihood = self.crf(emissions, tags, mask=mask, reduction="sum")
            n_tokens = mask.sum().clamp(min=1)  # 按有效 token 数归一化，梯度尺度与 bert_ori 一致
            return -log_likelihood / n_tokens
        return self.crf.decode(emissions, mask=mask)


# ================= 解码 & 评估 =================
def decode_entities(text, preds, offsets, id2tag):
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
            preds = paths[k]
            offs = offsets[k][:len(preds)]
            results.append(decode_entities(text, preds, offs, id2tag))
    return results


def evaluate(model, tokenizer, id2tag, rows, batch_size=32):
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
    save_dir = os.path.join(root_dir, "output", "bert_crf_hard")
    os.makedirs(save_dir, exist_ok=True)

    print(f"✅ 当前使用的设备: {DEVICE}")
    print("1. 初始化字典与数据集...")
    tag2id, id2tag = build_tag_dicts(train_path)
    print(f"   标签数: {len(tag2id)}  (O + B/I × {len(tag2id)//2} 个角色)")

    start_forbidden, trans_forbidden = build_constraints(id2tag)
    total_trans = len(tag2id) ** 2
    print(f"   硬约束: 禁止 {len(start_forbidden)} 个 I-* 作为起点；"
          f"禁止 {len(trans_forbidden)}/{total_trans} 条非法转移")
    print(f"   （合法的起始标签 {len(tag2id) - len(start_forbidden)} 个，"
          f"合法转移 {total_trans - len(trans_forbidden)} 条）")

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

    print("2. 加载预训练 BERT 并接上「硬约束 CRF」...")
    bert = BertModel.from_pretrained(model_path)
    model = BertCrfHardForTokenClassification(bert, id2tag).to(DEVICE)
    n_bert = sum(p.numel() for p in model.bert.parameters())
    n_head = sum(p.numel() for p in model.classifier.parameters()) + sum(p.numel() for p in model.crf.parameters())
    print(f"   BERT 参数: {n_bert/1e6:.1f}M | 分类头+CRF 参数: {n_head/1e6:.3f}M")
    _, _, ok0 = model.check_constraints()
    print(f"   约束初始化检查: {'✅ 通过' if ok0 else '❌ 失败'}")

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
            model.apply_constraints()        # ★ 关键：每步之后把非法转移钉回去

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
            model.apply_constraints()        # 保存前再钉一次，保证落盘的是约束后的参数
            print(f" 🌟 Dev F1 提升到 {best_f1:.4f}，保存模型...")
            torch.save({
                "model_state": model.state_dict(),
                "bert_path": model_path,
                "num_labels": len(tag2id),
            }, os.path.join(save_dir, "bert_crf_hard_model.pt"))
            tokenizer.save_pretrained(save_dir)
            with open(os.path.join(save_dir, "tag_dict.json"), "w", encoding="utf-8") as f:
                json.dump({"tag2id": tag2id, "id2tag": id2tag}, f, ensure_ascii=False, indent=2)
        else:
            bad_epochs += 1
            print(f" Dev F1 未提升（已 {bad_epochs}/{PATIENCE} 轮）")
            if bad_epochs >= PATIENCE:
                print(f" ⏹ 早停：连续 {PATIENCE} 轮没有提升。")
                break

    _, _, ok_end = model.check_constraints()
    print(f"✅ 训练完成！最优模型已保存至 {save_dir}  (best Dev Trigger F1 = {best_f1:.4f})")
    print(f"   约束最终检查: {'✅ 仍然全部生效' if ok_end else '❌ 有非法转移被学回来了'}")


if __name__ == "__main__":
    main()
