# -*- coding: utf-8 -*-
"""
消融实验：对比 9 种微调方式对文本分类效果的影响
========================================================
数据集：DAMO_NLP/yf_dianping（大众点评评论情感二分类，0=差评 1=好评）
模型  ：本地 week06 目录下的中文 RoBERTa backbone
指标  ：Micro-F1（在验证集 dev 上评测）
基线  ：实验 1（CLS 池化），其余 8 个实验每个只相对基线改【一处】，用来观察单一改动的影响。

为了方便你看懂，这里刻意写得直白、少抽象：
- 9 个实验用一个列表 EXPERIMENTS 描述，每个实验是一个字典，只写它跟基线不同的地方。
- 池化、损失函数、FGM 都是独立、显式的代码块，想看哪块直接往下翻即可。
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizerFast
from sklearn.metrics import f1_score


# =========================================================
# 0. 全局配置（想调快/调慢，改这里就行）
# =========================================================
MODEL_DIR   = "."            # 本地模型目录（week06 当前目录）
TRAIN_FILE  = "data/train.jsonl"
DEV_FILE    = "data/dev.jsonl"
NUM_LABELS  = 2              # 二分类
EPOCHS      = 2              # 每个实验训练几轮（9 个实验顺序跑，轮数别太大）
BATCH_SIZE  = 32
BASE_LR     = 2e-5           # 基线学习率（实验 5 会改成 3e-5）
BASE_MAXLEN = 128            # 基线最大长度（实验 6 会改成 256）
SEED        = 42

# 想先快速跑通流程，可以把下面两个设成一个小数字（比如 2000 / 1000），跑通后再设回 None 用全量数据
TRAIN_SUBSET = None          # None=用全部训练数据；或填一个整数只取前 N 条
DEV_SUBSET   = None          # None=用全部验证数据；或填一个整数只取前 N 条

# 设备：Mac 优先用 MPS，没有就用 CPU
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def set_seed(seed):
    """固定随机种子，让 9 个实验的对比尽量公平。"""
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# =========================================================
# 1. 数据：读 jsonl -> Dataset -> DataLoader
# =========================================================
def read_jsonl(path, subset=None):
    """读取 jsonl，每行是 {'sentence': ..., 'label': 0/1}。"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            data.append((item["sentence"], int(item["label"])))
    if subset is not None:
        data = data[:subset]
    return data


class DianpingDataset(Dataset):
    """只负责把 (文本, 标签) 一条条交出去，分词留给 collate_fn 做。"""
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def make_collate_fn(tokenizer, max_len):
    """返回一个 collate_fn：把一个 batch 的文本分词 + padding，并堆好标签。"""
    def collate_fn(batch):
        texts  = [text for text, label in batch]
        labels = [label for text, label in batch]
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc
    return collate_fn


# =========================================================
# 2. 模型：BERT backbone + 可切换的池化 + 一层分类头
# =========================================================
class BertClassifier(nn.Module):
    """
    pool 可选 4 种：
      'cls'      -> 取 [CLS] 位置的向量           （基线）
      'mean'     -> 对所有有效 token 向量求平均
      'max'      -> 对所有有效 token 向量求最大值
      'cls_mean' -> 把 CLS 向量和 mean 向量拼接   （维度翻倍）
    """
    def __init__(self, model_dir, num_labels, pool="cls"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_dir)
        self.pool = pool
        self.dropout = nn.Dropout(0.1)

        hidden = self.bert.config.hidden_size          # 768
        # 只有 cls_mean 是拼接，输入维度翻倍，其余都是 hidden
        in_dim = hidden * 2 if pool == "cls_mean" else hidden
        self.classifier = nn.Linear(in_dim, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        last_hidden = outputs.last_hidden_state          # [B, L, H] 每个 token 的向量

        # ----- 4 种池化，写得尽量直白 -----
        cls_vec = last_hidden[:, 0]                       # [B, H]，第 0 个位置就是 [CLS]

        # mask: [B, L] -> [B, L, 1]，方便和 [B, L, H] 相乘
        mask = attention_mask.unsqueeze(-1).float()

        # mean：只对真实 token（mask=1）求和，再除以真实 token 个数
        sum_vec  = (last_hidden * mask).sum(dim=1)        # [B, H]
        token_cnt = mask.sum(dim=1).clamp(min=1e-9)       # [B, 1]，避免除以 0
        mean_vec = sum_vec / token_cnt                    # [B, H]

        # max：把 padding 位置填成很小的数，再取最大，确保 padding 不会被选中
        masked_hidden = last_hidden.masked_fill(mask == 0, -1e9)
        max_vec = masked_hidden.max(dim=1).values         # [B, H]

        if self.pool == "cls":
            pooled = cls_vec
        elif self.pool == "mean":
            pooled = mean_vec
        elif self.pool == "max":
            pooled = max_vec
        elif self.pool == "cls_mean":
            pooled = torch.cat([cls_vec, mean_vec], dim=-1)  # [B, 2H]
        else:
            raise ValueError(f"未知池化方式: {self.pool}")

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)                  # [B, num_labels]
        return logits


# =========================================================
# 3. Focal Loss（实验 9 用）：让模型少被“容易分对的样本”带偏，更关注难样本
# =========================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, labels):
        # pt = 模型给“正确类别”的预测概率
        log_prob = torch.log_softmax(logits, dim=-1)      # [B, C]
        log_pt = log_prob.gather(1, labels.unsqueeze(1)).squeeze(1)  # [B]
        pt = log_pt.exp()
        # focal loss = -(1 - pt)^gamma * log(pt)，容易的样本(pt大)权重被压低
        loss = -((1 - pt) ** self.gamma) * log_pt
        return loss.mean()


# =========================================================
# 4. FGM 对抗训练（实验 7 用）：在词向量上加一点扰动，让模型更鲁棒
# =========================================================
class FGM:
    """
    用法（在训练循环里）：
        loss.backward()          # 先正常反向，得到梯度
        fgm.attack()             # 沿梯度方向给词向量加扰动
        loss_adv = ...           # 用扰动后的词向量再算一次 loss
        loss_adv.backward()      # 把对抗样本的梯度也累加进去
        fgm.restore()            # 还原词向量
        optimizer.step()
    """
    def __init__(self, model, epsilon=1.0, emb_name="word_embeddings"):
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()    # 先备份原始词向量
                norm = torch.norm(param.grad)             # 梯度的模长
                if norm != 0 and not torch.isnan(norm):
                    r = self.epsilon * param.grad / norm  # 扰动方向=梯度方向
                    param.data.add_(r)                    # 加扰动

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]            # 还原
        self.backup = {}


# =========================================================
# 5. 训练 + 评测一个实验，返回它在 dev 上的 Micro-F1
# =========================================================
def run_one_experiment(config, tokenizer, train_samples, dev_samples):
    # ---- 取出本实验的配置（没写的就用基线默认值）----
    pool            = config.get("pool", "cls")
    lr              = config.get("lr", BASE_LR)
    max_len         = config.get("max_len", BASE_MAXLEN)
    use_fgm         = config.get("fgm", False)
    label_smoothing = config.get("label_smoothing", 0.0)
    loss_type       = config.get("loss", "ce")       # 'ce' 交叉熵 / 'focal'

    set_seed(SEED)  # 每个实验用同样的种子，保证公平

    # ---- 数据加载器（max_len 可能随实验变化，所以在这里构建）----
    collate = make_collate_fn(tokenizer, max_len)
    train_loader = DataLoader(DianpingDataset(train_samples), batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate)
    dev_loader   = DataLoader(DianpingDataset(dev_samples), batch_size=BATCH_SIZE,
                              shuffle=False, collate_fn=collate)

    # ---- 模型、优化器、损失函数、FGM ----
    model = BertClassifier(MODEL_DIR, NUM_LABELS, pool=pool).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    if loss_type == "focal":
        criterion = FocalLoss(gamma=2.0)
    else:
        # label_smoothing=0 时就是普通交叉熵；>0 就是带标签平滑的交叉熵
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    fgm = FGM(model) if use_fgm else None

    # ---- 训练 ----
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            labels = batch.pop("labels")

            optimizer.zero_grad()
            logits = model(**batch)
            loss = criterion(logits, labels)
            loss.backward()                               # 正常反向

            if fgm is not None:                           # 实验 7：加一次对抗扰动
                fgm.attack()
                logits_adv = model(**batch)
                loss_adv = criterion(logits_adv, labels)
                loss_adv.backward()                       # 对抗梯度累加
                fgm.restore()

            optimizer.step()
            running_loss += loss.item()

            if (step + 1) % 100 == 0:
                print(f"    epoch {epoch+1} step {step+1}/{len(train_loader)} "
                      f"loss={running_loss/(step+1):.4f}")

    # ---- 评测：在 dev 上算 Micro-F1 ----
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            labels = batch.pop("labels")
            logits = model(**batch)
            preds = logits.argmax(dim=-1)                 # 取概率最大的类别
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    micro_f1 = f1_score(all_labels, all_preds, average="micro")
    return micro_f1


# =========================================================
# 6. 9 个实验的定义（基线=实验1，其余每个只改一处）
# =========================================================
EXPERIMENTS = [
    {"name": "1_CLS池化(基线)",      "pool": "cls"},
    {"name": "2_mean池化",           "pool": "mean"},
    {"name": "3_max池化",            "pool": "max"},
    {"name": "4_CLS+mean拼接",       "pool": "cls_mean"},
    {"name": "5_学习率3e-5",         "pool": "cls", "lr": 3e-5},
    {"name": "6_max_len=256",        "pool": "cls", "max_len": 256},
    {"name": "7_FGM对抗训练",        "pool": "cls", "fgm": True},
    {"name": "8_label_smoothing",    "pool": "cls", "label_smoothing": 0.1},
    {"name": "9_Focal_loss",         "pool": "cls", "loss": "focal"},
]


def main():
    print(f"设备: {DEVICE}")
    print("加载分词器与数据...")
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
    train_samples = read_jsonl(TRAIN_FILE, TRAIN_SUBSET)
    dev_samples   = read_jsonl(DEV_FILE, DEV_SUBSET)
    print(f"训练集 {len(train_samples)} 条，验证集 {len(dev_samples)} 条\n")

    results = {}
    for config in EXPERIMENTS:
        name = config["name"]
        print(f"==================== 实验 {name} ====================")
        print(f"配置: {config}")
        f1 = run_one_experiment(config, tokenizer, train_samples, dev_samples)
        results[name] = f1
        print(f">>> {name} 的 dev Micro-F1 = {f1:.4f}\n")

    # ---- 最终对比表：所有实验 vs 基线 ----
    baseline_name = EXPERIMENTS[0]["name"]
    baseline_f1 = results[baseline_name]
    print("\n==================== 消融实验对比（基线 = CLS池化）====================")
    print(f"{'实验':<22}{'Micro-F1':>10}{'相对基线':>12}")
    print("-" * 46)
    for config in EXPERIMENTS:
        name = config["name"]
        f1 = results[name]
        delta = f1 - baseline_f1
        if name == baseline_name:
            delta_str = "（基线）"
        else:
            delta_str = f"{delta:+.4f}"          # 带正负号，一眼看出比基线好还是差
        print(f"{name:<22}{f1:>10.4f}{delta_str:>12}")


if __name__ == "__main__":
    main()
