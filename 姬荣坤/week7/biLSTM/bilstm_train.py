import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchcrf import CRF
from collections import defaultdict
from tqdm import tqdm
import random
import numpy as np

# ================= 配置参数 =================
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-3
MAX_LEN = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# ================= 1. 数据处理与标签构建 =================
def build_tag_dicts(train_path):
    roles, triggers = set(), set()
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            event = item.get("event")
            if event:
                if event.get("trigger", {}).get("text"):
                    triggers.add("Trigger")
                for arg in event.get("args", []):
                    if arg.get("role"):
                        roles.add(arg["role"])
    
    # 构建 BIO 标签
    tag2id = {"O": 0}
    for t in triggers.union(roles):
        tag2id[f"B-{t}"] = len(tag2id)
        tag2id[f"I-{t}"] = len(tag2id)
    id2tag = {v: k for k, v in tag2id.items()}
    return tag2id, id2tag

def build_char_vocab(train_path):
    chars = set()
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            chars.update(item.get("text", ""))
    char2id = {"<PAD>": 0, "<UNK>": 1}
    for c in chars:
        char2id[c] = len(char2id)
    return char2id

class EventDataset(Dataset):
    def __init__(self, path, char2id, tag2id):
        self.data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                text = item.get("text", "")
                if not text or len(text) > MAX_LEN: continue
                
                tags = ["O"] * len(text)
                event = item.get("event")
                
                if event:
                    # 填 Trigger
                    trig = event.get("trigger")
                    if trig and trig.get("start") is not None:
                        s, e = trig["start"], trig["end"]
                        if e <= len(text):
                            tags[s] = "B-Trigger"
                            for i in range(s+1, e): tags[i] = "I-Trigger"
                    
                    # 填 Args
                    for arg in event.get("args", []):
                        role = arg.get("role")
                        s, e = arg.get("start"), arg.get("end")
                        if role and s is not None and e <= len(text):
                            tags[s] = f"B-{role}"
                            for i in range(s+1, e): tags[i] = f"I-{role}"
                
                x_ids = [char2id.get(c, char2id["<UNK>"]) for c in text]
                y_ids = [tag2id.get(t, 0) for t in tags]
                self.data.append((x_ids, y_ids))

    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx): return self.data[idx]

def collate_fn(batch):
    max_len = max(len(x) for x, _ in batch)
    x_pad, y_pad, masks = [], [], []
    for x, y in batch:
        pad_len = max_len - len(x)
        x_pad.append(x + [0]*pad_len)
        y_pad.append(y + [0]*pad_len)
        masks.append([1]*len(x) + [0]*pad_len)
    return torch.tensor(x_pad), torch.tensor(y_pad), torch.tensor(masks, dtype=torch.bool)

# ================= 2. 模型定义 =================
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag2id):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM, padding_idx=0)
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(HIDDEN_DIM, len(tag2id))
        self.crf = CRF(len(tag2id), batch_first=True)

    def forward(self, x, tags=None, mask=None):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        emissions = self.hidden2tag(lstm_out)
        if tags is not None:
            # 返回负对数似然损失 (用于训练)
            return -self.crf(emissions, tags, mask=mask, reduction='mean')
        else:
            # 返回解码结果 (用于推理)
            return self.crf.decode(emissions, mask=mask)

# ================= 3. 训练流程 =================
def main():
    set_seed(42)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    train_path = os.path.join(root_dir, "data", "processed", "train_extract.jsonl")
    save_dir = os.path.join(root_dir, "output", "bilstm_crf")
    os.makedirs(save_dir, exist_ok=True)
    
    print("1. 正在构建字典与数据集...")
    tag2id, id2tag = build_tag_dicts(train_path)
    char2id = build_char_vocab(train_path)
    
    dataset = EventDataset(train_path, char2id, tag2id)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    print(f"   数据量: {len(dataset)} 条, 标签类别数: {len(tag2id)}")

    print("2. 初始化 BiLSTM+CRF 模型...")
    print(f"当前使用的设备是: {DEVICE}")
    model = BiLSTM_CRF(len(char2id), tag2id).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("3. 开始训练...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for x, y, mask in pbar:
            x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
            optimizer.zero_grad()
            loss = model(x, y, mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
    # 保存模型及字典
    torch.save({
        "model_state": model.state_dict(),
        "char2id": char2id,
        "tag2id": tag2id,
        "id2tag": id2tag
    }, os.path.join(save_dir, "bilstm_crf_model.pt"))
    print(f"✅ 模型已保存至: {save_dir}/bilstm_crf_model.pt")

if __name__ == "__main__":
    main()