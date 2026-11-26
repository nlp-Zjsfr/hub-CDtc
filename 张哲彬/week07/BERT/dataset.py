import torch
from torch.utils.data import Dataset
import json

def load_labels(labels_path):
    """
    加载标签列表。
    labels_path: str，标签文件路径
    返回:
        labels2idx: dict，标签到索引的映射
        idx2labels: dict，索引到标签的映射
    """
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    labels2idx = {label: idx for idx, label in enumerate(labels)}
    idx2labels = {v: k for k, v in labels2idx.items()}
    return labels2idx, idx2labels

class PeopleDailyDataset(Dataset):
    def __init__(self, data_path, tokenizer, labels2idx, max_length=128):
        """
        data_path: str，数据文件路径
        tokenizer: BertTokenizer，分词器
        labels2idx: dict，标签到索引的映射
        max_length: int，最大序列长度
        """
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.labels2idx = labels2idx
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        tokens = record["tokens"]
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        ner_tags = record["ner_tags"]
        # 将标签转换为索引
        label_ids = [self.labels2idx[label] for label in ner_tags]
        word_ids = encoding.word_ids(batch_index=0)  # 获取每个 token 对应的 word 索引
        aligned_label_ids = []
        previous_word_id = None

        for word_id in word_ids:
            if word_id is None:
                aligned_label_ids.append(-100)
            elif word_id != previous_word_id:
                aligned_label_ids.append(label_ids[word_id])
            else:
                aligned_label_ids.append(-100)
            previous_word_id = word_id

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "labels": torch.tensor(aligned_label_ids, dtype=torch.long),
        }
