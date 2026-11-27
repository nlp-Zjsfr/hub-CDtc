import math

import torch
from torch.utils.data import Dataset


def prompt_ids(tokenizer, row):
    return tokenizer.apply_chat_template(row["messages"][:2], tokenize=True, add_generation_prompt=True)


def encode_example(tokenizer, row, max_length):
    prompt = prompt_ids(tokenizer, row)
    # 明确分段编码，确保推理时的 prompt token 序列与训练前缀逐 token 相同。
    completion = tokenizer.encode(row["messages"][-1]["content"], add_special_tokens=False)
    eos = tokenizer.eos_token_id
    ids = prompt + completion + [eos]
    if len(ids) > max_length:
        raise ValueError(f"样本 {row['id']} 长度 {len(ids)} 超过 {max_length}；请调整数据或预算，不能静默切掉答案/EOS")
    labels = [-100] * len(prompt) + completion + [eos]
    return {"input_ids": ids, "labels": labels, "prompt_length": len(prompt), "id": row["id"]}


class SFTDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length):
        self.examples = [encode_example(tokenizer, r, max_length) for r in rows]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class SFTCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        length = math.ceil(max(len(x["input_ids"]) for x in examples) / 8) * 8
        ids = torch.full((len(examples), length), self.pad_token_id, dtype=torch.long)
        labels = torch.full_like(ids, -100)
        attention = torch.zeros_like(ids)
        for i, e in enumerate(examples):
            n = len(e["input_ids"])
            ids[i, :n] = torch.tensor(e["input_ids"])
            labels[i, :n] = torch.tensor(e["labels"])
            attention[i, :n] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": attention}


def save_encoding_audit(dataset, tokenizer):
    from common import ROOT, write_json
    e = dataset.examples[0]
    p = e["prompt_length"]
    indices = list(range(p - 4, min(p + 8, len(e["input_ids"])))) + [len(e["input_ids"]) - 1]
    write_json(ROOT / "results" / "mask_example.json", {
        "sample_id": e["id"], "prompt_length": p, "sequence_length": len(e["input_ids"]),
        "positions": [{"position": i, "token_id": e["input_ids"][i],
                       "decoded_piece": tokenizer.decode([e["input_ids"][i]], skip_special_tokens=False),
                       "label": e["labels"][i], "attention_mask": 1} for i in indices]})
    lengths = [len(x["input_ids"]) for x in dataset.examples]
    write_json(ROOT / "results" / "tokenization.json", {
        "samples": len(dataset), "min_sequence_length": min(lengths), "max_sequence_length": max(lengths),
        "mean_sequence_length": sum(lengths)/len(lengths),
        "supervised_answer_tokens": sum(sum(y != -100 for y in x["labels"]) for x in dataset.examples),
        "eos_token": tokenizer.eos_token, "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id,
        "truncated_examples": 0, "encoding_policy": "prompt encoded once as at inference; completion encoded separately; native EOS appended"})


def answer_loss_sum(logits, labels):
    # 本项目手工 shift 一次。调用模型时不传 labels，避免再被 HF 内部 shift。
    targets = labels[:, 1:]
    valid = targets.ne(-100)
    selected = logits[:, :-1][valid]
    loss = torch.nn.functional.cross_entropy(selected.float(), targets[valid], reduction="sum")
    return loss, int(valid.sum())
