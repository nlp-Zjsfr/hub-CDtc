# -*- coding: utf-8 -*-
"""L4: Qwen2.5-0.5B + LoRA SFT 数据血缘抽取。"""
import argparse
import json
import sys
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))

from io_utils import load_schema, load_split
from model_paths import qwen_path

OUT = ROOT / "outputs"
ADAPTER = OUT / "l4_lora_adapter"


def event_to_json(event: dict) -> str:
    obj = {
        "type": event["type"],
        "trigger": event["trigger"]["text"],
        "args": [{"role": a["role"], "text": a["text"]} for a in event.get("args") or []],
    }
    return json.dumps(obj, ensure_ascii=False)


def build_messages(schema, text, event=None):
    schema_txt = "\n".join(f"- {k}: {', '.join(v)}" for k, v in schema.items())
    system = (
        "You extract data-lineage events from credit/market risk documentation. "
        "Return ONE JSON object only. "
        '{"type":"...","trigger":"...","args":[{"role":"...","text":"..."}]}\n'
        f"Types:\n{schema_txt}"
    )
    user = f"Sentence:\n{text}\n\nJSON:"
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if event is not None:
        msgs.append({"role": "assistant", "content": event_to_json(event)})
    return msgs


class SFTDataset(Dataset):
    def __init__(self, rows, tokenizer, schema, max_len=512):
        self.rows = rows
        self.tok = tokenizer
        self.schema = schema
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        msgs = build_messages(self.schema, r["text"], r["event"])
        prompt_msgs = msgs[:-1]
        full = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        prompt = self.tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        full_ids = self.tok(full, truncation=True, max_length=self.max_len)["input_ids"]
        prompt_ids = self.tok(prompt, truncation=True, max_length=self.max_len)["input_ids"]
        labels = list(full_ids)
        # mask prompt tokens
        plen = min(len(prompt_ids), len(labels))
        for i in range(plen):
            labels[i] = -100
        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate(features, pad_id):
    max_len = max(len(f["input_ids"]) for f in features)
    B = len(features)
    ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    mask = torch.zeros(B, max_len, dtype=torch.long)
    labels = torch.full((B, max_len), -100, dtype=torch.long)
    for i, f in enumerate(features):
        n = len(f["input_ids"])
        ids[i, :n] = f["input_ids"]
        mask[i, :n] = f["attention_mask"]
        labels[i, :n] = f["labels"]
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_len", type=int, default=512)
    args = ap.parse_args()

    schema = load_schema()
    train = load_split("train")
    model_dir = str(qwen_path())
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float16, trust_remote_code=True
    )
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = SFTDataset(train, tok, schema, args.max_len)
    targs = TrainingArguments(
        output_dir=str(OUT / "l4_lora_runs"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        logging_steps=20,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=lambda feats: collate(feats, tok.pad_token_id),
    )
    trainer.train()
    ADAPTER.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ADAPTER)
    tok.save_pretrained(ADAPTER)
    print("saved adapter:", ADAPTER)


if __name__ == "__main__":
    main()
