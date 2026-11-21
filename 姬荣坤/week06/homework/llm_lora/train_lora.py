import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, 
    Trainer, TrainingArguments, DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

DATASET_NAME = "sookiemonster/asrs-narratives-rebalance"
ROOT_DIR = Path(__file__).resolve().parent.parent


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, system_prompt, max_length=1024):
        self.features = []
        label_names = dataset.features["label"].names
        
        for item in dataset:
            text = item["text"]
            label_idx = item["label"]
            label_name = label_names[label_idx]
            
            # 1. 构建 prompt (System + User)
            prompt_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ]
            # 关闭 thinking，添加 generation prompt
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            
            # 2. 构建 response (Assistant)
            response_text = label_name + tokenizer.eos_token
            
            # 3. Tokenize
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
            response_ids = tokenizer(response_text, add_special_tokens=False).input_ids
            
            input_ids = prompt_ids + response_ids
            # 把 prompt 部分 mask 掉，不算 loss
            labels = [-100] * len(prompt_ids) + response_ids
            
            # 4. 截断 (如果太长)
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]
                
            self.features.append({
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": [1] * len(input_ids)
            })
            
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, i):
        return self.features[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="pretrain_models/Qwen3-0.6B")
    ap.add_argument("--output_dir", default="outputs/lora_adapter")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_length", type=int, default=1024)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = ROOT_DIR / args.model_name
    out_dir = ROOT_DIR / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[加载模型] {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        local_files_only=True
    )
    
    # 开启梯度检查点，省显存
    model.gradient_checkpointing_enable()
    
    # LoRA 配置
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"[加载数据] {DATASET_NAME} ...")
    ds = load_dataset(DATASET_NAME)
    train_ds = ds["train"]
    val_ds = ds["validation"]
    label_names = train_ds.features["label"].names
    
    # 构建 Prompt
    label_str = ", ".join(label_names)
    system_prompt = (
        "You are an expert text classification assistant. "
        "Classify the given text into exactly ONE of the following categories:\n"
        f"{label_str}\n\n"
        "Output ONLY the exact category name. Do not include any explanations, punctuation, or other text."
    )

    print("[预处理] 正在构建训练集与验证集 ...")
    train_dataset = SFTDataset(train_ds, tokenizer, system_prompt, args.max_length)
    val_dataset = SFTDataset(val_ds, tokenizer, system_prompt, args.max_length)

    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=(device == "cuda"),
        optim="adamw_torch",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
    )

    print("[训练] 开始 LoRA 微调 ...")
    trainer.train()

    print(f"[保存] 将 LoRA Adapter 保存至 {out_dir}")
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print("训练完成！")


if __name__ == "__main__":
    main()