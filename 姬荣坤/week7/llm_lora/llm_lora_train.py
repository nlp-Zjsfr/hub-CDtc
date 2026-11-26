# -*- coding: utf-8 -*-
"""
Qwen3-0.6B + LoRA 事件抽取训练脚本（生成式对话 / SFT）

任务：给一句话，让模型直接「生成」一个 JSON，里面是触发词和论元。

关键设计：
  1. 【提示词】system = 你指定的提示词；user = 原文 + 输出格式要求。
     提示词和输出格式在训练/推理两侧必须完全一致，所以抽成常量，
     并把 prompt_config.json 存进 adapter 目录，供 valid 复用。
  2. 【只输出 JSON】本地 transformers 没有 response_format 参数，
     等价做法：prompt 里给 JSON 骨架 + 严格约束 + 生成后健壮解析。
  3. 【保护预训练参数】LoRA 冻结原始权重；loss 只在 assistant 回答 token 上算。
  4. 【显存】见下面「显存预算」一节。

========================= 显存预算（重要） =========================
Qwen3-0.6B 的词表是 151936，logits 形状 [batch, seq, 151936]，

    logits 显存(bf16) ≈ batch × seq × 151936 × 2 字节

    batch=8, seq=1024 -> 2.5 GB
    而 HuggingFace 的 loss 会做 logits.float() -> 再翻一倍到 5.0 GB

这就是 16GB 显卡在 batch=8 时 OOM 的原因（报错里 trying to allocate
2090860544 / 2503999488 字节，正好对应 seq≈860 / 1030）。

本脚本的三道保险：
    a) BATCH_SIZE=4（logits 降到 1.25GB，fp32 副本 2.5GB）
    b) gradient_checkpointing（激活显存从 O(层数) 降到 O(1)）
    c) expandable_segments 减少显存碎片

如果还是 OOM，按这个顺序往下调（改文件顶部常量即可）：
    BATCH_SIZE 4 -> 2 -> 1，同时把 GRAD_ACCUM 翻倍保持有效 batch=16
    MAX_LEN 1024 -> 896 -> 768（会跳过少量长样本，脚本会打印跳过数）
====================================================================

数据：data/processed/train_extract.jsonl（与 BERT 训练集完全一致）
输出：output/llm_lora/（adapter + tokenizer + prompt_config.json + train_meta.json）
"""

import os

# 必须在第一次 CUDA 分配之前设置，缓解显存碎片
# 注意：Windows 上 PyTorch 不支持 expandable_segments，会刷一条 UserWarning，所以跳过
if os.name != "nt":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import re
import json
import math
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ================= 配置 =================
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LEN = 1024            # 样本 p95≈416、最大 808，留足余量
BATCH_SIZE = 4            # ★ 从 8 降到 4：logits 显存减半
GRAD_ACCUM = 4            # 有效 batch 仍 = 16
EPOCHS = 3
LR = 2e-4
WEIGHT_DECAY = 0.0
WARMUP_RATIO = 0.05
MAX_GRAD_NORM = 1.0
USE_GRAD_CKPT = True      # 梯度检查点：省激活显存，慢约 20%

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]

MAX_NEW_TOKENS = 384
EVAL_SUBSET = 120

# ---- 提示词（训练/推理必须一致） ----
SYSTEM_PROMPT = "请你作为一个语言提取助手，帮我提取下面语句中的信息。"
JSON_SCHEMA = ('{"events":[{"type":"事件类型","trigger":{"text":"触发词"},'
               '"args":[{"role":"角色","text":"论元"}]}]}')
EOS_TOKEN = "<|im_end|>"


def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ================= prompt / 答案构造 =================
def build_user_content(text):
    return (f"{text}\n\n"
            f"要求：只输出一个JSON对象，不要输出任何解释、Markdown代码块或多余文字。\n"
            f"输出格式：{JSON_SCHEMA}；如果没有事件，输出 {{\"events\":[]}}。")


def build_prompt_ids(tokenizer, text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_content(text)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
    )


def build_answer(event):
    obj = {"events": [{
        "type": event.get("type", ""),
        "trigger": {"text": event["trigger"]["text"]},
        "args": [{"role": a["role"], "text": a["text"]} for a in event.get("args", [])],
    }]}
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


# ================= 数据集 =================
class SFTDataset(Dataset):
    def __init__(self, path, tokenizer):
        self.samples = []
        self.skipped = 0
        eos_id = tokenizer.convert_tokens_to_ids(EOS_TOKEN)

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                text = item.get("text", "")
                event = item.get("event")
                if not text or not event:
                    continue

                prompt_ids = build_prompt_ids(tokenizer, text)
                answer = build_answer(event)
                answer_ids = tokenizer(answer, add_special_tokens=False).input_ids + [eos_id]

                input_ids = prompt_ids + answer_ids
                if len(input_ids) > MAX_LEN:
                    self.skipped += 1
                    continue

                labels = [-100] * len(prompt_ids) + answer_ids
                self.samples.append({
                    "input_ids": input_ids,
                    "labels": labels,
                    "n_prompt": len(prompt_ids),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, pad_token_id):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, attention_mask, labels = [], [], []
    for x in batch:
        pad = max_len - len(x["input_ids"])
        input_ids.append(x["input_ids"] + [pad_token_id] * pad)
        attention_mask.append([1] * len(x["input_ids"]) + [0] * pad)
        labels.append(x["labels"] + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


# ================= JSON 解析 =================
def extract_json(text):
    if not text:
        return None
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s).strip()

    start = s.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start:i + 1])
                    except json.JSONDecodeError:
                        return None
    return None


# ================= 生成 & 快速评估 =================
@torch.no_grad()
def generate_answers(model, tokenizer, texts, batch_size=BATCH_SIZE, max_new_tokens=MAX_NEW_TOKENS):
    model.eval()
    # 关掉 sampling 相关参数，避免每个 batch 都打印
    # "The following generation flags are not valid..." 警告
    try:
        gc = model.generation_config
        gc.do_sample = False
        gc.temperature = None
        gc.top_p = None
        gc.top_k = None
    except Exception:
        pass
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    answers = []
    try:
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            prompts = [build_prompt_ids(tokenizer, t) for t in chunk]
            enc = tokenizer.pad({"input_ids": prompts}, padding=True, return_tensors="pt")
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
            )
            new_tokens = out[:, enc["input_ids"].shape[1]:]
            answers.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    finally:
        tokenizer.padding_side = old_side
    model.train()
    return answers


def quick_trigger_f1(model, tokenizer, rows):
    answers = generate_answers(model, tokenizer, [r["text"] for r in rows])
    tp = fp = fn = 0
    for r, ans in zip(rows, answers):
        obj = extract_json(ans) or {"events": []}
        pred = set()
        for e in obj.get("events") or []:
            t = ((e.get("trigger") or {}).get("text") or "").strip()
            if t:
                pred.add(t)
        gt = {r["event"]["trigger"]["text"]} if r.get("event") else set()
        tp += len(pred & gt)
        fp += len(pred - gt)
        fn += len(gt - pred)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r_ = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r_ / (p + r_) if (p + r_) else 0.0
    return p, r_, f1


# ================= 主流程 =================
def main():
    set_seed()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)

    model_path = os.path.join(script_dir, "Qwen3-0.6B")
    train_path = os.path.join(root_dir, "data", "processed", "train_extract.jsonl")
    dev_path = os.path.join(root_dir, "data", "processed", "dev_extract_clean.jsonl")
    save_dir = os.path.join(root_dir, "output", "llm_lora")
    os.makedirs(save_dir, exist_ok=True)

    print(f"✅ 当前使用的设备: {DEVICE}")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"   GPU: {torch.cuda.get_device_name(0)}  "
              f"可用 {free/1e9:.1f}GB / 共 {total/1e9:.1f}GB")

    print("1. 加载 tokenizer 与数据集...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = SFTDataset(train_path, tokenizer)
    print(f"   训练样本: {len(dataset)}（因超长跳过 {dataset.skipped} 条）")

    dev_rows = []
    with open(dev_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dev_rows.append(json.loads(line))
    rng = random.Random(SEED)
    dev_subset = rng.sample(dev_rows, min(EVAL_SUBSET, len(dev_rows)))
    print(f"   验证样本: {len(dev_rows)}（每轮用 {len(dev_subset)} 条做监控）")

    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    print("2. 加载 Qwen3-0.6B 并注入 LoRA...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(DEVICE)
    model.config.use_cache = False

    if USE_GRAD_CKPT:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=LORA_TARGETS,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    if USE_GRAD_CKPT:
        model.enable_input_require_grads()   # 梯度检查点 + LoRA 必须开这个
    model.print_trainable_parameters()

    # 显存预算提示
    vocab = model.config.vocab_size
    worst_logits = BATCH_SIZE * MAX_LEN * vocab * 2 / 1e9
    print(f"   最坏情况 logits ≈ {worst_logits:.2f}GB（fp32 副本再 ×2 ≈ {worst_logits*2:.2f}GB）")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"   加载后剩余显存 {free/1e9:.1f}GB / {total/1e9:.1f}GB")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)

    steps_per_epoch = math.ceil(len(dataloader) / GRAD_ACCUM)
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"   优化步数: {total_steps}（warmup {warmup_steps}）")

    best_f1 = -1.0

    print("3. 开始训练...")
    for epoch in range(EPOCHS):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()      # 每轮开始清一次碎片
        model.train()
        running = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            (loss / GRAD_ACCUM).backward()

            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        train_loss = running / len(dataloader)

        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 1e9
            mem_txt = f" | 峰值显存 {peak:.1f}GB"
            torch.cuda.reset_peak_memory_stats()
        else:
            mem_txt = ""

        p, r, f1 = quick_trigger_f1(model, tokenizer, dev_subset)
        print(f" [Epoch {epoch + 1}] 训练Loss: {train_loss:.4f} | "
              f"Dev子集 Trigger  P={p:.4f} R={r:.4f} F1={f1:.4f}{mem_txt}")

        if f1 > best_f1:
            best_f1 = f1
            print(f" 🌟 Dev F1 提升到 {best_f1:.4f}，保存 LoRA adapter...")
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            with open(os.path.join(save_dir, "prompt_config.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "system_prompt": SYSTEM_PROMPT,
                    "json_schema": JSON_SCHEMA,
                    "max_len": MAX_LEN,
                    "enable_thinking": False,
                }, f, ensure_ascii=False, indent=2)

    # 兜底：如果一轮都没保存过（理论上不会发生），再存一次
    # 注意：这里不能无条件 save，否则会把「最优」adapter 覆盖成最后一轮的权重
    if best_f1 < 0:
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
    with open(os.path.join(save_dir, "train_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "base_model": model_path,
            "train_file": train_path,
            "n_train": len(dataset),
            "skipped": dataset.skipped,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "lr": LR,
            "max_len": MAX_LEN,
            "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT,
                     "targets": LORA_TARGETS},
            "best_dev_subset_f1": best_f1,
        }, f, ensure_ascii=False, indent=2)

    print(f"✅ 训练完成！LoRA adapter 已保存至 {save_dir}")
    print(f"   最优 Dev 子集 Trigger F1 = {best_f1:.4f}")


if __name__ == "__main__":
    main()
