"""
L4 Qwen2.5-0.5B-Instruct + LoRA SFT（SemEval-2020 Task 11 手法分类）

教学重点（对照课件）：
  1. chat 模板一致；
  2. Loss Masking：prompt 段 labels=-100，只在答案 token 上算 loss；
  3. print_trainable_parameters() 防止 target_modules 写错却静默无效。
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import load_splits
from common.labels import BOUNDARY_NOTES, LABELS
from common.model_paths import qwen_path

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

LABEL_BLOCK = "\n".join(f"- {n}: {BOUNDARY_NOTES[n]}" for n in LABELS)
SYSTEM = (
    "你是宣传手法分类助手。给定英文新闻片段，把它分到下列类别之一。"
    "只输出类别名，必须与列表中的字符串完全一致。\n"
    + LABEL_BLOCK
)


class SFTDataset(Dataset):
    def __init__(self, data, tok, max_length=384):
        self.data, self.tok, self.max_length = data, tok, max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]
        answer = LABELS[d["label"]]
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": d["sentence"]}]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = self.tok(answer, add_special_tokens=False)["input_ids"] + [self.tok.eos_token_id]
        room = self.max_length - len(a_ids)
        if room < 1:
            a_ids = a_ids[: self.max_length]
            room = self.max_length - len(a_ids)
        p_ids = p_ids[:room]
        input_ids = p_ids + a_ids
        labels = [-100] * len(p_ids) + a_ids
        return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels)}


def collate(batch, pad_id):
    L = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = b["input_ids"]
        lab[i, :n] = b["labels"]
        att[i, :n] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_train", type=int, default=0, help="0=用全部训练样本")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=768)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="从 outputs/lora_adapter 接着训")
    args = ap.parse_args()

    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model_dir = str(qwen_path())
    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype)
    adapter_dir = OUT_DIR / "lora_adapter"
    if args.resume:
        model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=True).to(device)
    else:
        cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(base, cfg).to(device)
    model.print_trainable_parameters()

    train, val = load_splits()
    if args.num_train > 0:
        train = train[: args.num_train]
    val = val[: min(200, len(val))]
    train_dl = DataLoader(
        SFTDataset(train, tok, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, tok.pad_token_id),
    )
    val_dl = DataLoader(
        SFTDataset(val, tok, args.max_length),
        batch_size=args.batch_size,
        collate_fn=lambda b: collate(b, tok.pad_token_id),
    )

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    log_path = OUT_DIR / "lora_train_log.json"
    log = json.loads(log_path.read_text(encoding="utf-8")) if args.resume and log_path.exists() else []
    step = 0
    t0 = time.time()
    start_epoch = len(log) + 1

    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        running, n_batch = 0.0, 0
        for i, batch in enumerate(train_dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                loss = model(**batch).loss / args.grad_accum
            scaler.scale(loss).backward()
            if (i + 1) % args.grad_accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1
            running += loss.item() * args.grad_accum
            n_batch += 1
            if i % 50 == 0:
                print(f"  epoch {epoch} {i}/{len(train_dl)} loss={loss.item()*args.grad_accum:.4f}", flush=True)
            if args.max_steps and step >= args.max_steps:
                break

        model.eval()
        vloss, vn = 0.0, 0
        with torch.no_grad():
            for batch in val_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                vloss += model(**batch).loss.item()
                vn += 1
        row = {
            "epoch": epoch,
            "train_loss": round(running / max(1, n_batch), 4),
            "val_loss": round(vloss / max(1, vn), 4),
            "steps": step,
            "elapsed_s": round(time.time() - t0, 1),
            "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2) if device == "cuda" else None,
        }
        log.append(row)
        print(
            f"epoch {epoch}: train_loss={row['train_loss']} val_loss={row['val_loss']} "
            f"peak_mem_gb={row['peak_mem_gb']}",
            flush=True,
        )
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(adapter_dir))
        tok.save_pretrained(str(adapter_dir))
        (OUT_DIR / "lora_train_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        if device == "cuda":
            torch.cuda.empty_cache()
        if args.max_steps:
            break

    print(f"adapter → {adapter_dir}")


if __name__ == "__main__":
    main()
