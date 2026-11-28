"""
方法 B：Qwen2.5-0.5B-Instruct + LoRA SFT，把「句子 → 实体 JSON」当生成任务学。

  样本 = chat_template(system, user) + 目标 JSON + <|im_end|>
  labels：prompt 部分置 -100，只对 assistant 回复（JSON + <|im_end|>）算交叉熵
  LoRA：q/k/v/o_proj，r=8，alpha=16；基座 bf16 冻结，adapter 由 peft 自动升到 fp32

MPS 注意：训练用 sdpa 没问题；**generate() 在 MPS 上 sdpa 会崩**，评测脚本改用 eager。

  python llm_sft/train_lora.py --smoke      # 64 条 × 4 步冒烟
  python llm_sft/train_lora.py              # 正式：9,674 条 × 2 epoch
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

# MPS 显存水位：默认 1.7/1.4 会把缓存涨到整机内存上限，导致系统 swap、单步耗时从 1.4s 恶化到 20s。
# 调到 1.0/0.9：超出就回收缓存块，长时间训练才稳定。（必须在 import torch 之前设）
if sys.platform == "darwin":
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.0")
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.9")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "llm_sft"))
from prompt import build_messages, build_target  # noqa: E402
from shared.data import load_jsonl, subsample  # noqa: E402
from shared.paths import CKPT_DIR, DATA_DIR, LOG_DIR, QWEN_DIR, ensure_dirs  # noqa: E402


def encode(tok, row, max_len):
    p_ids = tok.apply_chat_template(build_messages(row["text"]), add_generation_prompt=True, tokenize=True)
    a_ids = tok(build_target(row["entities"]) + "<|im_end|>", add_special_tokens=False)["input_ids"]
    ids = (p_ids + a_ids)[:max_len]
    labels = ([-100] * len(p_ids) + a_ids)[:max_len]
    return {"input_ids": ids, "labels": labels}


def collate(items, pad_id, T=None):
    T = T or max(len(x["input_ids"]) for x in items)
    ids = torch.full((len(items), T), pad_id, dtype=torch.long)
    lab = torch.full((len(items), T), -100, dtype=torch.long)
    att = torch.zeros((len(items), T), dtype=torch.long)
    for b, x in enumerate(items):
        n = len(x["input_ids"])
        ids[b, :n] = torch.tensor(x["input_ids"])
        lab[b, :n] = torch.tensor(x["labels"])
        att[b, :n] = 1
    return ids, att, lab


def make_buckets(data, batch_size, grad_accum, seed=0, pad_mult=16):
    """
    按 token 长度分桶后再切 micro-batch，pad 到 pad_mult 的倍数。两件事：
      1) 同一个 micro-batch 里长度接近，padding 浪费少；
      2) 序列长度稳定成少数几种（16 的倍数），MPS 分配器不会因形状抖动一直膨胀——
         实测动态长度训练时显存从 19.8G 涨到 22.6G、单步从 1.4s 恶化到 3.6s，分桶后稳定。
    """
    rng = random.Random(seed)
    idx = list(range(len(data)))
    jitter = {i: len(data[i]["input_ids"]) + rng.uniform(0, pad_mult) for i in idx}   # 同长度内顺序随机
    idx.sort(key=lambda i: jitter[i])
    group = batch_size * grad_accum * 8
    batches = []
    for g0 in range(0, len(idx), group):
        g = [data[i] for i in idx[g0:g0 + group]]
        T = -(-max(len(x["input_ids"]) for x in g) // pad_mult) * pad_mult
        for k in range(0, len(g), batch_size):
            batches.append((g[k:k + batch_size], T))
    return batches


def pick_device(name):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lm_loss(model, ids, att, lab):
    """
    只在「有标签的位置」算词表 logits 的交叉熵，数学上等价于 model(labels=...).loss。

    为什么不直接用 model(labels=lab)：它会对全部 B×T 个位置算 151,936 维 logits 并升到 fp32，
    最长分桶（8×208）时 logits+梯度要好几 GB，正式训练实测 MPS OOM（16GB 上限）。
    而 prompt 部分（~110 token）labels 全是 -100，根本不参与 loss —— 只取 assistant 回复的
    ~20 个位置过 lm_head，显存降一个数量级。
      h      : [B, T, H]       解码器最后一层隐状态（peft 的 LoRA 是原地注入的，这里已包含 LoRA）
      target : lab[:, 1:]       位置 t 的隐状态预测 t+1 的 token（与 HF 相同的移位）
      sel    : target != -100   → hs [N, H] → logits [N, V] → 对 N 个 token 求平均
    """
    base = model.get_base_model()
    h = base.model(input_ids=ids, attention_mask=att).last_hidden_state
    target = lab[:, 1:]
    sel = target != -100
    # 先转 fp32 再做布尔索引：MPS 上 bf16 的布尔索引反向（index_put）不被支持；这个中间张量只有几 MB
    hs = h[:, :-1].float()[sel].to(h.dtype)
    logits = base.lm_head(hs).float()
    return torch.nn.functional.cross_entropy(logits, target[sel])


@torch.no_grad()
def eval_loss(model, data, bs, pad_id, device):
    """
    val loss。注意：MPS 上「no_grad + sdpa」这条路径会让 mps_matmul 在 C++ 层直接崩溃
    （LLVM ERROR，进程 abort，没有 Python traceback），所以评测时切到 eager attention。
    """
    prev = model.config._attn_implementation
    model.set_attn_implementation("eager")
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(data), bs):
        items = data[i:i + bs]
        if len(items) == 1:                     # 单条 micro-batch 也会触发同一个内核 bug，复制一份
            items = items * 2                   # （eval_loss 按 token 数加权，复制不改变结果）
        ids, att, lab = collate(items, pad_id)
        out = model(input_ids=ids.to(device), attention_mask=att.to(device), labels=lab.to(device))
        k = int((lab[:, 1:] != -100).sum()) * (2 if len(data[i:i + bs]) == 1 else 1)
        tot += out.loss.item() * k
        n += k
    model.set_attn_implementation(prev)
    model.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=str(QWEN_DIR))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2, help="等效 batch = batch_size × grad_accum")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--max_len", type=int, default=320)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--val_size", type=int, default=200, help="每个 epoch 末在 val 前 N 条上算 loss")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="")
    ap.add_argument("--log_every", type=int, default=50, help="每多少个优化器步打印/记录一次")
    ap.add_argument("--run_name", default="llm_lora")
    ap.add_argument("--smoke", action="store_true", help="64 条训练、4 个优化器步")
    ap.add_argument("--frac", type=float, default=1.0, help="试运行：train/val 各随机取该比例（其余配置不变）")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.run_name, args.val_size, args.log_every = 1, args.run_name + "_smoke", 16, 1

    from peft import LoraConfig, get_peft_model

    ensure_dirs()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    train_rows = subsample(load_jsonl(DATA_DIR / "train.jsonl"), args.frac, args.seed)
    val_rows = subsample(load_jsonl(DATA_DIR / "val.jsonl"), args.frac, args.seed)[:args.val_size]
    if args.smoke:
        train_rows = train_rows[:64]
    train = [encode(tok, r, args.max_len) for r in train_rows]
    val = [encode(tok, r, args.max_len) for r in val_rows]
    lens = [len(x["input_ids"]) for x in train]
    n_trunc = sum(1 for r, x in zip(train_rows, train) if x["input_ids"][-1] != tok.convert_tokens_to_ids("<|im_end|>"))
    print(f"device={device} dtype={args.dtype}  train={len(train)} val={len(val)}  "
          f"token 长度 平均 {sum(lens) / len(lens):.1f} 最长 {max(lens)}  被截断 {n_trunc}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=dtype, attn_implementation="sdpa").to(device)
    model.config.use_cache = False
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"可训练参数 {n_trainable:,} / 总参数 {n_total:,} ({n_trainable / n_total:.3%})", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    micro_per_epoch = math.ceil(len(train) / args.batch_size)
    total = math.ceil(micro_per_epoch / args.grad_accum) * args.epochs
    if args.smoke:
        total = min(total, 4)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(total * args.warmup_ratio)), total)

    out_dir = CKPT_DIR / args.run_name
    log = {"run_name": args.run_name, "args": vars(args), "device": str(device),
           "n_trainable": n_trainable, "n_total": n_total, "n_train": len(train),
           "avg_tokens": round(sum(lens) / len(lens), 1), "n_truncated": n_trunc, "epochs": []}
    steps_f = open(LOG_DIR / f"{args.run_name}_steps.jsonl", "w", encoding="utf-8")

    v0 = eval_loss(model, val, args.batch_size, tok.pad_token_id, device)
    print(f"[ep0] val loss {v0:.4f}（训练前）", flush=True)
    log["val_loss_before"] = round(v0, 4)

    model.train()
    step, best, t0, done = 0, float("inf"), time.time(), False
    for ep in range(1, args.epochs + 1):
        batches = make_buckets(train, args.batch_size, args.grad_accum, seed=args.seed + ep)
        t_ep, run, n_run = time.time(), 0.0, 0
        for mi, (items, T) in enumerate(batches):
            ids, att, lab = collate(items, tok.pad_token_id, T)
            ids, att, lab = collate(items, tok.pad_token_id)
            loss = lm_loss(model, ids.to(device), att.to(device), lab.to(device))
            (loss / args.grad_accum).backward()
            run += loss.item(); n_run += 1
            if (mi + 1) % args.grad_accum == 0 or mi + 1 == micro_per_epoch:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                step += 1
                if step % args.log_every == 0:
                    el = time.time() - t0
                    rec = {"step": step, "epoch": ep, "loss": round(run / n_run, 4), "lr": sched.get_last_lr()[0],
                           "elapsed_s": round(el, 1)}
                    if device.type == "mps":
                        rec["mps_gib"] = round(torch.mps.driver_allocated_memory() / 2**30, 2)
                    steps_f.write(json.dumps(rec) + "\n"); steps_f.flush()
                    eta = el / step * (total - step)
                    mem = f"  显存 {rec['mps_gib']:.1f}G" if "mps_gib" in rec else ""
                    print(f"  ep{ep} step {step}/{total}  loss {rec['loss']:.4f}  "
                          f"{el / 60:.1f} min  ETA {eta / 60:.1f} min{mem}", flush=True)
                    run, n_run = 0.0, 0
                if step >= total:
                    done = True
                    break
        vl = eval_loss(model, val, args.batch_size, tok.pad_token_id, device)
        if device.type == "mps":
            torch.mps.empty_cache()          # 每轮结束把缓存还给系统，避免下一轮起点越来越高
        rec = {"epoch": ep, "val_loss": round(vl, 4), "epoch_time_s": round(time.time() - t_ep, 1)}
        log["epochs"].append(rec)
        saved = vl < best
        if saved:
            best = vl
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
        print(f"[ep{ep}] val loss {vl:.4f}  {rec['epoch_time_s'] / 60:.1f} min {'★saved' if saved else ''}", flush=True)
        if done:
            break
    steps_f.close()
    log["train_time_s"] = round(time.time() - t0, 1)
    log["best_val_loss"] = round(best, 4)
    if device.type == "mps":
        log["peak_mem_gib"] = round(torch.mps.driver_allocated_memory() / 2**30, 2)
    (LOG_DIR / f"{args.run_name}_train.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"训练完成 {log['train_time_s'] / 60:.1f} min，adapter → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
