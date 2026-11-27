import argparse
import hashlib
import itertools
import math
import platform
import random
import time
from contextlib import nullcontext

import peft
from peft import get_peft_model_state_dict, set_peft_model_state_dict
import torch
import transformers
from torch.utils.data import DataLoader

from common import ROOT, config, read_jsonl, write_json, sha256, load_tokenizer
from modeling import trainable_model
from sft_data import SFTDataset, SFTCollator, answer_loss_sum, save_encoding_audit


def autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


@torch.inference_mode()
def score(model, loader, device):
    model.eval()
    loss_sum, tokens, correct = 0.0, 0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast(device):
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
            loss, n = answer_loss_sum(logits, batch["labels"])
        loss_sum += float(loss)
        tokens += n
        labels = batch["labels"][:, 1:]
        correct += int((logits[:, :-1].argmax(-1).eq(labels) & labels.ne(-100)).sum())
    return {"answer_nll": loss_sum / tokens, "answer_ppl": math.exp(loss_sum / tokens),
            "answer_tokens": tokens, "teacher_forced_token_accuracy": correct / tokens}


def frozen_fingerprint(model):
    # 这是冻结检查的补充抽查；冻结性由 requires_grad 和优化器参数列表共同保证。
    h = hashlib.sha256()
    selected = [p for n, p in model.named_parameters() if not p.requires_grad][:5]
    for p in selected:
        h.update(p.detach().flatten()[:2048].float().cpu().numpy().tobytes())
    return h.hexdigest()


def train(resume=False, max_epochs=None):
    cfg = config()
    if max_epochs is not None:
        cfg["epochs"] = max_epochs
    if resume:
        completed = torch.load(ROOT / "checkpoints" / "last.pt", map_location="cpu", weights_only=True)
        assert completed["config"] == cfg, "断点配置与当前配置不一致"
        assert completed["train_data_sha256"] == sha256(ROOT / "data" / "train.jsonl")
        assert completed["val_data_sha256"] == sha256(ROOT / "data" / "val.jsonl")
        assert completed["template_sha256"] == sha256(ROOT / "chat_template.jinja")
        if completed["epoch"] >= cfg["epochs"]:
            print("Checkpoint already completed the configured epochs; no files or weights changed.", flush=True)
            return
        del completed
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("本配置要求 CUDA BF16；可在 CPU 上运行 FP32 小规模检查")
        torch.cuda.manual_seed_all(cfg["seed"])
        torch.cuda.reset_peak_memory_stats()
    tokenizer = load_tokenizer(cfg)
    ds = {s: SFTDataset(read_jsonl(ROOT / "data" / f"{s}.jsonl"), tokenizer, cfg["max_length"]) for s in ("train", "val")}
    save_encoding_audit(ds["train"], tokenizer)
    collator = SFTCollator(tokenizer.pad_token_id)
    generator = torch.Generator().manual_seed(cfg["seed"])
    loaders = {s: DataLoader(ds[s], batch_size=cfg["micro_batch_size"], shuffle=s == "train",
                             generator=generator if s == "train" else None, collate_fn=collator, num_workers=0) for s in ds}
    model = trainable_model(cfg, device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg["learning_rate"], betas=(.9, .999), eps=1e-8, weight_decay=cfg["weight_decay"])
    updates_per_epoch = math.ceil(len(loaders["train"]) / cfg["gradient_accumulation_steps"])
    total_steps = updates_per_epoch * cfg["epochs"]
    warmup = max(1, math.ceil(total_steps * cfg["warmup_ratio"]))
    checkpoint_path = ROOT / "checkpoints" / "last.pt"
    history, step_history, step, start_epoch, best, stale = [], [], 0, 1, float("inf"), 0
    initial_val = score(model, loaders["val"], device)
    fingerprint = frozen_fingerprint(model)
    if resume:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        assert ckpt["config"] == cfg, "续训必须使用相同配置，不能静默改变学习率计划"
        assert ckpt["train_data_sha256"] == sha256(ROOT / "data" / "train.jsonl")
        assert ckpt["val_data_sha256"] == sha256(ROOT / "data" / "val.jsonl")
        assert ckpt["template_sha256"] == sha256(ROOT / "chat_template.jinja")
        set_peft_model_state_dict(model, ckpt["adapter"])
        optimizer.load_state_dict(ckpt["optimizer"])
        history, step_history = ckpt["history"], ckpt["step_history"]
        step, start_epoch, best, stale = ckpt["step"], ckpt["epoch"] + 1, ckpt["best"], ckpt["stale"]
        generator.set_state(ckpt["loader_rng"])
        torch.set_rng_state(ckpt["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
        print("Resuming after epoch", ckpt["epoch"], flush=True)
    trainable = sum(p.numel() for p in parameters)
    total = sum(p.numel() for p in model.parameters())
    print(f"Device={device}; trainable={trainable:,}/{total:,} ({trainable/total:.3%}); initial val NLL={initial_val['answer_nll']:.4f}", flush=True)
    t0 = time.perf_counter()
    best_dir = ROOT / "models" / "best_adapter"
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        model.train()
        iterator = iter(loaders["train"])
        epoch_loss, epoch_tokens, epoch_start = 0.0, 0, time.perf_counter()
        while True:
            group = list(itertools.islice(iterator, cfg["gradient_accumulation_steps"]))
            if not group:
                break
            # 按整个累积窗口的有效回答 token 数归一化，处理变长回答及不足一组的尾批。
            window_tokens = sum(int(b["labels"][:, 1:].ne(-100).sum()) for b in group)
            step += 1
            if step <= warmup:
                lr = cfg["learning_rate"] * step / warmup
            else:
                progress = (step - warmup) / max(1, total_steps - warmup)
                lr = cfg["min_learning_rate"] + .5 * (cfg["learning_rate"] - cfg["min_learning_rate"]) * (1 + math.cos(math.pi * progress))
            for g in optimizer.param_groups:
                g["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            window_loss = 0.0
            for batch in group:
                batch = {k: v.to(device) for k, v in batch.items()}
                with autocast(device):
                    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
                    summed, count = answer_loss_sum(logits, batch["labels"])
                    loss = summed / window_tokens
                if not torch.isfinite(loss):
                    raise RuntimeError("非有限损失，停止训练，不覆盖最佳模型")
                loss.backward()
                window_loss += float(summed.detach())
                del logits, summed, loss
            norm = torch.nn.utils.clip_grad_norm_(parameters, cfg["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            epoch_loss += window_loss
            epoch_tokens += window_tokens
            step_history.append({"step": step, "epoch": epoch, "answer_nll": window_loss/window_tokens,
                                 "answer_tokens": window_tokens, "micro_batches": len(group), "lr": lr, "grad_norm_before_clip": float(norm)})
            if step % 10 == 0:
                print(f"epoch={epoch} update={step}/{total_steps} answer_nll={window_loss/window_tokens:.4f} lr={lr:.7f}", flush=True)
        val = score(model, loaders["val"], device)
        improved = val["answer_nll"] < best
        if improved:
            best, stale = val["answer_nll"], 0
            model.save_pretrained(best_dir, safe_serialization=True)
            (best_dir / "README.md").write_text("# 售后工单 SFT 适配器\n\n基座：Qwen/Qwen3-0.6B-Base。完整训练与加载说明见 [项目 README](../../README.md)。\n", encoding="utf-8")
        else:
            stale += 1
        history.append({"epoch": epoch, "train_answer_nll": epoch_loss/epoch_tokens, "train_answer_tokens": epoch_tokens,
                        "val": val, "seconds": time.perf_counter() - epoch_start, "best": improved})
        write_json(ROOT / "results" / "history.json", history)
        write_json(ROOT / "results" / "steps.json", step_history)
        assert frozen_fingerprint(model) == fingerprint, "抽查的基座参数发生变化"
        checkpoint_path.parent.mkdir(exist_ok=True)
        torch.save({"config": cfg, "epoch": epoch, "step": step, "best": best, "stale": stale,
                    "adapter": {n: p.detach().cpu().clone() for n,p in get_peft_model_state_dict(model).items()},
                    "optimizer": optimizer.state_dict(), "history": history, "step_history": step_history,
                    "loader_rng": generator.get_state(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                    "train_data_sha256": sha256(ROOT / "data" / "train.jsonl"),
                    "val_data_sha256": sha256(ROOT / "data" / "val.jsonl"),
                    "template_sha256": sha256(ROOT / "chat_template.jinja")}, checkpoint_path)
        print(f"Epoch {epoch}: train NLL={epoch_loss/epoch_tokens:.4f} val NLL={val['answer_nll']:.4f} PPL={val['answer_ppl']:.3f} {history[-1]['seconds']:.1f}s{' *' if improved else ''}", flush=True)
        if stale >= cfg["patience"]:
            break
    best_epoch = min(history, key=lambda r: r["val"]["answer_nll"])["epoch"]
    summary = {"config": cfg, "base_parameters": total-trainable, "trainable_parameters": trainable,
               "trainable_fraction": trainable/total, "initial_val": initial_val, "best_epoch": best_epoch,
               "epochs_run": len(history), "optimizer_updates": step, "training_seconds_this_run": time.perf_counter()-t0,
               "resumed": resume, "frozen_base_fingerprint_unchanged": frozen_fingerprint(model)==fingerprint,
               "test_used_for_selection": False, "best_adapter_sha256": sha256(best_dir / "adapter_model.safetensors"),
               "template_sha256": sha256(ROOT / "chat_template.jinja"),
               "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "transformers": transformers.__version__, "peft": peft.__version__, "device": str(device),
                    "gpu": torch.cuda.get_device_name(0) if device.type=="cuda" else None,
                    "peak_allocated_GB": torch.cuda.max_memory_allocated()/1024**3 if device.type=="cuda" else None,
                    "precision": "BF16 base/autocast; FP32 trainable LoRA and optimizer"}}
    write_json(ROOT / "results" / "training.json", summary)
    print("Training complete; best epoch", best_epoch, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-epochs", type=int)
    args = parser.parse_args()
    train(args.resume, args.max_epochs)
