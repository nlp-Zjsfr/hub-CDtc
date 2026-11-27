"""
方法五：Qwen3-0.6B 指令微调（LoRA 与全量微调两种模式）

【这个脚本在干什么】
把新闻标题 → 类别名，做成「chat 格式的监督微调」样本，训一个 Qwen3-0.6B，
存成 LoRA adapter 或全量权重，供 eval_lora.py 评测。

教学重点：
  1. SFT 的关键不是模型，而是三件事：**chat 模板、Loss Masking、数据格式**。
  2. Loss Masking：prompt 部分 labels 全设 -100（不算 loss），只在答案 token 上算；
     不 mask 的话模型学的是「复述模板」，不是「答标签」。
  3. 两种微调模式的分工：
      LoRA（默认）     只训低秩旁路，2.29M / 596M = 0.38%，显存约 3 GB，adapter 9 MB
      全量微调 --full_ft  596M 参数全更新，显存 6~8 GB（8G 卡要开梯度检查点），权重 1.2 GB
    实测在这个 15 类任务上全量与 LoRA 的差距很小（都约 0.48–0.49），
    但成本差一个量级 —— 这就是「什么时候不值得全量微调」的答案。
  4. Qwen3 必须 enable_thinking=False，且训练与推理的模板必须完全一致。

用法：
  python llm_lora/train_lora.py                                   # LoRA，5000 条 × 2 epoch
  python llm_lora/train_lora.py --num_train 1000 --epochs 1        # 课堂快速版
  python llm_lora/train_lora.py --full_ft --num_train 3000 --epochs 2   # 全量微调
  python llm_lora/train_lora.py --max_steps 20                     # 冒烟测流程
"""

import argparse                                    # 命令行参数
import json                                       # 写训练日志
import sys                                        # 模块搜索路径
import time                                       # 计时 / 峰值显存统计
from pathlib import Path                          # 路径处理

import torch                                      # PyTorch
from torch.utils.data import DataLoader, Dataset  # 数据集与批加载

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ★把 code/ 加进 sys.path

from common.data import load_splits                # 读数据
from common.labels import LABELS                   # 15 类
from common.model_paths import qwen3_path          # 本地 Qwen3 目录

CODE_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = CODE_ROOT / "outputs"

SYSTEM = ("你是一个新闻标题分类助手。请将给定的新闻标题分类到以下类别之一，"
          "只输出类别名称，不要输出任何其他内容。\n可选类别：" + "、".join(LABELS))
# ↑ ★必须和推理端（eval_lora.py / classify_zero_shot.py）**逐字一致**，否则分布对不上


class SFTDataset(Dataset):
    """
    把分类样本转成 chat 格式的 SFT 样本。

    labels 的构造是本文件最核心的一行：
      prompt 部分 = -100（CrossEntropyLoss 会忽略 -100）
      答案部分   = 真实 token id（只有这里产生梯度）
    """

    def __init__(self, data, tok, max_length=128):
        self.data, self.tok, self.max_length = data, tok, max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]
        answer = LABELS[d["label"]]                 # 目标：类别名文本
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": d["sentence"]}]
        # tokenize=False 先拿文本：transformers 5.x 里 tokenize=True 返回 BatchEncoding，不好拼
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        # ↑ 生成 prompt 部分（含 assistant 起始标记）；enable_thinking=False 必须有
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]     # prompt token ids
        a_ids = self.tok(answer, add_special_tokens=False)["input_ids"] + [self.tok.eos_token_id]
        # ↑ 答案 token ids + EOS（告诉模型"说完了"）
        input_ids = (p_ids + a_ids)[: self.max_length]                      # 截断到定长
        labels = ([-100] * len(p_ids) + a_ids)[: self.max_length]
        # ★★★ Loss Masking：prompt 段全 -100（不算 loss），只有答案段算 loss
        return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels)}


def collate(batch, pad_id):
    """手写 collate：input_ids 用 pad 补齐，labels 用 -100 补齐（PAD 不产生 loss）"""
    L = max(len(b["input_ids"]) for b in batch)   # 本批最长长度
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)   # 补齐的 input_ids
    lab = torch.full((len(batch), L), -100, dtype=torch.long)      # 补齐的 labels（-100 = 不计 loss）
    att = torch.zeros((len(batch), L), dtype=torch.long)          # 补齐的 attention_mask（0 = 忽略）
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = b["input_ids"]
        lab[i, :n] = b["labels"]
        att[i, :n] = 1                      # 真实位置置 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_train", type=int, default=5000)          # 用多少条训练数据
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=4, help="全量微调建议 1~2")
    ap.add_argument("--grad_accum", type=int, default=4)            # 梯度累积步数（等效放大 batch）
    ap.add_argument("--lr", type=float, default=None, help="不填则 LoRA=2e-4、全量=2e-5")
    ap.add_argument("--lora_r", type=int, default=8)                # LoRA 秩
    ap.add_argument("--lora_alpha", type=int, default=16)          # LoRA 缩放系数
    ap.add_argument("--full_ft", action="store_true", help="全量微调；不加则 LoRA")
    ap.add_argument("--gradient_checkpointing", action="store_true", help="省显存（全量微调在 8G 卡上必开）")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--max_steps", type=int, default=0, help=">0 时只跑这么多步（冒烟测试）")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 全量微调：用 bf16 训练（fp16 + Adam 容易溢出）；LoRA：fp16 足够
    dtype = torch.bfloat16 if (args.full_ft and device == "cuda") else (
        torch.float16 if device == "cuda" else torch.float32)
    tok = AutoTokenizer.from_pretrained(str(qwen3_path()))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token        # Qwen3 默认没有 pad，用 eos 代替

    base = AutoModelForCausalLM.from_pretrained(str(qwen3_path()), dtype=dtype)  # 载基座

    if args.full_ft:
        model = base.to(device)                     # 所有参数默认 requires_grad=True
        if args.gradient_checkpointing:
            # 梯度检查点：用时间换显存，激活值不再常驻（全量微调在 8G 卡上基本必开）
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()      # 配合 checkpointing：让输入也需要梯度
        lr = args.lr or 2e-5                        # 全量更新必须用更小的 lr，否则冲掉预训练知识
        print(f"[full-ft] 参数量 {sum(p.numel() for p in model.parameters())/1e6:.1f}M（全部可训练），lr={lr}")
    else:
        from peft import LoraConfig, TaskType, get_peft_model
        cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
                         lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        # ↑ 只在注意力的 4 个投影上加低秩旁路（最省显存、效果也够）
        model = get_peft_model(base, cfg).to(device)   # 包装成 PEFT 模型（基座被冻结）
        lr = args.lr or 2e-4                          # LoRA 可以用大得多的 lr
        # 一定要打印：target_modules 写错名字不会报错，只会「训了等于没训」
        model.print_trainable_parameters()

    train, val = load_splits()
    train = train[: args.num_train]                # 只取前 N 条（课堂机器跑不满全量）
    val = val[: min(500, len(val))]                # 验证集最多 500 条
    train_dl = DataLoader(SFTDataset(train, tok, args.max_length), batch_size=args.batch_size,
                          shuffle=True, collate_fn=lambda b: collate(b, tok.pad_token_id))
    val_dl = DataLoader(SFTDataset(val, tok, args.max_length), batch_size=args.batch_size,
                        collate_fn=lambda b: collate(b, tok.pad_token_id))

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    # ↑ 只优化 requires_grad 的参数（LoRA 模式下就是那些 adapter）
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and dtype == torch.float16))
    log, step = [], 0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()                              # 训练模式
        running, n_batch = 0.0, 0
        for i, batch in enumerate(train_dl):
            batch = {k: v.to(device) for k, v in batch.items()}   # 搬到设备
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda" and dtype == torch.float16)):
                loss = model(**batch).loss / args.grad_accum     # 梯度累积：等效放大 batch
            scaler.scale(loss).backward()                         # 反传（梯度按 grad_accum 分摊）
            if (i + 1) % args.grad_accum == 0:                    # 累积够了才更新
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)                   # 清梯度
                step += 1
            running += loss.item() * args.grad_accum               # 还原成真实 loss 再累加
            n_batch += 1
            if i % 100 == 0:
                print(f"  epoch {epoch} {i}/{len(train_dl)} loss={loss.item()*args.grad_accum:.4f}", flush=True)
            if args.max_steps and step >= args.max_steps:
                break                                           # 冒烟：到步数就停

        model.eval()                               # 切推理模式
        vloss, vn = 0.0, 0
        with torch.no_grad():
            for batch in val_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                vloss += model(**batch).loss.item()   # 验证 loss
                vn += 1
        log.append({"epoch": epoch, "train_loss": round(running / max(1, n_batch), 4),
                    "val_loss": round(vloss / max(1, vn), 4), "steps": step,
                    "elapsed_s": round(time.time() - t_start, 1),
                    "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2) if device == "cuda" else None})
        # ↑ 峰值显存：写进日志才能说明「LoRA vs 全量」的成本差距
        print(f"epoch {epoch}: train_loss={log[-1]['train_loss']} val_loss={log[-1]['val_loss']}", flush=True)
        if args.max_steps:
            break

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.full_ft:
        # 全量微调存整个权重（约 1.2 GB）；交付件里不带，学生自己跑出来
        model.save_pretrained(str(OUT_DIR / "full_ft_ckpt"))
        log_path = OUT_DIR / "full_ft_train_log.json"
        print(f"全量权重 → {OUT_DIR / 'full_ft_ckpt'}")
    else:
        model.save_pretrained(str(OUT_DIR / "lora_adapter"))   # 只存 adapter，9 MB 左右
        log_path = OUT_DIR / "lora_train_log.json"
        print(f"adapter → {OUT_DIR / 'lora_adapter'}")
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
