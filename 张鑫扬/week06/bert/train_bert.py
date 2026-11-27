"""
方法三：ModelScope 中文 RoBERTa 微调 —— 中文文本领域分类（18 类）

任务与模型：
  - 预训练模型：ModelScope「RoBERTa预训练模型-中文-base」
      id: iic/nlp_roberta_backbone_base_std
      本地目录：common/pretrain_models/roberta-base-chinese
  - 训练数据：train_data/zh.test.csv（阿里达摩院中文文本领域分类测试集，18 个行业大类）
      原始数据只有一份测试集，common/domain_data.py 会按 8:1:1 分层切成
      train / val / test 并缓存到 data/domain/，本脚本在 train 上训练、val 上选模型。

教学重点：
  1. 架构只改两处——**池化方式**和**损失函数**，RoBERTa 本体 12 层不动。
  2. 三个旋钮比换模型更值钱：分层学习率（主体 2e-5 / 头 ×5）、Warmup、类别加权。
  3. 池化前一定要 mask 掉 [PAD]，否则 Mean/Max 会把补位统计进去（常见隐形 bug）。
  4. 四个「训练技巧」本文件都实现，默认关闭；在 18 类均衡数据上通常只作为消融项：
        Label Smoothing / Focal Loss / FGM 对抗训练 / 类别加权（默认开，权重≈1 无副作用）

用法：
  python bert/train_bert.py                       # 默认 cls 池化，3 epoch
  python bert/train_bert.py --pool mean --epochs 5
  python bert/train_bert.py --no_class_weight     # 关掉类别加权做对照
  python bert/train_bert.py --label_smoothing 0.1 # 软标签
  python bert/train_bert.py --focal_gamma 2       # Focal Loss（与类别加权共用权重）
  python bert/train_bert.py --fgm                 # FGM 对抗训练（训练时间约 ×1.5）
  python bert/train_bert.py --max_steps 20 --max_eval 500 --no_save   # 冒烟：不落盘、只跑 20 步
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.domain_data import DOMAIN_LABELS, load_domain_splits, sample_subset
from common.model_paths import roberta_path

CODE_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = CODE_ROOT / "outputs"
CKPT_NAME = "roberta_domain_ckpt.pt"
LOG_NAME = "roberta_domain_train_log.json"


class DomainDataset(Dataset):
    """
    把 {"text","label"} 变成 RoBERTa 的输入张量。

    为什么在 __init__ 里一次性 tokenize：
      CPU 训练时逐条 tokenize 会成为瓶颈（每个 epoch 都要重算一遍），
      预编码后每个 epoch 只做张量搬运，36k 条只占几十 MB。
    """

    def __init__(self, data, tok, max_length=64):
        self.data = data
        self.tok = tok
        self.max_length = max_length
        self._encode()

    def _encode(self):
        texts = [d["text"] for d in self.data]
        enc = self.tok(texts, truncation=True, max_length=self.max_length,
                       padding="max_length", return_tensors="pt")
        self.encodings = [{"input_ids": enc["input_ids"][i],
                           "attention_mask": enc["attention_mask"][i],
                           "labels": torch.tensor(self.data[i]["label"])}
                          for i in range(len(self.data))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.encodings[i]


class RobertaClassifier(nn.Module):
    """
    RoBERTa + 池化 + 线性头。

    pool 三选一：
      cls  —— 取 [CLS] 向量（预训练就为它设计，但不一定最适合短文本）
      mean —— 对所有有效 token 取均值（短文本常更稳）
      max  —— 逐维取最大（抓最显著特征，噪声也大）
    """

    def __init__(self, model_dir, num_labels=len(DOMAIN_LABELS), pool="cls", dropout=0.1):
        super().__init__()
        from transformers import AutoModel
        self.roberta = AutoModel.from_pretrained(str(model_dir))
        self.pool = pool
        hidden = self.roberta.config.hidden_size          # roberta-base = 768
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        # return_dict=True：transformers 5.x 里不传会拿到 tuple，容易下标取错
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hs = out.last_hidden_state                         # [B, L, 768]
        if self.pool == "cls":
            vec = hs[:, 0]
        else:
            m = attention_mask.unsqueeze(-1).float()       # [B, L, 1]
            if self.pool == "mean":
                vec = (hs * m).sum(1) / m.sum(1).clamp(min=1e-9)
            else:                                         # max：PAD 位置压到极小值再取 max
                vec = hs.masked_fill(m == 0, -1e9).max(1).values
        return self.classifier(self.dropout(vec))


# 兼容旧调用：service/ 里按 BertClassifier 的名字导入，行为与 RobertaClassifier 一致
BertClassifier = RobertaClassifier


class FGM:
    """
    FGM（Fast Gradient Method）对抗训练 —— 三步：
      ① 正常前向 + 反传，拿到参数的梯度 g；
      ② 在 **word_embeddings** 上沿梯度方向加一点扰动：r = eps * g / ||g||；
      ③ 用被扰动的权重再前向 + 反传一次，两次梯度一起更新，最后把权重还原。

    为什么只扰动 embedding：文本任务里「输入」就是 embedding，扰它等于在字的表示上加噪；
    扰动全部参数既慢又容易破坏预训练知识。为什么必须归一化：不除 ||g||，eps 的单位随梯度大小漂，
    同一个 eps 在不同 step 上的扰动强度完全不同。AMP 下用的是缩放后的梯度，方向不变，可以直接用。
    """

    def __init__(self, model, emb_name="word_embeddings", eps=1.0):
        self.model, self.emb_name, self.eps = model, emb_name, eps
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    param.data.add_(self.eps * param.grad / norm)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


def focal_loss(logits, targets, weight=None, gamma=2.0):
    """
    Focal Loss：把「已经学会的样本」的权重压下去，让梯度集中在难样本上。
        L = -alpha_t * (1 - p_t)^gamma * log(p_t)
    alpha 直接用类别权重（balanced）—— 和「类别加权」共用同一张表，不额外加超参。
    """
    logp = F.log_softmax(logits, dim=-1)
    logp_t = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
    p_t = logp_t.exp()
    alpha = weight[targets] if weight is not None else 1.0
    return (-alpha * ((1.0 - p_t) ** gamma) * logp_t).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    from sklearn.metrics import accuracy_score, f1_score
    model.eval()
    ys, ps, logits_all = [], [], []
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        ps.extend(logits.argmax(-1).cpu().tolist())
        ys.extend(batch["labels"].tolist())
        logits_all.append(logits.cpu())
    return {
        "acc": round(float(accuracy_score(ys, ps)), 4),
        "macro_f1": round(float(f1_score(ys, ps, average="macro")), 4),
        "logits": torch.cat(logits_all),
    }


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=None, help="默认自动定位 common/pretrain_models/roberta-base-chinese")
    ap.add_argument("--pool", default="cls", choices=["cls", "mean", "max"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5, help="RoBERTa 主体学习率，越小越保守")
    ap.add_argument("--head_lr_mult", type=float, default=5.0, help="分类头 lr = 主体 lr × 该倍数")
    ap.add_argument("--max_length", type=int, default=64)
    ap.add_argument("--no_class_weight", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    # ↓ 四个训练技巧（默认关闭，保留为消融项）
    ap.add_argument("--label_smoothing", type=float, default=0.0, help="软标签 ε，常用 0.05~0.1")
    ap.add_argument("--focal_gamma", type=float, default=0.0, help="Focal Loss 的 γ，>0 才启用（与类别加权共用权重）")
    ap.add_argument("--fgm", action="store_true", help="打开 FGM 对抗训练（训练时间约 ×1.5）")
    ap.add_argument("--fgm_eps", type=float, default=1.0, help="FGM 扰动强度 ε")
    # ↓ 冒烟/调试用：不改变默认行为
    ap.add_argument("--max_steps", type=int, default=0, help="每个 epoch 最多跑多少步，0=全跑（冒烟用）")
    ap.add_argument("--max_eval", type=int, default=0, help="验证集最多评多少条，0=全评（冒烟用）")
    ap.add_argument("--no_save", action="store_true", help="不落盘 checkpoint（冒烟用，省 400 MB）")
    ap.add_argument("--max_train", type=int, default=0, help="最多用多少条训练样本，0=全部（CPU 调试用）")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = Path(args.model_dir) if args.model_dir else roberta_path()
    tok = AutoTokenizer.from_pretrained(str(model_dir))

    train, val, _ = load_domain_splits()
    if args.max_train:
        train = sample_subset(train, args.max_train, args.seed)
    if args.max_eval:
        val = val[: args.max_eval]
    print(f"[data] train={len(train)} val={len(val)} labels={len(DOMAIN_LABELS)}", flush=True)

    train_dl = DataLoader(DomainDataset(train, tok, args.max_length),
                          batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(DomainDataset(val, tok, args.max_length),
                        batch_size=args.batch_size)

    model = RobertaClassifier(model_dir, num_labels=len(DOMAIN_LABELS), pool=args.pool).to(device)

    # 分层学习率：头是随机初始化的，需要学得更快；主体要小步，否则冲掉预训练知识
    head_params = list(model.classifier.parameters()) + list(model.dropout.parameters())
    base_params = [p for n, p in model.named_parameters()
                   if not n.startswith("classifier") and not n.startswith("dropout")]
    opt = torch.optim.AdamW([
        {"params": base_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr * args.head_lr_mult},
    ], weight_decay=0.01)

    # 类别加权：本数据集 18 类基本均衡，权重≈1；换成不均衡数据时这个开关就有意义
    if args.no_class_weight:
        weight = None
    else:
        from sklearn.utils.class_weight import compute_class_weight
        cw = compute_class_weight("balanced", classes=np.arange(len(DOMAIN_LABELS)),
                                  y=[d["label"] for d in train])
        weight = torch.tensor(cw, dtype=torch.float).to(device)

    # 损失函数：三个旋钮选一个（Focal 与 Label Smoothing 不叠加，Focal 优先）
    if args.focal_gamma > 0:
        if args.label_smoothing > 0:
            print("[warn] 同时给了 --focal_gamma 和 --label_smoothing，按 Focal 跑", flush=True)
        crit = lambda logits, y: focal_loss(logits, y, weight, args.focal_gamma)   # noqa: E731
        crit_desc = f"Focal(gamma={args.focal_gamma})"
    else:
        crit = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)
        crit_desc = "CE" + (f"+LS({args.label_smoothing})" if args.label_smoothing > 0 else "")

    steps_per_epoch = min(len(train_dl), args.max_steps) if args.max_steps else len(train_dl)
    total_steps = steps_per_epoch * args.epochs
    warmup = int(total_steps * 0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: step / max(1, warmup) if step < warmup
        else max(0.0, (total_steps - step) / max(1, total_steps - warmup)))
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))   # AMP：显存约降 40%，训练更快
    fgm = FGM(model, eps=args.fgm_eps) if args.fgm else None

    print(f"[cfg] model={model_dir.name} pool={args.pool} loss={crit_desc} "
          f"class_weight={'off' if weight is None else 'on'} "
          f"fgm={'on(eps=%s)' % args.fgm_eps if fgm else 'off'} epochs={args.epochs} "
          f"steps/epoch={steps_per_epoch} eval_n={len(val)} device={device} "
          f"save={'off' if args.no_save else 'on'}", flush=True)

    best, log = 0.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running, n_step = time.time(), 0.0, 0
        for i, batch in enumerate(train_dl):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)

            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                loss = crit(model(ids, mask), y)
            scaler.scale(loss).backward()

            if fgm is not None:
                # 对抗训练：先按梯度扰动 embedding，再前向一次、反传一次（两次梯度累加后一起更新）
                fgm.attack()
                with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                    loss_adv = crit(model(ids, mask), y)
                scaler.scale(loss_adv).backward()
                fgm.restore()

            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            n_step += 1
            if i % 200 == 0:
                print(f"  epoch {epoch} step {i}/{steps_per_epoch} loss={loss.item():.4f}", flush=True)
            if args.max_steps and n_step >= args.max_steps:
                break

        res = evaluate(model, val_dl, device)
        log.append({"epoch": epoch, "train_loss": round(running / max(1, n_step), 4),
                    "val_acc": res["acc"], "val_macro_f1": res["macro_f1"],
                    "elapsed_s": round(time.time() - t0, 1)})
        print(f"epoch {epoch}: val_acc={res['acc']} val_f1={res['macro_f1']}", flush=True)

        if res["macro_f1"] > best and not args.no_save:
            best = res["macro_f1"]
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "pool": args.pool,
                        "labels": DOMAIN_LABELS, "base_model": model_dir.name,
                        "model_dir": str(model_dir),
                        "val_acc": res["acc"], "val_macro_f1": res["macro_f1"],
                        "args": vars(args)}, OUT_DIR / CKPT_NAME)

    if not args.no_save:
        (OUT_DIR / LOG_NAME).write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"最佳 val Macro-F1 = {best} → {OUT_DIR / CKPT_NAME}")
    else:
        print("--no_save：本次不落盘（冒烟模式），ckpt 与日志都没写")


if __name__ == "__main__":
    main()
