"""
方法三：RoBERTa 微调（L2 主力）

基座：nlp_roberta_backbone_base_std（ModelScope 中文 RoBERTa-base，12 层 / 768 / 12 头 / 102.27M）。
  权重就在项目根目录，由 common/model_paths.roberta_path() 定位。
  注意：它的 state_dict 带 `bert.` 前缀 + Paddle 风格 LayerNorm.gamma/beta；
  transformers 5.x 会自动修正，4.x 不会（只打 warning 就随机初始化），
  所以 RobertaClassifier 里加了 missing_keys 校验，权重没对上直接抛错。

教学重点：
  1. 换基座只需要换**权重目录**这一处，架构只改两处——**池化方式**和**损失函数**，
     RoBERTa 主干 12 层不动。这就是「预训练模型可替换」的真正含义：
     下游脚本拼的不是模型代码，而是 checkpoint + config + tokenizer 三件套。
  2. 三个旋钮比换模型更值钱：分层学习率（主干 2e-5 / 头 ×5）、Warmup、类别加权。
     实测 9 组消融，单点提升都只有 1~2 个点，说明这个任务已贴近数据上限。
  3. 池化前一定要 mask 掉 [PAD]，否则 Mean/Max 会把补位统计进去（常见隐形 bug）。
  4. 再往上是四个「训练技巧」，本文件都实现了，但**这个任务上都不涨**（实测见下）：
       类别加权（默认开）  长尾类 recall 从「全错」回到「能中一部分」——唯一有业务意义的
       Label Smoothing    0.5272（−0.7）：15 类软标签把有限的信号又稀释了
       Focal Loss         0.5345（±0） ：难样本主要来自标签口径重叠，不是「学不会」
       FGM 对抗训练       0.5346（+0.02）：几乎不动，训练时间还 ×1.5
     → 结论：这几个旋钮只在「类别更不均衡（>100×）」「数据少到明显过拟合」时才值得开。

用法：
  python roberta/train_roberta.py                       # 默认 cls 池化，3 epoch
  python roberta/train_roberta.py --pool max --epochs 5
  python roberta/train_roberta.py --no_class_weight     # 关掉类别加权做对照
  python roberta/train_roberta.py --label_smoothing 0.1 # 软标签
  python roberta/train_roberta.py --focal_gamma 2       # Focal Loss（与类别加权共用权重）
  python roberta/train_roberta.py --fgm                 # FGM 对抗训练（训练时间约 ×1.5）
  python roberta/train_roberta.py --max_steps 20 --max_eval 500 --no_save   # CPU/冒烟：不落盘、只跑 20 步
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

from common.data import load_splits
from common.labels import LABELS
from common.model_paths import roberta_path

CODE_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = CODE_ROOT / "outputs"


class TnewsDataset(Dataset):
    """把 {"sentence","label"} 变成 RoBERTa 的输入张量"""

    def __init__(self, data, tok, max_length=64):
        self.data, self.tok, self.max_length = data, tok, max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]
        enc = self.tok(d["sentence"], truncation=True, max_length=self.max_length,
                       padding="max_length", return_tensors="pt")
        return {"input_ids": enc["input_ids"][0],
                "attention_mask": enc["attention_mask"][0],
                "labels": torch.tensor(d["label"])}


class RobertaClassifier(nn.Module):
    """
    RoBERTa 主干 + 池化 + 线性头。

    类名 RobertaClassifier：里面用的是 AutoModel（不依赖 BERT 特有的结构），
    换成任何 768 维的编码器（MacBERT / DeBERTa / ELECTRA）都只需要换 model_dir。
    这也是「换基座只改路径」能成立的原因——真正的耦合只有 hidden_size 一个数。

    pool 三选一：
      cls  —— 取 [CLS] 向量（预训练就为它设计，但不一定最适合短文本）
      mean —— 对所有有效 token 取均值（短文本常更稳）
      max  —— 逐维取最大（抓最显著特征，噪声也大）
    """

    def __init__(self, model_dir, num_labels=15, pool="cls", dropout=0.1):
        super().__init__()
        from transformers import AutoModel
        # output_loading_info=True：权重没对上时必须炸掉，不能只打 warning。
        # 这份 ModelScope RoBERTa 在 transformers 4.x 上就会「199 个 key 全部 missing」→ 随机初始化，
        # 训练 loss 照样下降、准确率停在 1/15≈0.07，属于最难查的一类 bug。
        self.backbone, info = AutoModel.from_pretrained(str(model_dir), output_loading_info=True)
        if info["missing_keys"]:
            raise RuntimeError(
                f"基座权重没有完整加载（会静默随机初始化，准确率掉到瞎猜水平）："
                f"{info['missing_keys'][:3]} … 共 {len(info['missing_keys'])} 个 key")
        if info["unexpected_keys"]:
            print(f"[load] 忽略 {len(info['unexpected_keys'])} 个多余 key（预训练头 cls.* / 根部 LayerNorm，正常）",
                  flush=True)
        self.pool = pool
        hidden = self.backbone.config.hidden_size          # RoBERTa-base = 768
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        # return_dict=True：transformers 5.x 里不传会拿到 tuple，容易下标取错
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hs = out.last_hidden_state                     # [B, L, 768]
        if self.pool == "cls":
            vec = hs[:, 0]
        else:
            m = attention_mask.unsqueeze(-1).float()   # [B, L, 1]
            if self.pool == "mean":
                vec = (hs * m).sum(1) / m.sum(1).clamp(min=1e-9)
            else:                                      # max：PAD 位置压到极小值再取 max
                vec = hs.masked_fill(m == 0, -1e9).max(1).values
        return self.classifier(self.dropout(vec))


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="cls", choices=["cls", "mean", "max"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5, help="主干学习率，越小越保守")
    ap.add_argument("--head_lr_mult", type=float, default=5.0, help="分类头 lr = 主干 lr × 该倍数")
    ap.add_argument("--max_length", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42, help="固定随机种子，保证对比实验可复现")
    ap.add_argument("--no_class_weight", action="store_true")
    # ↓ 四个训练技巧（实测在本任务上都不涨，保留是为了让学生自己复现/在别的任务上试）
    ap.add_argument("--label_smoothing", type=float, default=0.0, help="软标签 ε，常用 0.05~0.1")
    ap.add_argument("--focal_gamma", type=float, default=0.0, help="Focal Loss 的 γ，>0 才启用（与类别加权共用权重）")
    ap.add_argument("--fgm", action="store_true", help="打开 FGM 对抗训练（训练时间约 ×1.5）")
    ap.add_argument("--fgm_eps", type=float, default=1.0, help="FGM 扰动强度 ε")
    # ↓ 冒烟/调试用：不改变默认行为
    ap.add_argument("--max_steps", type=int, default=0, help="每个 epoch 最多跑多少步，0=全跑（冒烟用）")
    ap.add_argument("--max_eval", type=int, default=0, help="验证集最多评多少条，0=全评（冒烟用）")
    ap.add_argument("--no_save", action="store_true", help="不落盘 checkpoint（冒烟用，省 390 MB）")
    ap.add_argument("--out_dir", default="outputs",
                    help="产物目录（相对项目根）。做消融/多组实验时用它隔离 ckpt，避免互相覆盖")
    args = ap.parse_args()
    out_dir = Path(args.out_dir) if Path(args.out_dir).is_absolute() else CODE_ROOT / args.out_dir

    from transformers import AutoTokenizer

    # 对比实验的前提：同一份数据 + 同一个种子。原来完全没设 seed，
    # 两次跑差 1 个点分不清是「换了基座」还是「换了随机初始化」。
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_dir = roberta_path()
    print(f"[base] RoBERTa -> {base_dir}", flush=True)
    tok = AutoTokenizer.from_pretrained(str(base_dir))
    train, val = load_splits()
    if args.max_eval:
        val = val[: args.max_eval]

    train_dl = DataLoader(TnewsDataset(train, tok, args.max_length), batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(TnewsDataset(val, tok, args.max_length), batch_size=args.batch_size)

    model = RobertaClassifier(base_dir, num_labels=len(LABELS), pool=args.pool).to(device)

    # 分层学习率：头是随机初始化的，需要学得更快；主干要小步，否则冲掉预训练知识
    head_params = list(model.classifier.parameters()) + list(model.dropout.parameters())
    backbone_params = [p for n, p in model.named_parameters()
                       if not n.startswith("classifier") and not n.startswith("dropout")]
    opt = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr * args.head_lr_mult},
    ], weight_decay=0.01)

    # 类别加权：23× 不均衡，不加权时尾部类 recall 会趋近 0
    if args.no_class_weight:
        weight = None
    else:
        from sklearn.utils.class_weight import compute_class_weight
        cw = compute_class_weight("balanced", classes=np.arange(len(LABELS)), y=[d["label"] for d in train])
        weight = torch.tensor(cw, dtype=torch.float).to(device)

    # 损失函数：三个旋钮选一个（Focal 与 Label Smoothing 不叠加，Focal 优先）
    if args.focal_gamma > 0:
        if args.label_smoothing > 0:
            print("[warn] 同时给了 --focal_gamma 和 --label_smoothing，按 Focal 跑（两者都是在改目标分布）", flush=True)
        crit = lambda logits, y: focal_loss(logits, y, weight, args.focal_gamma)   # noqa: E731
        crit_desc = f"Focal(gamma={args.focal_gamma})"
    else:
        crit = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)
        crit_desc = "CE" + (f"+LS({args.label_smoothing})" if args.label_smoothing > 0 else "")

    steps_per_epoch = min(len(train_dl), args.max_steps) if args.max_steps else len(train_dl)
    total_steps = steps_per_epoch * args.epochs
    warmup = int(total_steps * 0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: step / max(1, warmup) if step < warmup else max(0.0, (total_steps - step) / max(1, total_steps - warmup)))
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))   # AMP：显存约降 40%，训练更快
    fgm = FGM(model, eps=args.fgm_eps) if args.fgm else None

    print(f"[cfg] pool={args.pool} loss={crit_desc} class_weight={'off' if weight is None else 'on'} "
          f"fgm={'on(eps=%s)' % args.fgm_eps if fgm else 'off'} epochs={args.epochs} seed={args.seed} "
          f"steps/epoch={steps_per_epoch} eval_n={len(val)} save={'off' if args.no_save else 'on'}", flush=True)

    best, best_epoch, log = 0.0, 0, []
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

        if res["macro_f1"] > best:
            best, best_epoch = res["macro_f1"], epoch
            # 只有要落盘时才写；--no_save 也要照常记录最佳分数（否则冒烟时打印 0）
            if not args.no_save:
                out_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": model.state_dict(), "pool": args.pool, "base": str(base_dir),
                            "val_acc": res["acc"], "val_macro_f1": res["macro_f1"],
                            "best_epoch": epoch, "args": vars(args)},
                           out_dir / "roberta_ckpt.pt")

    if not args.no_save:
        (out_dir / "roberta_train_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"最佳 val Macro-F1 = {best}（epoch {best_epoch}）→ {out_dir / 'roberta_ckpt.pt'}")
    else:
        print(f"最佳 val Macro-F1 = {best}（epoch {best_epoch}）；--no_save：本次不落盘（冒烟模式）")


if __name__ == "__main__":
    main()
