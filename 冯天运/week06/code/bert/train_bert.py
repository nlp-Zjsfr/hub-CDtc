"""
方法三：BERT 微调（L2 主力）

【这个脚本在干什么】
把预训练 BERT（bert-base-chinese 或 chinese-roberta-wwm-ext）接一个分类头，
在本任务的小样本（万级）上微调，输出 15 类预测 + checkpoint。

教学重点：
  1. 架构只改两处——**池化方式**和**损失函数**，BERT 本体 12 层不动。
  2. 三个旋钮比换模型更值钱：分层学习率（BERT 2e-5 / 头 ×5）、Warmup、类别加权。
     实测 9 组消融，单点提升都只有 1~2 个点，说明这个任务已贴近数据上限。
  3. 池化前一定要 mask 掉 [PAD]，否则 Mean/Max 会把补位统计进去（常见隐形 bug）。
  4. 再往上是四个「训练技巧」，本文件都实现了，但**这个任务上都不涨**（实测见下）：
      类别加权（默认开）  长尾类 recall 从「全错」回到「能中一部分」——唯一有业务意义的
      Label Smoothing    0.5272（−0.7）：15 类软标签把有限的信号又稀释了
      Focal Loss         0.5345（±0） ：难样本主要来自标签口径重叠，不是「学不会」
      FGM 对抗训练       0.5346（+0.02）：几乎不动，训练时间还 ×1.5
    → 结论：这几个旋钮只在「类别更不均衡（>100×）」「数据少到明显过拟合」时才值得开。

【动手顺序建议】
  ① 先 --max_steps 20 --max_eval 500 --no_save 跑通冒烟（几分钟内看到 loss 下降）
  ② 再 --pool mean / max 各跑一次，对比 val Macro-F1
  ③ 再 --no_class_weight 看长尾类 recall 掉多少
  ④ 最后才做换模型对照（换模型时其余参数必须完全不变，见 换模型对照实验.py）

用法：
  python bert/train_bert.py                       # 默认 cls 池化，3 epoch
  python bert/train_bert.py --pool max --epochs 5
  python bert/train_bert.py --no_class_weight     # 关掉类别加权做对照
  python bert/train_bert.py --label_smoothing 0.1 # 软标签
  python bert/train_bert.py --focal_gamma 2       # Focal Loss（与类别加权共用权重）
  python bert/train_bert.py --fgm                 # FGM 对抗训练（训练时间约 ×1.5）
  python bert/train_bert.py --max_steps 20 --max_eval 500 --no_save   # CPU/冒烟：不落盘、只跑 20 步
"""
import argparse                                    # 命令行参数
import json                                        # 写训练日志 JSON
import sys                                         # 需要把 code/ 加入模块搜索路径
import time                                        # 统计每个 epoch 耗时
from pathlib import Path                          # 路径处理

import torch                                       # PyTorch 主包
import torch.nn as nn                              # 神经网络模块（Linear / Dropout / Loss）
import torch.nn.functional as F                    # 函数式 API（log_softmax 等）
from torch.utils.data import DataLoader, Dataset    # 数据集封装与批加载器

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ★把 code/ 目录插到 sys.path 最前面，这样下面 from common.data import ... 才能找到
#   （脚本在 code/bert/ 下，而 common/ 在 code/ 下，所以要往上退一级）

from common.data import load_splits                # 读训练集/验证集
from common.labels import LABELS                   # 15 个类名（标签 id 顺序）
from common.model_paths import bert_path          # 按环境变量 WEEK6_BERT_DIR 决定用哪个预训练模型

CODE_ROOT = Path(__file__).resolve().parent.parent   # code/ 根目录
OUT_DIR = CODE_ROOT / "outputs"                     # 产物目录（权重、日志都写这里）


class TnewsDataset(Dataset):
    """把 {"sentence","label"} 变成 BERT 的输入张量"""

    def __init__(self, data, tok, max_length=64):
        # data: [{"sentence":..., "label":...}, ...]
        # tok : 分词器（AutoTokenizer）
        # max_length: 截断/补齐的最大长度（新闻标题很短，64 足够）
        self.data, self.tok, self.max_length = data, tok, max_length

    def __len__(self):
        return len(self.data)          # 样本数，DataLoader 靠它算 batch 数

    def __getitem__(self, i):
        d = self.data[i]               # 取第 i 条
        # 分词：超长截断（truncation=True）、补齐到定长（padding="max_length"）、
        #       直接返回 torch 张量（return_tensors="pt"）
        enc = self.tok(d["sentence"], truncation=True, max_length=self.max_length,
                       padding="max_length", return_tensors="pt")
        # enc["input_ids"] 形状 [1, L]：1 是 batch 维（这里单条），L 是长度
        # [0] 去掉 batch 维 → [L]
        return {"input_ids": enc["input_ids"][0],
                "attention_mask": enc["attention_mask"][0],   # 1=真实 token，0=[PAD]
                "labels": torch.tensor(d["label"])}          # 标签转张量


class BertClassifier(nn.Module):
    """
    BERT + 池化 + 线性头。

    pool 三选一：
      cls  —— 取 [CLS] 向量（预训练就为它设计，但不一定最适合短文本）
      mean —— 对所有有效 token 取均值（短文本常更稳）
      max  —— 逐维取最大（抓最显著特征，噪声也大）
    """

    def __init__(self, model_dir, num_labels=15, pool="cls", dropout=0.1):
        super().__init__()
        from transformers import AutoModel       # 延迟导入：只在真正建模型时才加载 transformers
        self.bert = AutoModel.from_pretrained(str(model_dir))  # 加载预训练权重（BERT 本体，12 层）
        self.pool = pool                        # 池化方式：cls / mean / max
        hidden = self.bert.config.hidden_size   # bert-base = 768
        self.dropout = nn.Dropout(dropout)      # 分类头前的 dropout（只作用在头的输入上）
        self.classifier = nn.Linear(hidden, num_labels)  # 768 → 15 的线性分类头（随机初始化）

    def forward(self, input_ids, attention_mask):
        # return_dict=True：transformers 5.x 里不传会拿到 tuple，容易下标取错
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hs = out.last_hidden_state                # [B, L, 768] 每个 token 的隐向量
        if self.pool == "cls":
            vec = hs[:, 0]                      # 取 [CLS]（第 0 个 token）→ [B, 768]
        else:
            m = attention_mask.unsqueeze(-1).float()   # [B, L, 1]：把 mask 扩成可广播的形状
            if self.pool == "mean":
                # ★关键：先乘 mask 把 [PAD] 位置清零，再除以有效 token 数 → 不会被补位拉偏
                vec = (hs * m).sum(1) / m.sum(1).clamp(min=1e-9)
            else:                                # max：PAD 位置压到极小值再取 max
                vec = hs.masked_fill(m == 0, -1e9).max(1).values
        return self.classifier(self.dropout(vec))   # dropout → 线性头 → 15 类 logits


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
        self.backup = {}      # 备份被扰动前的参数（attack 时存，restore 时还原）

    def attack(self):
        """第①+②步：先拿到梯度，再沿归一化梯度方向扰动 embedding"""
        for name, param in self.model.named_parameters():
            # 只动 embedding 层、且必须有梯度
            if param.requires_grad and self.emb_name in name and param.grad is not None:
                self.backup[name] = param.data.clone()      # 备份原始权重（.clone() 必须有，否则是同一个存储）
                norm = torch.norm(param.grad)              # 梯度范数
                if norm != 0 and not torch.isnan(norm):     # 避免除以 0 / NaN
                    param.data.add_(self.eps * param.grad / norm)  # 原地加扰动（不参与自动求导）

    def restore(self):
        """第③步末尾：把 embedding 还原回没被扰动的样子"""
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]   # 直接换回备份张量
        self.backup = {}                         # 清空备份


def focal_loss(logits, targets, weight=None, gamma=2.0):
    """
    Focal Loss：把「已经学会的样本」的权重压下去，让梯度集中在难样本上。
        L = -alpha_t * (1 - p_t)^gamma * log(p_t)
    alpha 直接用类别权重（balanced）—— 和「类别加权」共用同一张表，不额外加超参。
    """
    logp = F.log_softmax(logits, dim=-1)                    # 对数概率（数值稳定，优于 log(softmax)）
    logp_t = logp.gather(1, targets.unsqueeze(1)).squeeze(1)  # 取每个样本「真实类别」的那个对数概率
    p_t = logp_t.exp()                                       # 转回概率 p_t
    alpha = weight[targets] if weight is not None else 1.0    # 每样本的类别权重（长尾类权重更大）
    return (-alpha * ((1.0 - p_t) ** gamma) * logp_t).mean()   # focal 因子 (1-p)^gamma：越容易权重越小


@torch.no_grad()                                              # 评测不建计算图，省显存/时间
def evaluate(model, loader, device):
    from sklearn.metrics import accuracy_score, f1_score
    model.eval()                                              # 切到推理模式：关闭 dropout
    ys, ps, logits_all = [], [], []                          # 收集真值、预测、logits
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        ps.extend(logits.argmax(-1).cpu().tolist())           # 最大 logit 的类别 = 预测
        ys.extend(batch["labels"].tolist())                  # 真值
        logits_all.append(logits.cpu())                      # logits 留着做温度缩放/校准分析
    return {
        "acc": round(float(accuracy_score(ys, ps)), 4),
        "macro_f1": round(float(f1_score(ys, ps, average="macro")), 4),
        "logits": torch.cat(logits_all),      # [N, 15]
    }


def main():
    ap = argparse.ArgumentParser()
    # ---- 架构与训练超参
    ap.add_argument("--pool", default="cls", choices=["cls", "mean", "max"])  # 池化方式
    ap.add_argument("--epochs", type=int, default=3)              # 训练轮数（小样本 3 轮足够，再多过拟合）
    ap.add_argument("--batch_size", type=int, default=64)        # 批大小（CPU 上 64 偏大，可调 16/32 提速）
    ap.add_argument("--lr", type=float, default=2e-5, help="BERT 层学习率，越小越保守")
    ap.add_argument("--head_lr_mult", type=float, default=5.0, help="分类头 lr = BERT lr × 该倍数")
    ap.add_argument("--max_length", type=int, default=64)        # 文本最大长度（标题很短）
    ap.add_argument("--no_class_weight", action="store_true")    # 关闭类别加权（做对照用）
    # ↓ 四个训练技巧（实测在本任务上都不涨，保留是为了让学生自己复现/在别的任务上试）
    ap.add_argument("--label_smoothing", type=float, default=0.0, help="软标签 ε，常用 0.05~0.1")
    ap.add_argument("--focal_gamma", type=float, default=0.0, help="Focal Loss 的 γ，>0 才启用（与类别加权共用权重）")
    ap.add_argument("--fgm", action="store_true", help="打开 FGM 对抗训练（训练时间约 ×1.5）")
    ap.add_argument("--fgm_eps", type=float, default=1.0, help="FGM 扰动强度 ε")
    # ↓ 冒烟/调试用：不改变默认行为
    ap.add_argument("--max_steps", type=int, default=0, help="每个 epoch 最多跑多少步，0=全跑（冒烟用）")
    ap.add_argument("--max_eval", type=int, default=0, help="验证集最多评多少条，0=全评（冒烟用）")
    ap.add_argument("--no_save", action="store_true", help="不落盘 checkpoint（冒烟用，省 390 MB）")
    ap.add_argument("--ckpt_out", default=None,
                    help="checkpoint 落盘文件名（写到 code/outputs/ 下）。默认 bert_ckpt.pt；"
                         "做模型对照实验时用它给每个模型单独命名，避免互相覆盖")
    ap.add_argument("--log_every", type=int, default=200,
                    help="每多少步打印一次 loss（默认 200，与原行为一致；调小可看进度）")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"    # 有 GPU 用 GPU
    tok = AutoTokenizer.from_pretrained(str(bert_path()))      # 分词器必须和预训练模型同源
    train, val = load_splits()                                 # 读原始训练集 / 验证集
    if args.max_eval:
        val = val[: args.max_eval]                             # 冒烟：只评前 N 条

    train_dl = DataLoader(TnewsDataset(train, tok, args.max_length), batch_size=args.batch_size, shuffle=True)
    # ↑ 训练集要 shuffle=True（打乱顺序，否则梯度有偏）
    val_dl = DataLoader(TnewsDataset(val, tok, args.max_length), batch_size=args.batch_size)
    # ↑ 验证集不 shuffle（评测顺序固定，结果可复现）

    model = BertClassifier(bert_path(), num_labels=len(LABELS), pool=args.pool).to(device)

    # 分层学习率：头是随机初始化的，需要学得更快；BERT 要小步，否则冲掉预训练知识
    head_params = list(model.classifier.parameters()) + list(model.dropout.parameters())
    bert_params = [p for n, p in model.named_parameters() if not n.startswith("classifier") and not n.startswith("dropout")]
    opt = torch.optim.AdamW([
        {"params": bert_params, "lr": args.lr},                          # BERT 本体：2e-5（保守）
        {"params": head_params, "lr": args.lr * args.head_lr_mult},      # 分类头：2e-5×5 = 1e-4（快）
    ], weight_decay=0.01)   # 权重衰减（正则）

    # 类别加权：23× 不均衡，不加权时尾部类 recall 会趋近 0
    if args.no_class_weight:
        weight = None
    else:
        from sklearn.utils.class_weight import compute_class_weight
        import numpy as np
        # "balanced"：每个类的权重 ∝ 1/该类样本数 → 自动平衡 15 类
        cw = compute_class_weight("balanced", classes=np.arange(len(LABELS)), y=[d["label"] for d in train])
        weight = torch.tensor(cw, dtype=torch.float).to(device)   # 放到设备上，供 loss 直接索引

    # 损失函数：三个旋钮选一个（Focal 与 Label Smoothing 不叠加，Focal 优先）
    if args.focal_gamma > 0:
        if args.label_smoothing > 0:
            print("[warn] 同时给了 --focal_gamma 和 --label_smoothing，按 Focal 跑（两者都是在改目标分布）", flush=True)
        crit = lambda logits, y: focal_loss(logits, y, weight, args.focal_gamma)   # noqa: E731
        crit_desc = f"Focal(gamma={args.focal_gamma})"
    else:
        crit = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)
        # label_smoothing=0 就是普通 CE；>0 时把 one-hot 目标软化，抑制过度自信
        crit_desc = "CE" + (f"+LS({args.label_smoothing})" if args.label_smoothing > 0 else "")

    steps_per_epoch = min(len(train_dl), args.max_steps) if args.max_steps else len(train_dl)
    total_steps = steps_per_epoch * args.epochs            # 总步数（调度器要靠它）
    warmup = int(total_steps * 0.1)                        # 前 10% 步做 warmup（学习率从 0 线性升上来）
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: step / max(1, warmup) if step < warmup else max(0.0, (total_steps - step) / max(1, total_steps - warmup)))
    # ↑ 学习率曲线：warmup 段线性升，之后线性降到 0
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))   # AMP：显存约降 40%，训练更快
    fgm = FGM(model, eps=args.fgm_eps) if args.fgm else None   # 不开 FGM 就是 None

    print(f"[cfg] pool={args.pool} loss={crit_desc} class_weight={'off' if weight is None else 'on'} "
          f"fgm={'on(eps=%s)' % args.fgm_eps if fgm else 'off'} epochs={args.epochs} "
          f"steps/epoch={steps_per_epoch} eval_n={len(val)} save={'off' if args.no_save else 'on'}", flush=True)
    # ↑ 打印本次实验的全部设置 —— 做对照实验时，这行日志就是「控制变量」的证据

    best, log = 0.0, []     # 最好的 val Macro-F1 和训练日志
    for epoch in range(1, args.epochs + 1):
        model.train()                                  # 切训练模式：dropout 生效
        t0, running, n_step = time.time(), 0.0, 0      # 计时器、loss 累加、步数计数
        for i, batch in enumerate(train_dl):
            ids = batch["input_ids"].to(device)        # 搬到设备
            mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)

            opt.zero_grad(set_to_none=True)             # 清空上一轮梯度（set_to_none=True 省显存）
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                loss = crit(model(ids, mask), y)       # 前向 + 算损失（AMP 自动混合精度）
            scaler.scale(loss).backward()               # 反传（loss 先放大，避免 fp16 下溢）

            if fgm is not None:
                # 对抗训练：先按梯度扰动 embedding，再前向一次、反传一次（两次梯度累加后一起更新）
                fgm.attack()                           # ① 扰动 embedding
                with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                    loss_adv = crit(model(ids, mask), y)   # ② 扰动后前向
                scaler.scale(loss_adv).backward()        # ③ 第二次反传，梯度累加
                fgm.restore()                           # ④ 还原 embedding

            scaler.step(opt)      # 更新参数（GradScaler 会自动 unscale 梯度）
            scaler.update()       # 更新缩放因子
            sched.step()          # 推进学习率调度（每步都要调）
            running += loss.item()  # 累加 loss 用于算 epoch 平均
            n_step += 1
            if i % max(1, args.log_every) == 0:
                print(f"  epoch {epoch} step {i}/{steps_per_epoch} loss={loss.item():.4f}", flush=True)
            if args.max_steps and n_step >= args.max_steps:
                break                                     # 冒烟模式：到步数上限就停

        res = evaluate(model, val_dl, device)          # 本轮结束，在验证集上评一次
        log.append({"epoch": epoch, "train_loss": round(running / max(1, n_step), 4),
                    "val_acc": res["acc"], "val_macro_f1": res["macro_f1"],
                    "elapsed_s": round(time.time() - t0, 1)})
        print(f"epoch {epoch}: val_acc={res['acc']} val_f1={res['macro_f1']}", flush=True)

        if res["macro_f1"] > best and not args.no_save:
            # ★只保存「最好的一轮」的权重（早停思想的简化版）：泛化最好的一轮才是要评测的那一轮
            best = res["macro_f1"]
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            ckpt_path = OUT_DIR / (args.ckpt_out or "bert_ckpt.pt")
            # 权重里除参数外还存 pool / 指标 / 全部超参 → 评测脚本靠它们复原结构
            torch.save({"state_dict": model.state_dict(), "pool": args.pool,
                        "val_acc": res["acc"], "val_macro_f1": res["macro_f1"], "args": vars(args)},
                       ckpt_path)
            print(f"checkpoint → {ckpt_path}", flush=True)

    if not args.no_save:
        (OUT_DIR / "bert_train_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"最佳 val Macro-F1 = {best} → {OUT_DIR / 'bert_ckpt.pt'}")
    else:
        print("--no_save：本次不落盘（冒烟模式），ckpt 与日志都没写")


if __name__ == "__main__":
    main()
