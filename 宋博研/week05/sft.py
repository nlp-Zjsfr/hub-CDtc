"""
SFT（监督微调）：在预训练的 Transformer 字符语言模型上继续训练。

与预训练的差别只有三处：
    1. 数据是"问：xxx\n答：xxx"整条样本，而不是切块的连续语料
    2. loss 只计算答案部分，"问"的位置置成 -100 跳过
    3. 保留原 checkpoint 的字表，不重新建词表

用法:
    python sft.py
    python sft.py --epochs 30 --lr 1e-4
    python sft.py --train sft_train.jsonl --val sft_val.jsonl --freeze_layers 2

产物 transformer_sft.pt 的格式与原 ckpt 一致（model_state/char2idx/idx2char/args），
generate.py 和 test_generation.py 可以直接加载。
"""

import argparse
import copy
import json
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

from language_model import LM
from test_generation import generate as generate_text


# ─────────────────────────── 数据 ───────────────────────────

# 模板 A：问句 + 换行 + "答："，答案紧跟在后面
PROMPT_TEMPLATE = "问：{q}\n答："
# 答案里不允许出现这两个串，否则 masking 的边界会被切错
FORBIDDEN_IN_ANSWER = ("问：", "答：")


class SampleError(ValueError):
    """单条样本无法使用。oov 保存导致它被跳过的字表外字符，便于汇总。"""

    def __init__(self, msg, oov):
        super().__init__(msg)
        self.oov = set(oov)


def load_jsonl(path):
    """jsonl：每行一个 JSON 对象；也兼容整文件是一个 JSON 数组的旧格式。"""
    with open(path, encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    if text.lstrip().startswith("["):
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: 没有解析到任何样本")
    return rows


def row_text(row):
    """取一条样本用于模板化的文本（答案字段兼容 output / out）。"""
    a_key = "output" if "output" in row else "out"
    return PROMPT_TEMPLATE.format(q=row["instruction"]) + row[a_key]


def extend_vocab(model, char2idx, idx2char, chars, device):
    """
    把 SFT 数据里出现、但预训练字表没有的字追加到词表末尾，并同步扩展
    embed 与 fc。旧权重原样保留，新行用已有词向量的均值初始化。

    这是 SFT 复用预训练模型时唯一需要改结构的地方：不改的话，含这些字的
    样本只能整条丢弃（本数据集 3000 条会只剩 352 条）。
    """
    new_chars = sorted(set(chars) - set(char2idx))
    if not new_chars:
        return char2idx, idx2char

    old_v = len(char2idx)
    for i, c in enumerate(new_chars):
        char2idx[c] = old_v + i
        idx2char[old_v + i] = c
    new_v = len(char2idx)

    with torch.no_grad():
        old_w = model.embed.weight.detach().clone()
        new_w = torch.empty(new_v, old_w.size(1), device=device, dtype=old_w.dtype)
        new_w[:old_v] = old_w
        new_w[old_v:] = old_w.mean(0, keepdim=True)
        model.embed = nn.Embedding(new_v, old_w.size(1)).to(device)
        model.embed.weight.copy_(new_w)

        old_fc = model.fc
        new_fc = nn.Linear(old_fc.in_features, new_v).to(device)
        new_fc.weight[:old_v] = old_fc.weight.detach()
        new_fc.bias[:old_v] = old_fc.bias.detach()
        new_fc.weight[old_v:] = old_fc.weight.detach().mean(0, keepdim=True)
        new_fc.bias[old_v:] = old_fc.bias.detach().mean()
        model.fc = new_fc
    return char2idx, idx2char


class SFTDataset(Dataset):
    """
    一条样本 = 输入 ids（去掉最后一个字）+ labels（错开一位，问句部分置 -100）。

        text : 问 ： 数 字 1 ... ？ \n 答 ： 数 字 1 和 ...
        x    : 问 ： 数 字 1 ... ？ \n 答 ： 数 字 1 和 ...（去掉最后一个）
        y    : ： 数 字 1 ... ？ \n 答 ： 数 字 1 和 ...（错开一位）
        l    : -100 ... -100（问句部分）  ： 数 字 1 和 ...（答案部分）
                         ↑ 从 plen-1 开始，才是"第一个答案字"
    """

    def __init__(self, rows, char2idx, warn_seq_len=64, max_len=512, oov="skip"):
        self.samples = []
        self.skipped = 0
        self.oov_chars = set()
        self.long_samples = 0

        for i, row in enumerate(rows):
            try:
                ids, plen = self.encode(row, char2idx, i)
            except SampleError as e:
                if oov == "error":
                    raise
                self.skipped += 1
                self.oov_chars |= e.oov
                continue
            if len(ids) - 1 > max_len - 1:
                raise ValueError(
                    f"样本 {i} 长度 {len(ids)} 超过位置编码上限 {max_len}，请缩短数据"
                )
            if len(ids) - 1 > warn_seq_len:
                self.long_samples += 1
            self.samples.append((torch.tensor(ids, dtype=torch.long), plen))

    @staticmethod
    def encode(row, char2idx, idx):
        # 兼容两种字段名：新数据用 output，旧数据用 out
        a_key = "output" if "output" in row else "out"
        for key in ("instruction", a_key):
            if key not in row:
                raise SampleError(f"样本 {idx} 缺少字段 {key!r}", set())

        q, a = row["instruction"], row[a_key]
        for sep in FORBIDDEN_IN_ANSWER:
            if sep in a:
                raise SampleError(
                    f"样本 {idx} 的答案里出现了 {sep!r}，会和模板分隔符冲突", set()
                )

        prompt = PROMPT_TEMPLATE.format(q=q)
        text = prompt + a
        missing = {c for c in text if c not in char2idx}
        if missing:
            raise SampleError(
                f"样本 {idx} 含字表外的字: {''.join(sorted(missing))!r}", missing
            )
        return [char2idx[c] for c in text], len(prompt)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        ids, plen = self.samples[i]
        x = ids[:-1]
        labels = ids[1:].clone()
        labels[: plen - 1] = -100        # 问句部分不参与 loss
        return x, labels


def collate(batch, pad_id):
    """右侧填充（因果掩码下 pad 不会被真实 token 看到），pad 位置 labels=-100。"""
    xs, ys = zip(*batch)
    T = max(len(x) for x in xs)

    xb = torch.full((len(xs), T), pad_id, dtype=torch.long)
    yb = torch.full((len(ys), T), -100, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        xb[i, : len(x)] = x
        yb[i, : len(y)] = y
    return xb, yb


# ─────────────────────────── 训练 ───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True, clip=1.0):
    """返回 (平均 loss, 答案 token 准确率)。loss 只在非 -100 的位置上算。"""
    model.train(train)
    total_loss, total_tokens, correct = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        valid = y != -100
        n = valid.sum().item()
        total_loss += loss.item() * n
        total_tokens += n
        correct += (logits.argmax(-1)[valid] == y[valid]).sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    acc = correct / max(total_tokens, 1)
    return avg_loss, acc


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",         default="transformer.pt", help="预训练 checkpoint")
    parser.add_argument("--train",        default="sft_train.jsonl", help="训练集（jsonl/json）")
    parser.add_argument("--val",          default="sft_val.jsonl",
                        help="验证集（jsonl/json）；给空串则从训练集按 --val_ratio 划分")
    parser.add_argument("--save",         default="transformer_sft.pt")
    parser.add_argument("--epochs",       type=int,   default=20)
    parser.add_argument("--batch_size",   type=int,   default=8)
    parser.add_argument("--lr",           type=float, default=2e-4,
                        help="比预训练小一个量级，避免冲掉已有能力")
    parser.add_argument("--val_ratio",    type=float, default=0.1,
                        help="未提供 --val 时，从训练集划分验证集的比例")
    parser.add_argument("--oov",          choices=["skip", "error"], default="skip",
                        help="含字表外字符的样本：skip=跳过并统计，error=直接报错")
    parser.add_argument("--extend_vocab", action="store_true", default=True,
                        help="把 SFT 中出现的新字追加进词表并扩展 embed/fc（默认开）")
    parser.add_argument("--no_extend_vocab", dest="extend_vocab", action="store_false")
    parser.add_argument("--freeze_layers", type=int, default=0,
                        help="冻结前 N 层 + embed（0=全部可训）")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_preview",   action="store_true", help="训练后不做生成预览")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    # ── 1. 加载预训练模型：结构参数一律取自 ckpt，不能用默认值 ──
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    a = ck["args"]
    if a.get("model") != "transformer":
        raise ValueError(f"{args.ckpt} 不是 transformer（实际 {a.get('model')}）")

    char2idx, idx2char = ck["char2idx"], ck["idx2char"]
    model = LM(
        vocab_size=len(char2idx),
        embed_dim=a["embed_dim"], hidden_dim=a["hidden_dim"],
        num_layers=a["num_layers"], model_type=a["model"],
        dropout=a["dropout"], nhead=a.get("nhead", 4), max_len=a.get("max_len", 512),
    ).to(device)
    model.load_state_dict(ck["model_state"])
    print(f"已加载 {args.ckpt}: embed={a['embed_dim']} ffn={a['hidden_dim']} "
          f"layers={a['num_layers']} heads={a.get('nhead', 4)} 字表={len(char2idx)}")

    # ── 2. 数据 ──
    warn_len = a.get("seq_len", 64)
    max_len = a.get("max_len", 512)

    train_rows = load_jsonl(args.train)
    val_rows = load_jsonl(args.val) if args.val else []

    # 先扩词表，再建数据集：词表固定了，含 OOV 的样本才能用上
    if args.extend_vocab:
        seen = set()
        for row in train_rows + val_rows:
            seen |= set(row_text(row))
        before = len(char2idx)
        char2idx, idx2char = extend_vocab(model, char2idx, idx2char, seen, device)
        print(f"词表: {before} → {len(char2idx)}（新增 {len(char2idx) - before} 个字）")

    # 冻结必须放在扩词表之后：extend_vocab 会新建 embed/fc 模块，
    # 先冻结会被新模块默认的 requires_grad=True 覆盖掉
    if args.freeze_layers > 0:
        model.embed.requires_grad_(False)
        model.pos_embed.requires_grad_(False)
        for i, layer in enumerate(model.transformer.layers):
            if i < args.freeze_layers:
                layer.requires_grad_(False)
        n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(f"已冻结 embed/pos_embed + 前 {args.freeze_layers} 层，"
              f"共 {n_frozen:,} 个参数不更新")

    train_ds = SFTDataset(train_rows, char2idx, warn_seq_len=warn_len,
                          max_len=max_len, oov=args.oov)
    print(f"训练集: {args.train}  "
          f"{len(train_rows)} 条 → 可用 {len(train_ds)} 条"
          f"（跳过 {train_ds.skipped} 条）")
    if train_ds.oov_chars:
        print(f"  [字表外字符] {len(train_ds.oov_chars)} 种: "
              f"{''.join(sorted(train_ds.oov_chars))[:60]}")
    if train_ds.long_samples:
        print(f"  [提示] {train_ds.long_samples} 条样本长度 > 预训练 seq_len {warn_len}，"
              f"位置编码 {warn_len}+ 部分训练时没学过")
    if not train_ds.samples:
        raise ValueError("训练集没有任何可用样本")

    # 验证集：优先用独立文件，否则从训练集划分
    # 用 Subset 而不是 samples 列表：__getitem__ 里才会把 ids 切成 (x, labels 掩码)
    val_items, val_ds = [], None
    if args.val:
        val_ds = SFTDataset(val_rows, char2idx, warn_seq_len=warn_len,
                            max_len=max_len, oov=args.oov)
        val_items = list(range(len(val_ds)))
        print(f"验证集: {args.val}  {len(val_rows)} 条 → 可用 {len(val_items)} 条"
              f"（跳过 {val_ds.skipped} 条）")
        if val_ds.oov_chars:
            print(f"  [字表外字符] {len(val_ds.oov_chars)} 种: "
                  f"{''.join(sorted(val_ds.oov_chars))[:60]}")
        train_items = list(range(len(train_ds)))
    elif args.val_ratio > 0 and len(train_ds) >= 5:
        idx = list(range(len(train_ds)))
        random.shuffle(idx)
        n_val = max(1, int(len(train_ds) * args.val_ratio))
        val_items = idx[:n_val]
        train_items = idx[n_val:]
        print(f"未提供 --val，从训练集划分: 训练 {len(train_items)} / 验证 {len(val_items)}")
    else:
        train_items, val_items = list(range(len(train_ds))), []
        if len(train_ds) == 1:
            print("  [提示] 只有 1 条样本，模型会直接背下来；用来验证流程是否跑通")

    pad_id = char2idx.get("\n", 0)

    def _collate(batch):
        return collate(batch, pad_id)

    bs = min(args.batch_size, len(train_items))
    train_loader = DataLoader(Subset(train_ds, train_items), batch_size=bs,
                              shuffle=True, collate_fn=_collate)
    val_loader = None
    if val_items:
        val_loader = DataLoader(Subset(val_ds if val_ds is not None else train_ds, val_items),
                                batch_size=bs, shuffle=False, collate_fn=_collate)
    print(f"batch_size={bs}  批数/epoch={len(train_loader)}")

    # ── 3. 训练 ──
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    # 有验证集时按 val loss 选最佳；没有验证集时退化为按 train loss 选，
    # 绝不能拿"未训练"的初始权重当 best_state（否则训练完会被回滚）
    best_metric = float("inf")
    best_state = None

    print(f"\n{'Epoch':>6}  {'Train Loss':>11}  {'Answer Acc':>11}"
          f"  {'Val Loss':>10}  {'Val Acc':>9}")
    print("-" * 55)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        if val_loader:
            with torch.no_grad():
                va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
            metric = va_loss
            marker = "  *" if metric < best_metric else ""
            if metric < best_metric:
                best_metric = metric
                best_state = copy.deepcopy(model.state_dict())
            print(f"{epoch:>6}  {tr_loss:>11.4f}  {tr_acc:>11.2%}"
                  f"  {va_loss:>10.4f}  {va_acc:>9.2%}{marker}")
        else:
            metric = tr_loss
            marker = "  *" if metric < best_metric else ""
            if metric < best_metric:
                best_metric = metric
                best_state = copy.deepcopy(model.state_dict())
            print(f"{epoch:>6}  {tr_loss:>11.4f}  {tr_acc:>11.2%}"
                  f"  {'-':>10}  {'-':>9}{marker}")

    model.load_state_dict(best_state)
    model.eval()

    # ── 4. 保存：保持与原 ckpt 相同的四件套，generate.py 可直接加载 ──
    out_args = copy.deepcopy(a)
    out_args.update({
        "sft_source": args.ckpt,
        "sft_epochs": args.epochs,
        "sft_lr": args.lr,
        "sft_samples": len(train_items),
        "sft_val_samples": len(val_items),
        "sft_oov_skipped": train_ds.skipped + (val_ds.skipped if args.val else 0),
        "sft_template": PROMPT_TEMPLATE,
        "freeze_layers": args.freeze_layers,
    })
    torch.save({
        "model_state": model.state_dict(),
        "char2idx": char2idx,
        "idx2char": idx2char,
        "args": out_args,
    }, args.save)
    print(f"\nSFT 完成，已保存至 {args.save}")
    if val_loader:
        print(f"最佳验证 loss: {best_metric:.4f}")

    # ── 5. 立刻做一次贪心生成预览 ──
    if not args.no_preview:
        prompt = PROMPT_TEMPLATE.format(q=train_rows[0]["instruction"])
        text, _ = generate_text(
            model, char2idx, idx2char, prompt, length=60,
            max_ctx=a.get("max_len", 512), device=device,
            greedy=True, seed=args.seed,
        )
        print("\n" + "─" * 60)
        print("生成预览（贪心）:")
        print(text)
        print("─" * 60)


if __name__ == "__main__":
    main()
