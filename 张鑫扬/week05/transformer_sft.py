"""
transformer_sft.py — 在第 4 周字符级 Transformer 的基础上做监督微调（SFT）

【用法】在 week4 目录下执行：
    # 训练：用 sft_train.jsonl 微调预训练模型
    python transformer_sft.py --mode train --data sft_train.jsonl --ckpt tf_128.pt --val_data sft_val.jsonl --save tf_128_sft.pt --epochs 3 --batch_size 16 --lr 0.0001

    # 评测：用 sft_val.jsonl 测评微调后的模型
    python transformer_sft.py --mode eval --ckpt tf_128_sft.pt --val_data sft_val.jsonl

训练从 --ckpt 指定的预训练权重 tf_128.pt 开始，用 --data 指定的 sft_train.jsonl 更新权重。
每轮结束后，用 --val_data 指定的 sft_val.jsonl 计算验证损失。
验证损失最低的一轮权重保存到 --save 指定的 tf_128_sft.pt。

【数据】sft_train.jsonl 和 sft_val.jsonl 都是 UTF-8 JSONL，每行一条问答。
训练集中的一条实际样本：
    {"instruction":"请问北京奥运会的举办时间和地点分别是什么？","output":"北京奥运会的举办时间是2008年8月8日至8月24日，地点是北京市。"}

每条 instruction / output 会拼成「指令 + 换行 + 答：+ 回答」，损失只计算回答。
兼容 prompt / response 格式，此时分隔符由调用者写在 prompt 里。
字表和窗口长度均从 --ckpt 指定的预训练权重读取。字表外字符逐个映射到 UNK，超长样本整条跳过。
优先使用字表已有的 UNK。当前权重没有专用 UNK，默认借用「？」的 ID，也可用 --unk_token 指定。
可用 --demo 先试跑。指定 --val_data 时读取独立验证文件，否则按 --val_ratio 划分。

【用 PowerShell 和旧脚本试生成】指令后保留训练时的换行和「答：」：
    python -B transformer_lm.py --ckpt tf_128_sft.pt --generate "请问北京奥运会的举办时间和地点分别是什么？`n答：" --n_chars 40
旧脚本按指定长度续写，本脚本没有新增回答结束符。
"""

import argparse
import json
import math
import random
import sys
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

# 只导入模型定义。禁止这次导入生成或更新项目内的 __pycache__ 文件。
previous_bytecode_setting = sys.dont_write_bytecode
sys.dont_write_bytecode = True
try:
    from transformer_lm import TransformerLM
finally:
    sys.dont_write_bytecode = previous_bytecode_setting

HERE = Path(__file__).resolve().parent
IGNORE_INDEX = -100  # cross_entropy 忽略的标签值，不占字表中的任何一个 id
ANSWER_PREFIX = "\n答："


# ─────────────────────── 读数据 / 构造 SFT 标签 ───────────────────────

def load_pairs(path):
    """读取并统一两种问答格式，留待后面按预训练字表和窗口筛选、编码。"""
    if path is None:
        # 内置短句用于检查训练流程，也走 instruction / output 的读取和拼接逻辑。
        rows = [
            (1, '{"instruction":"今天天气如何？","output":"天气很好。"}'),
            (2, '{"instruction":"我们做什么？","output":"一起学习。"}'),
        ]
        source = "内置试跑数据"
    else:
        # utf-8-sig 同时支持普通 UTF-8 和带 BOM 的 UTF-8，常见于 Windows 编辑器。
        rows = enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1)
        source = str(path)

    pairs = []
    for line_no, line in rows:
        if not line.strip():
            continue
        where = f"{source} 第 {line_no} 行"
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{where} 的 JSON 格式有误：{exc.msg}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"{where} 需要一个 JSON 对象")

        instruction_format = "prompt" not in item and "response" not in item
        if not instruction_format:
            prompt, response = item.get("prompt"), item.get("response")
            fields = "prompt 和 response"
        else:
            prompt, response = item.get("instruction"), item.get("output")
            fields = "instruction 和 output"
        if any(not isinstance(value, str) or not value.strip() for value in (prompt, response)):
            raise ValueError(f"{where} 的 {fields} 都需要是非空字符串")
        if instruction_format:
            prompt += ANSWER_PREFIX
        pairs.append((prompt, response))

    if not pairs:
        raise ValueError(f"{source} 没有可用问答")
    return pairs


def resolve_unk_token(char2idx, requested_token):
    """从原字表选择一个 ID 表示未知字符。"""
    if requested_token is not None:
        if requested_token not in char2idx:
            raise ValueError(f"--unk_token {requested_token!r} 不在原模型字表中")
        return requested_token
    for token in ("<UNK>", "<unk>", "[UNK]", "�", "？", "?"):
        if token in char2idx:
            return token
    raise ValueError("原字表没有可自动选用的 UNK，请用 --unk_token 指定一个已有字符")


def encode_pairs(pairs, char2idx, seq_len, label, unk_token):
    """把字表外字符映射为 UNK，保留每个字符的位置。"""
    samples = []
    unk_id = char2idx[unk_token]
    unk_count, unk_sample_count, too_long_count = 0, 0, 0
    for prompt, response in pairs:
        full_text = prompt + response
        # 最后一个字符只作预测标签，因此实际输入长度是全文长度减 1。
        if len(full_text) - 1 > seq_len:
            too_long_count += 1
            continue
        # 一个未知字符仍然占一个位置，提示词与回答的边界保持原样。
        # 回答里的未知字符以 UNK 为训练目标，提示词里的未知字符只作为上下文。
        unknown = sum(ch not in char2idx for ch in full_text)
        unk_count += unknown
        unk_sample_count += bool(unknown)
        ids = torch.tensor([char2idx.get(ch, unk_id) for ch in full_text], dtype=torch.long)
        x, y = ids[:-1], ids[1:].clone()
        # 例：prompt="问好"，response="你好"
        # 输入 x：[问, 好, 你]        原始下一字：[好, 你, 好]
        # 标签 y：[-100, 你, 好]      第一个回答字「你」由最后一个 prompt 字预测。
        # 因为标签已经向左移了一格，所以只遮掉 len(prompt)-1 个标签。
        # prompt 仍留在 x 中作为上下文，回答的梯度仍可通过注意力传回 prompt 相关计算。
        y[:len(prompt) - 1] = IGNORE_INDEX
        samples.append((x, y))

    print(
        f"{label}保留 {len(samples)}/{len(pairs)} 条｜超长跳过 {too_long_count} 条"
        f"｜{unk_sample_count} 条共替换 {unk_count} 个字符为 UNK {unk_token!r}"
    )
    return samples


def collate_batch(samples, fill_id):
    """将不同长度的样本在右侧补齐到当前 batch 的最大长度。"""
    xs, ys = zip(*samples)
    # 原模型没有 PAD id，所以用任意有效 id 填输入。因果掩码保证真实位置看不到右侧补位。
    # 标签用 -100 填补，这些位置也不会进入交叉熵。两种掩码各有各的职责。
    x = pad_sequence(xs, batch_first=True, padding_value=fill_id)
    y = pad_sequence(ys, batch_first=True, padding_value=IGNORE_INDEX)
    return x, y


# ─────────────────────── 恢复模型 / 训练和验证 ───────────────────────

def run_epoch(model, loader, device, optimizer=None):
    """有 optimizer 就更新参数，否则只算回答字符上的平均交叉熵。"""
    training = optimizer is not None
    model.train(training)  # 训练时开启 dropout，验证时关闭
    total_loss, total_chars = 0.0, 0

    # set_grad_enabled 控制是否记录反向传播所需的计算图，和 dropout 开关独立。
    with torch.set_grad_enabled(training):
        for x, y in loader:
            count = int(y.ne(IGNORE_INDEX).sum())  # 当前 batch 有多少个有效回答字符
            x, y = x.to(device), y.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)  # 梯度默认累加，每步先清空

            logits = model(x)  # (B, T, V)，每个位置预测一个下一字符
            loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                ignore_index=IGNORE_INDEX, reduction="sum",
            )

            if training:
                (loss_sum / count).backward()  # 每一步用当前 batch 的字符平均损失
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # 整轮统计按字符加权。这样即使最后一个 batch 较小，统计口径也一致。
            total_loss += loss_sum.detach().item()
            total_chars += count

    return total_loss / total_chars


def restore_model(ckpt, device):
    """训练和评测都按权重文件中的原始配置恢复模型。"""
    base_args = ckpt["args"]
    if base_args.get("no_mask", False):
        raise ValueError("回答部分的训练和评测需要带因果掩码的模型")
    model = TransformerLM(
        len(ckpt["char2idx"]), base_args["d_model"], base_args["heads"], base_args["layers"],
        base_args["seq_len"], base_args["dropout"],
        use_pos=not base_args.get("no_pos", False), causal=True,
    )
    # 严格加载会检查参数键和张量尺寸。
    model.load_state_dict(ckpt["model_state"], strict=True)
    return model.to(device)


def evaluate_checkpoint(args, ckpt, device):
    """用指定验证集计算回答字符的平均交叉熵和困惑度。"""
    path = args.val_data or (HERE / "sft_val.jsonl")
    char2idx = ckpt["char2idx"]
    seq_len = ckpt["args"]["seq_len"]
    print(f"评测权重 {args.ckpt}｜验证文件 {path}｜设备 {device}")
    # 评测 SFT 权重时默认沿用微调时的 UNK 映射，命令行可以显式覆盖。
    saved_unk_token = ckpt.get("sft_args", {}).get("unk_token")
    requested_token = args.unk_token if args.unk_token is not None else saved_unk_token
    unk_token = resolve_unk_token(char2idx, requested_token)
    samples = encode_pairs(load_pairs(path), char2idx, seq_len, "验证集", unk_token)
    if not samples:
        raise ValueError("筛选后没有可评测样本，请检查验证数据与原模型是否兼容")
    collate = partial(collate_batch, fill_id=next(iter(char2idx.values())))
    loader = DataLoader(samples, batch_size=args.batch_size, collate_fn=collate)
    # run_epoch 收不到优化器时会关闭 dropout 和梯度，只对回答位置累计损失。
    loss = run_epoch(restore_model(ckpt, device), loader, device)
    if not math.isfinite(loss):
        raise ValueError("评测损失出现非有限值")
    # 困惑度是按回答字符平均的交叉熵取指数。
    ppl = math.exp(loss) if loss < math.log(sys.float_info.max) else math.inf
    print(f"验证回答 loss {loss:.4f}｜字符困惑度 PPL {ppl:.2f}")


def train(args, ckpt, device):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 提前发现同名文件。最终保存时还会用 xb 独占创建，避免覆盖训练途中出现的文件。
    if args.save.exists():
        raise FileExistsError(f"输出文件已存在，请换一个 --save 路径：{args.save}")

    base_args = ckpt["args"]

    train_pairs = load_pairs(args.data)
    if args.val_data is not None:
        val_pairs = load_pairs(args.val_data)
    else:
        # 没有单独的验证文件时，才按整条问答随机划分。
        random.shuffle(train_pairs)
        n_val = min(len(train_pairs) - 1, max(1, int(len(train_pairs) * args.val_ratio))) if args.val_ratio else 0
        val_pairs, train_pairs = train_pairs[:n_val], train_pairs[n_val:]

    char2idx = ckpt["char2idx"]
    seq_len = base_args["seq_len"]
    unk_token = resolve_unk_token(char2idx, args.unk_token)
    args.unk_token = unk_token  # 随权重保存实际使用的映射，供评测复用
    train_samples = encode_pairs(train_pairs, char2idx, seq_len, "训练集", unk_token)
    val_samples = encode_pairs(val_pairs, char2idx, seq_len, "验证集", unk_token)
    if not train_samples:
        raise ValueError("筛选后没有可训练样本，请检查数据与原字表、窗口长度是否兼容")
    if val_pairs and not val_samples:
        raise ValueError("筛选后没有可验证样本，请检查验证数据与原模型是否兼容")

    model = restore_model(ckpt, device)

    collate = partial(collate_batch, fill_id=next(iter(char2idx.values())))
    train_loader = DataLoader(train_samples, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_samples, batch_size=args.batch_size, collate_fn=collate)

    # SFT 从预训练参数起步，为本次训练创建新的 AdamW；这里更新全部模型参数。
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"设备 {device}｜原字表 {len(char2idx)} 字｜原窗口 {seq_len}")
    best_loss, best_epoch, best_state = math.inf, 0, None
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer)
        val_loss = run_epoch(model, val_loader, device) if val_samples else None
        summary = f"第 {epoch:>2} 轮｜训练回答 loss {train_loss:.4f}"
        if val_loss is not None:
            summary += f"｜验证回答 loss {val_loss:.4f}"
        print(summary)

        if not math.isfinite(train_loss) or (val_loss is not None and not math.isfinite(val_loss)):
            raise ValueError("损失出现非有限值，请检查权重或尝试降低学习率")

        # 有验证集就留验证 loss 最低的一轮，否则保留最后一轮。
        # 必须拷贝张量，因为 state_dict() 的张量仍与训练中的模型共享存储。
        if val_loss is None or val_loss < best_loss:
            best_loss = val_loss if val_loss is not None else train_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().to(device="cpu", copy=True)
                for name, value in model.state_dict().items()
            }

    # 沿用原 checkpoint 的四个字段，原脚本能直接加载做生成。
    # 保存原模型配置和原字表，旧脚本可按原尺寸恢复模型。
    result = {
        "model_state": best_state,
        "char2idx": char2idx,
        "idx2char": ckpt["idx2char"],
        "args": base_args,
        "sft_args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    with args.save.open("xb") as output:
        torch.save(result, output)
    print(f"已保存第 {best_epoch} 轮权重到 {args.save}")


def main():
    ap = argparse.ArgumentParser(description="字符级 Transformer 的监督微调与评测")
    ap.add_argument("--mode", choices=("train", "eval"), default="train", help="train 微调，eval 评测")
    data = ap.add_mutually_exclusive_group()
    data.add_argument("--data", type=Path, help="UTF-8 JSONL 训练集，支持 instruction/output 或 prompt/response")
    data.add_argument("--demo", action="store_true", help="使用内置两条短句试跑")
    ap.add_argument("--val_data", type=Path, help="独立 UTF-8 JSONL 验证集，评测默认使用同目录的 sft_val.jsonl")
    ap.add_argument("--ckpt", type=Path, default=HERE / "tf_mask.pt", help="输入模型权重")
    ap.add_argument("--unk_token", help="用作 UNK 的原字表项，默认优先选专用 UNK，否则用「？」或「?」")
    ap.add_argument("--save", type=Path, default=HERE / "tf_sft.pt", help="输出权重，文件必须尚不存在")
    ap.add_argument("--epochs", type=int, default=3, help="训练轮数")
    ap.add_argument("--batch_size", type=int, default=16, help="每步训练的问答条数")
    ap.add_argument("--lr", type=float, default=1e-4, help="AdamW 学习率")
    ap.add_argument("--val_ratio", type=float, default=0.1, help="未指定 --val_data 时的验证比例，0 表示关闭验证")
    ap.add_argument("--seed", type=int, default=42, help="控制样本划分和训练随机性")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="训练设备")
    args = ap.parse_args()
    if args.mode == "eval" and (args.data is not None or args.demo):
        ap.error("评测数据请用 --val_data 指定")
    if args.mode == "train" and args.data is None and not args.demo:
        ap.error("训练模式需要 --data 或 --demo")
    if args.demo and args.val_data is not None:
        ap.error("--val_data 需要和 --data 一起使用")
    if args.data is not None and args.val_data is not None and args.data.exists() and args.val_data.exists():
        if args.data.samefile(args.val_data):
            ap.error("训练集和独立验证集需要使用不同文件")
    if args.batch_size < 1:
        ap.error("--batch_size 必须大于 0")
    if args.mode == "train" and (args.epochs < 1 or not math.isfinite(args.lr) or args.lr <= 0):
        ap.error("--epochs 和 --lr 必须大于 0，学习率需要是有限数")
    if args.mode == "train" and not 0 <= args.val_ratio < 1:
        ap.error("--val_ratio 必须在 [0, 1) 之间")
    if args.device == "cuda" and not torch.cuda.is_available():
        ap.error("当前 PyTorch 无法使用 CUDA，请选择 --device cpu 或 auto")

    try:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name if args.device == "auto" else args.device)
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        if args.mode == "eval":
            evaluate_checkpoint(args, ckpt, device)
        else:
            train(args, ckpt, device)
    except (OSError, ValueError) as exc:
        ap.exit(1, f"{'评测' if args.mode == 'eval' else '训练'}失败：{exc}\n")


if __name__ == "__main__":
    main()

