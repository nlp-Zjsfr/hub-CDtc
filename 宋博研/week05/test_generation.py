"""
Transformer 字符级语言模型的生成测试脚本。

目的：不只是"看看能写出什么"，而是可复现地测出三件事——
    1. grep 模式下模型是否连贯（最长重复子串 = 复读的直接证据）
    2. 采样策略下多样性是否提升而不崩坏（distinct-2 / distinct-3）
    3. 模型每步的置信度（原始 logits 上的 top-1 概率均值）

用法:
    python test_generation.py                                   # 默认测试
    python test_generation.py --prompt "从前有座山" --length 300
    python test_generation.py --prompts "今天" "他说" --greedy --out result.md
    python test_generation.py --temperature 1.0 --top_p 0.95 --rep_penalty 1.15
"""

import argparse
import time
import torch
import torch.nn.functional as F

from language_model import LM


# ─────────────────────────── 加载 ───────────────────────────

def load_model(path, device):
    """从 checkpoint 重建 Transformer 模型。"""
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck["args"]

    if a.get("model") != "transformer":
        raise ValueError(f"{path} 不是 transformer 模型（实际是 {a.get('model')}）")

    model = LM(
        vocab_size=len(ck["char2idx"]),
        embed_dim=a["embed_dim"],
        hidden_dim=a["hidden_dim"],
        num_layers=a["num_layers"],
        model_type=a["model"],
        dropout=a["dropout"],
        nhead=a.get("nhead", 4),
        max_len=a.get("max_len", 512),
    ).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()                       # 关掉 dropout，否则每次采样结果都不同
    return model, ck["char2idx"], ck["idx2char"], a


def encode(prompt, char2idx):
    """prompt 转字符 id，词表外的字跳过并返回被丢掉的字符。"""
    ids, dropped = [], []
    for c in prompt:
        if c in char2idx:
            ids.append(char2idx[c])
        else:
            dropped.append(c)
    return ids, dropped


# ─────────────────────────── 解码 ───────────────────────────

def apply_repetition_penalty(logits, history, penalty):
    """
    对已经出现过的字施加惩罚：正 logit 除以 penalty，负 logit 乘以 penalty。
    penalty>1 会压低重复字的概率，用来区分"模型在复读"还是"prompt 太短导致只能复读"。
    """
    if penalty == 1.0 or not history:
        return logits
    seen = torch.tensor(sorted(set(history)), device=logits.device, dtype=torch.long)
    vals = logits[seen]
    logits[seen] = torch.where(vals > 0, vals / penalty, vals * penalty)
    return logits


def filter_top_k(logits, k):
    if k <= 0:
        return logits
    v, _ = torch.topk(logits, min(k, logits.size(-1)))
    logits[logits < v[-1]] = float("-inf")
    return logits


def filter_top_p(logits, p):
    """核采样：只保留累计概率达到 p 的最小集合。"""
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    cutoff = torch.cumsum(probs, dim=-1) > p
    cutoff[1:] = cutoff[:-1].clone()   # 保留第一个越过 p 的字
    cutoff[0] = False
    sorted_logits[cutoff] = float("-inf")
    logits[sorted_idx] = sorted_logits
    return logits


@torch.no_grad()
def generate(model, char2idx, idx2char, prompt, length, max_ctx, device,
             temperature=0.8, top_k=20, top_p=1.0, rep_penalty=1.0,
             seed=42, greedy=False):
    """
    自回归生成。返回 (完整文本, 每步 top-1 原始概率列表)。

    注意 ctx 截断：Transformer 的位置编码最大 max_len，喂进去的序列
    必须截到最近 max_ctx 个 token，否则 pos_embed 会越界。
    """
    torch.manual_seed(seed)

    ids, dropped = encode(prompt, char2idx)
    if dropped:
        print(f"  [警告] prompt 中 {len(dropped)} 个字不在词表，已跳过: {''.join(dropped)}")
    if not ids:
        ids = [torch.randint(len(char2idx), (1,)).item()]

    out = list(prompt)
    confidences = []

    for _ in range(length):
        ctx = ids[-max_ctx:]
        x = torch.tensor([ctx], dtype=torch.long, device=device)
        logits = model(x)[0, -1]                     # (V,) 最后一个位置的打分

        # 置信度在原始 logits 上算，不受 temperature 影响
        probs_raw = F.softmax(logits, dim=-1)
        confidences.append(probs_raw.max().item())

        logits = logits.clone()
        logits = apply_repetition_penalty(logits, ids, rep_penalty)

        if greedy:
            nxt = torch.argmax(logits).item()
        else:
            logits = logits / max(temperature, 1e-6)
            logits = filter_top_k(logits, top_k)
            logits = filter_top_p(logits, top_p)
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1).item()

        ids.append(nxt)
        out.append(idx2char[nxt])

    return "".join(out), confidences


# ─────────────────────────── 质量指标 ───────────────────────────

def distinct_n(text, n):
    """不同 n-gram 数 / 总 n-gram 数。越接近 1 越不重复。"""
    grams = [text[i: i + n] for i in range(len(text) - n + 1)]
    if not grams:
        return 0.0
    return len(set(grams)) / len(grams)


def longest_repeated_substring(text):
    """
    O(n^2) DP 找最长重复子串（子串出现 ≥2 次）。200 字以内性能无压力。
    返回值 (长度, 具体子串)。
    """
    n = len(text)
    if n < 2:
        return 0, ""
    prev = [0] * (n + 1)
    best_len, best_end = 0, 0
    for i in range(1, n + 1):
        cur = [0] * (n + 1)
        for j in range(i + 1, n + 1):
            if text[i - 1] == text[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best_len:
                    best_len, best_end = cur[j], i
        prev = cur
    return best_len, text[best_end - best_len: best_end]


def repetition_rate(text, n=4):
    """n-gram 重复率：有多少 n-gram 是前面已经出现过的。"""
    seen, dup = set(), 0
    total = 0
    for i in range(len(text) - n + 1):
        g = text[i: i + n]
        total += 1
        if g in seen:
            dup += 1
        seen.add(g)
    return dup / total if total else 0.0


def compute_metrics(gen_text, prompt, char2idx):
    body = gen_text[len(prompt):]
    lrs_len, lrs = longest_repeated_substring(body)
    oov = sum(1 for c in body if c not in char2idx)
    return {
        "长度": len(body),
        "distinct-2": distinct_n(body, 2),
        "distinct-3": distinct_n(body, 3),
        "重复率(4-gram)": repetition_rate(body, 4),
        "最长重复子串": f"{lrs_len} 字「{lrs}」",
        "OOV": oov,
    }


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",        default="transformer.pt")
    parser.add_argument("--prompt",      default="今天")
    parser.add_argument("--prompts",     nargs="+", default=None,
                        help="一次测多个开头，优先于 --prompt")
    parser.add_argument("--length",      type=int,   default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k",       type=int,   default=20, help="0=不限制")
    parser.add_argument("--top_p",       type=float, default=1.0, help="1.0=不限制")
    parser.add_argument("--rep_penalty", type=float, default=1.0, help=">1 抑制重复")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--greedy",      action="store_true",
                        help="额外跑一遍贪心解码作为对照")
    parser.add_argument("--max_ctx",     type=int,   default=None,
                        help="生成时喂给模型的最大上下文长度，默认取 ckpt 的 max_len")
    parser.add_argument("--device",      default="cpu")
    parser.add_argument("--out",         default=None, help="结果写入 markdown 文件")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, char2idx, idx2char, a = load_model(args.ckpt, device)
    max_ctx = args.max_ctx or a.get("max_len", 512)

    prompts = args.prompts or [args.prompt]
    print(f"模型: {args.ckpt}  "
          f"(embed={a['embed_dim']} ffn={a['hidden_dim']} "
          f"layers={a['num_layers']} heads={a.get('nhead', 4)})")
    print(f"参数: length={args.length} temperature={args.temperature} "
          f"top_k={args.top_k} top_p={args.top_p} rep={args.rep_penalty} "
          f"max_ctx={max_ctx} seed={args.seed}")

    modes = [("采样", False)]
    if args.greedy:
        modes.append(("贪心", True))

    report = []
    report.append(f"# 生成测试报告\n")
    report.append(f"- ckpt: `{args.ckpt}`")
    report.append(f"- 参数: length={args.length}, temperature={args.temperature}, "
                  f"top_k={args.top_k}, top_p={args.top_p}, "
                  f"rep_penalty={args.rep_penalty}, seed={args.seed}\n")

    for prompt in prompts:
        for mode_name, greedy in modes:
            t0 = time.time()
            text, conf = generate(
                model, char2idx, idx2char, prompt, args.length, max_ctx, device,
                temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                rep_penalty=args.rep_penalty, seed=args.seed, greedy=greedy,
            )
            elapsed = time.time() - t0
            m = compute_metrics(text, prompt, char2idx)

            print("\n" + "═" * 70)
            print(f"prompt=「{prompt}」  模式={mode_name}  耗时={elapsed:.2f}s")
            print("═" * 70)
            print(text)
            print("─" * 70)
            print("  " + "  ".join(f"{k}={v}" if isinstance(v, float) else f"{k}={v}"
                                   for k, v in m.items()))
            print(f"  平均 top-1 置信度={sum(conf) / len(conf):.4f}  "
                  f"最低={min(conf):.4f}")

            report.append(f"## prompt「{prompt}」— {mode_name}\n")
            report.append("```\n" + text + "\n```\n")
            report.append("| 指标 | 值 |\n|---|---|")
            for k, v in m.items():
                report.append(f"| {k} | {v} |")
            report.append(f"| 平均置信度 | {sum(conf) / len(conf):.4f} |")
            report.append(f"| 耗时 | {elapsed:.2f}s |\n")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(report))
        print(f"\n报告已写入 {args.out}")


if __name__ == "__main__":
    main()
