"""
方法 B 评测：批量生成 → 解析 JSON → 回填 span → 统一指标（shared/metrics.py）。

  python llm_sft/evaluate.py --adapter outputs/checkpoints/llm_lora   # LoRA 微调后
  python llm_sft/evaluate.py --zero_shot                              # 不加 adapter 的 zero-shot 对照
  python llm_sft/evaluate.py --adapter ... --limit 32                 # 冒烟

除了 P/R/F1，还记录生成式特有的工程指标：
  json_ok / json_repaired / json_fail  解析状态分布（失败按空预测，不删行）
  hallucinated   输出的实体文本在原句里找不到
  unknown_type   输出了 10 类以外的类型名
  latency_ms_per_sent / gen_tokens_avg
MPS 上 generate() 必须用 eager attention（sdpa 会 mps_matmul 崩溃）。
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "llm_sft"))
from prompt import build_messages, parse_output, to_entities  # noqa: E402
from shared.data import load_jsonl, save_jsonl, subsample  # noqa: E402
from shared.metrics import evaluate, format_report  # noqa: E402
from shared.paths import CKPT_DIR, DATA_DIR, METRIC_DIR, PRED_DIR, QWEN_DIR, ensure_dirs  # noqa: E402


def pick_device(name):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=str(QWEN_DIR))
    ap.add_argument("--adapter", default=str(CKPT_DIR / "llm_lora"))
    ap.add_argument("--zero_shot", action="store_true", help="不加载 adapter，直接用 Instruct 基座")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=0, help="只评前 N 句（冒烟）")
    ap.add_argument("--frac", type=float, default=1.0, help="随机取该比例的句子评测（试运行）")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--device", default="")
    ap.add_argument("--name", default="", help="输出文件名；默认 llm_lora / llm_zero_shot")
    args = ap.parse_args()

    ensure_dirs()
    device = pick_device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    name = args.name or ("llm_zero_shot" if args.zero_shot else "llm_lora")
    if args.limit and not args.name:
        name += f"_n{args.limit}"

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    tok.padding_side = "left"                       # 批量生成必须左 padding
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=dtype, attn_implementation="eager")
    if not args.zero_shot:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()   # 合并后推理无额外开销
    model.to(device).eval()

    rows = load_jsonl(DATA_DIR / f"{args.split}.jsonl")
    rows = subsample(rows, args.frac)
    if args.limit:
        rows = rows[:args.limit]
    print(f"device={device} {name}  句数 {len(rows)}", flush=True)

    # 按 prompt 长度排序再分批：减少左 padding 浪费；输出再按原顺序写回
    prompts = [tok.apply_chat_template(build_messages(r["text"]), add_generation_prompt=True, tokenize=False)
               for r in rows]
    order = sorted(range(len(rows)), key=lambda i: len(prompts[i]))
    outputs, gen_tokens = [None] * len(rows), [0] * len(rows)
    t0 = time.time()
    for bi in range(0, len(order), args.batch_size):
        idx = order[bi:bi + args.batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        new = out[:, enc["input_ids"].shape[1]:].cpu()
        for k, i in enumerate(idx):
            ids = new[k].tolist()
            n = next((j for j, t in enumerate(ids) if t in (tok.pad_token_id, tok.eos_token_id,
                      tok.convert_tokens_to_ids("<|im_end|>"))), len(ids))
            gen_tokens[i] = n
            outputs[i] = tok.decode(ids[:n], skip_special_tokens=True)
        done = min(bi + args.batch_size, len(order))
        if (bi // args.batch_size) % 10 == 0 or done == len(order):
            print(f"  {done}/{len(order)}  {time.time() - t0:.0f}s", flush=True)
    if device.type == "mps":
        torch.mps.synchronize()
    sec = time.time() - t0

    preds, status, agg = [], Counter(), Counter()
    for r, o in zip(rows, outputs):
        d, st = parse_output(o)
        ents, s = to_entities(r["text"], d)
        status[st] += 1
        agg.update(s)
        preds.append({"id": r["id"], "entities": ents, "raw": o, "parse": st})

    m = evaluate(rows, preds)
    m.update({
        "json_ok": status["ok"], "json_repaired": status["repaired"], "json_fail": status["fail"],
        "json_ok_rate": round(status["ok"] / len(rows), 4),
        "hallucinated": agg["hallucinated"], "unknown_type": agg["unknown_type"],
        "n_returned_entities": agg["n_returned"],
        "hallucination_rate": round(agg["hallucinated"] / max(agg["n_returned"], 1), 4),
        "latency_ms_per_sent": round(sec / len(rows) * 1000, 1),
        "gen_tokens_avg": round(sum(gen_tokens) / len(rows), 1),
        "truncated_at_max_new_tokens": sum(1 for n in gen_tokens if n >= args.max_new_tokens),
        "eval_time_s": round(sec, 1),
    })
    save_jsonl(preds, PRED_DIR / f"{name}.jsonl")
    (METRIC_DIR / f"{name}.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    print(format_report(m, f"{args.split} / {name}"))
    print(f"JSON: ok {status['ok']}  repaired {status['repaired']}  fail {status['fail']}  |  "
          f"幻觉 {agg['hallucinated']}/{agg['n_returned']}  未知类型 {agg['unknown_type']}  |  "
          f"{m['latency_ms_per_sent']} ms/句  平均生成 {m['gen_tokens_avg']} token  截断 {m['truncated_at_max_new_tokens']}")


if __name__ == "__main__":
    main()
