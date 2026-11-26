# -*- coding: utf-8 -*-
"""
Qwen3-0.6B + LoRA 事件抽取验证脚本（对应 llm_lora_train.py）

用法：
    python llm_lora_valid.py                 # 全量 920 条
    python llm_lora_valid.py --limit 16      # 只跑 16 条，用来先测速（结果另存）
    python llm_lora_valid.py --batch 16      # 调大生成 batch（显存够的话更快）

流程：
  1. 加载 Qwen3-0.6B + output/llm_lora 下的 LoRA adapter；
  2. 用与训练完全一致的 prompt 批量生成 JSON；
  3. 从生成的 JSON 里取出 trigger/args 的文本，再用「在原文中反查位置」还原 start/end
     （不让模型直接生成字符下标 —— 小模型算不准 offset，训练目标里也没有 offset）；
  4. 用与 bert/bert_crf/bilstm 完全相同的口径算 Trigger / Argument / Event 指标。

关于「只生成 JSON」：
    本地 transformers 没有 OpenAI 的 response_format 参数。这里的等价做法是
    prompt 里写死 JSON 骨架 + 严格指令 + 贪心解码 + 平衡花括号解析。
    解析失败会记入 bad_json 统计；若大量失败，说明该加 epoch 或调 prompt。
"""

import os
import re
import json
import time
import argparse
from collections import defaultdict, Counter

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MAX_LEN = 1024
BATCH_SIZE = 32              # 实测 32 比 8 快 3.2 倍（17.5s/32条 vs 13.8s/8条）
MAX_NEW_TOKENS = 256          # 训练集答案最长 234 token，256 足够；调小能显著防「跑满上限」
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SYSTEM_PROMPT = "请你作为一个语言提取助手，帮我提取下面语句中的信息。"
JSON_SCHEMA = ('{"events":[{"type":"事件类型","trigger":{"text":"触发词"},'
               '"args":[{"role":"角色","text":"论元"}]}]}')


# ================= prompt 构造（与训练一致） =================
def build_user_content(text):
    return (f"{text}\n\n"
            f"要求：只输出一个JSON对象，不要输出任何解释、Markdown代码块或多余文字。\n"
            f"输出格式：{JSON_SCHEMA}；如果没有事件，输出 {{\"events\":[]}}。")


def build_prompt_ids(tokenizer, text, system_prompt=SYSTEM_PROMPT):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_user_content(text)},
    ]
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
    )
    return ids[:MAX_LEN]


# ================= JSON 解析 =================
def extract_json(text):
    if not text:
        return None
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s).strip()

    start = s.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start:i + 1])
                    except json.JSONDecodeError:
                        return None
    return None


# ================= 由 JSON 还原事件（含 offset 反查） =================
def json_to_events(obj, text, used=None):
    if used is None:
        used = []

    def locate(name):
        if not isinstance(name, str):
            return -1, -1
        name = name.strip()
        if not name:
            return -1, -1
        pos = -1
        while True:
            pos = text.find(name, pos + 1)
            if pos < 0:
                return -1, -1
            span = (pos, pos + len(name))
            if not any(s < span[1] and e > span[0] for s, e in used):
                used.append(span)
                return span

    events = []
    for e in (obj.get("events") or []):
        if not isinstance(e, dict):
            continue
        trig = e.get("trigger") or {}
        if isinstance(trig, str):
            trig = {"text": trig}
        tname = (trig.get("text") or "").strip()
        ts, te = locate(tname)

        args = []
        for a in (e.get("args") or []):
            if not isinstance(a, dict):
                continue
            role = (a.get("role") or "").strip()
            aname = (a.get("text") or "").strip()
            if not role or not aname:
                continue
            s, en = locate(aname)
            args.append({"role": role, "text": aname, "start": s, "end": en})

        events.append({
            "type": e.get("type") or "Unknown",
            "trigger": {"text": tname, "start": ts, "end": te},
            "args": args,
        })
    return events


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return round(p, 4), round(r, 4), round(f1, 4)


# ================= 批量生成（带进度 & 计时） =================
@torch.inference_mode()
def generate_answers(model, tokenizer, texts, system_prompt,
                     batch_size=BATCH_SIZE, max_new_tokens=MAX_NEW_TOKENS):
    model.eval()
    # 关掉 sampling 相关参数，避免每个 batch 刷警告
    try:
        gc = model.generation_config
        gc.do_sample = False
        gc.temperature = None
        gc.top_p = None
        gc.top_k = None
    except Exception:
        pass

    pad_id = tokenizer.pad_token_id
    eos_ids = [tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id]

    answers, hit_cap = [], 0
    n_batch = (len(texts) + batch_size - 1) // batch_size
    t_start = time.time()
    pbar = tqdm(range(0, len(texts), batch_size), total=n_batch,
                desc="生成中", unit="batch")

    for bi, i in enumerate(pbar):
        chunk = texts[i:i + batch_size]
        prompts = [build_prompt_ids(tokenizer, t, system_prompt) for t in chunk]

        # 手动左填充（生成必须左填充），顺便避开 tokenizer.pad 的 warning
        maxlen = max(len(p) for p in prompts)
        input_ids = [[pad_id] * (maxlen - len(p)) + p for p in prompts]
        attn_mask = [[0] * (maxlen - len(p)) + [1] * len(p) for p in prompts]

        enc = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=DEVICE),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.long, device=DEVICE),
        }
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.05,
            eos_token_id=eos_ids,
            pad_token_id=pad_id,
        )
        new_tokens = out[:, enc["input_ids"].shape[1]:]
        for row in new_tokens:
            if row.numel() >= max_new_tokens:
                hit_cap += 1
        answers.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))

        elapsed = time.time() - t_start
        done = len(answers)
        eta = elapsed / done * (len(texts) - done) if done else 0
        pbar.set_postfix(done=done, hit_cap=hit_cap,
                         s_per_batch=f"{elapsed / (bi + 1):.1f}",
                         eta_min=f"{eta / 60:.1f}")

    total = time.time() - t_start
    print(f"   生成完成：{len(texts)} 条，用时 {total/60:.1f} 分钟，"
          f"平均 {total/len(texts)*1000:.0f} ms/条")
    if hit_cap:
        print(f"   ⚠ 有 {hit_cap} 条生成到 max_new_tokens={max_new_tokens} 仍未结束"
              f"（模型没吐 <|im_end|>，会拖慢速度，但不影响前面 JSON 的解析）")
    return answers


# ================= 主流程 =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条（测速用）")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE, help="生成 batch size")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)

    model_path = os.path.join(script_dir, "Qwen3-0.6B")
    adapter_dir = os.path.join(root_dir, "output", "llm_lora")
    eval_path = os.path.join(root_dir, "data", "processed", "dev_extract_clean.jsonl")

    output_dir = os.path.join(root_dir, "output", "bilstm_crf")
    os.makedirs(output_dir, exist_ok=True)
    if args.limit:
        output_path = os.path.join(output_dir, f"llm_lora_sample{args.limit}.jsonl")
    else:
        output_path = os.path.join(output_dir, "llm_lora.jsonl")

    if not os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
        print(f"❌ 找不到 LoRA adapter: {adapter_dir}，请先运行 llm_lora_train.py")
        return

    print(f"✅ 当前使用的设备: {DEVICE}")
    print("1. 加载模型与 LoRA adapter...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    system_prompt = SYSTEM_PROMPT
    cfg_path = os.path.join(adapter_dir, "prompt_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        system_prompt = cfg.get("system_prompt", SYSTEM_PROMPT)
        print(f"   已读取 prompt_config.json")
    print(f"   system prompt: {system_prompt}")

    base = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, adapter_dir).to(DEVICE)
    model.eval()
    model.config.use_cache = True
    print(f"   模型参数: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"   显存: 空闲 {free/1e9:.1f}GB / {total/1e9:.1f}GB")

    print("2. 加载验证集...")
    rows = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if args.limit:
        rows = rows[:args.limit]

    gt_by_sent = defaultdict(list)
    for r in rows:
        if r.get("event"):
            gt_by_sent[r["sent_id"]].append(r["event"])
    n_gt_events = sum(len(v) for v in gt_by_sent.values())
    n_gt_args = sum(len(g.get("args", [])) for v in gt_by_sent.values() for g in v)
    print(f"   句子数: {len(rows)} | 含事件句子: {len(gt_by_sent)} | GT事件: {n_gt_events} | GT论元: {n_gt_args}")

    print(f"3. 开始生成（batch={args.batch}, max_new_tokens={args.max_new_tokens}）...")
    answers = generate_answers(model, tokenizer, [r["text"] for r in rows], system_prompt,
                               batch_size=args.batch, max_new_tokens=args.max_new_tokens)

    print("4. 计算指标...")
    stats = {"trig": [0, 0, 0], "arg": [0, 0, 0], "event": [0, 0, 0]}
    results = []
    bad_json = 0

    for r, ans in zip(rows, answers):
        text = r["text"]
        obj = extract_json(ans)
        if obj is None:
            bad_json += 1
            obj = {"events": []}
        pred_events = json_to_events(obj, text)
        true_events = gt_by_sent.get(r["sent_id"], [])

        gt_trig = Counter((g["trigger"]["text"], g["trigger"]["start"]) for g in true_events)
        pr_trig = Counter((e["trigger"]["text"], e["trigger"]["start"]) for e in pred_events)
        tp = sum((gt_trig & pr_trig).values())
        stats["trig"][0] += tp
        stats["trig"][1] += sum(pr_trig.values()) - tp
        stats["trig"][2] += sum(gt_trig.values()) - tp

        gt_args = Counter((a["role"], a["text"], a["start"])
                          for g in true_events for a in g.get("args", []))
        pr_args = Counter((a["role"], a["text"], a["start"])
                          for e in pred_events for a in e.get("args", []))
        tp = sum((gt_args & pr_args).values())
        stats["arg"][0] += tp
        stats["arg"][1] += sum(pr_args.values()) - tp
        stats["arg"][2] += sum(gt_args.values()) - tp

        def key(e):
            return (e["trigger"]["text"], e["trigger"]["start"],
                    frozenset((a["role"], a["text"]) for a in e.get("args", [])))
        gt_ev = Counter(key(g) for g in true_events)
        pr_ev = Counter(key(e) for e in pred_events)
        tp = sum((gt_ev & pr_ev).values())
        stats["event"][0] += tp
        stats["event"][1] += sum(pr_ev.values()) - tp
        stats["event"][2] += sum(gt_ev.values()) - tp

        results.append({
            "sent_id": r["sent_id"],
            "text": text,
            "pred_event": pred_events[0] if pred_events else None,
            "pred_events": pred_events,
            "true_events": true_events,
            "raw_answer": ans,
        })

    t_p, t_r, t_f1 = prf(*stats["trig"])
    a_p, a_r, a_f1 = prf(*stats["arg"])
    e_p, e_r, e_f1 = prf(*stats["event"])

    metrics = {
        "Trigger": {"Precision": t_p, "Recall": t_r, "F1": t_f1},
        "Argument": {"Precision": a_p, "Recall": a_r, "F1": a_f1},
        "Event": {"Precision": e_p, "Recall": e_r, "F1": e_f1},
    }

    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.write(json.dumps({"global_metrics": metrics}, ensure_ascii=False) + "\n")
        for res in results:
            f_out.write(json.dumps(res, ensure_ascii=False) + "\n")

    print("\n---- 预测示例 ----")
    shown = 0
    for res in results:
        if not res["true_events"] or shown >= 2:
            continue
        print(f"[{shown + 1}] {res['text'][:50]}")
        print(f"    GT  : {res['true_events'][0]['trigger']['text']!r} "
              f"args={[(a['role'], a['text']) for a in res['true_events'][0].get('args', [])][:5]}")
        pe = res["pred_event"]
        if pe:
            print(f"    PRED: {pe['trigger']['text']!r} "
                  f"args={[(a['role'], a['text']) for a in pe.get('args', [])][:5]}")
        else:
            print("    PRED: <无>")
        shown += 1

    print("\n" + "=" * 50)
    print("✅ LLM+LoRA 验证完成！全局指标如下：")
    print(f" - Trigger F1 : {t_f1:.4f} (P: {t_p:.4f}, R: {t_r:.4f})")
    print(f" - Argument F1: {a_f1:.4f} (P: {a_p:.4f}, R: {a_r:.4f})")
    print(f" - Event F1   : {e_f1:.4f} (P: {e_p:.4f}, R: {e_r:.4f})")
    print(f" - JSON 解析失败: {bad_json}/{len(rows)}")
    print(f"\n📁 详细结果已保存至: {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
