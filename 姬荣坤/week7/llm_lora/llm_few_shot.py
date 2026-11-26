# -*- coding: utf-8 -*-
"""
Qwen3-0.6B Few-Shot 事件抽取验证（统一口径版）

与 api_few_shot.py 共用完全相同的：system prompt、few-shot 示例、评估集、指标实现。
两者唯一的差别是「谁来推理」——本地 Qwen3-0.6B vs DeepSeek API。

相对旧版的改动：
  1. 【评估集】旧版读 dev_detect_clean.jsonl 前 1000 行（80% 是无 GT 的负样本），
     现改为 dev_extract_clean.jsonl 前 200 条，全部带标注。
     旧版 api 跑 200 条、llm 跑 1000 条，两个数字本来就没法比。
  2. 【指标】旧版 `fn += len(true_events)` 把命中句也计入 FN，Recall 上限 0.5。
     现改为 fn = gt - tp，并按 (text, start) 严格匹配。
  3. 【提示词】把 event_schema.json 的类型→角色映射写进 system prompt
     （旧版没给角色清单，模型只能猜，论元几乎全错）。
  4. 贪心解码 + 手动左填充 + 关掉 sampling 警告。

用法：
    python llm_few_shot.py
    python llm_few_shot.py --n 200 --batch 8
"""

import os
import re
import json
import time
import random
import argparse
from pathlib import Path
from collections import defaultdict, Counter

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"

N_EVAL = 200
SEED = 42
MAX_LEN = 2048
BATCH_SIZE = 8
MAX_NEW_TOKENS = 256

JSON_SCHEMA = ('{"events":[{"type":"事件类型","trigger":{"text":"触发词"},'
               '"args":[{"role":"角色","text":"论元"}]}]}')


# ================= 提示词（与 api_few_shot.py 完全一致） =================
def build_system_prompt(schema):
    lines = [f"- {t}：{'、'.join(roles)}" for t, roles in schema.items()]
    return ("请你作为一个语言提取助手，帮我提取下面语句中的信息。\n\n"
            "事件类型与论元角色必须严格使用下列名称：\n" + "\n".join(lines))


def build_user_content(text):
    return (f"{text}\n\n"
            f"要求：只输出一个JSON对象，不要输出任何解释、Markdown代码块或多余文字。\n"
            f"输出格式：{JSON_SCHEMA}；如果没有事件，输出 {{\"events\":[]}}。")


def build_answer(event):
    obj = {"events": [{
        "type": event.get("type", ""),
        "trigger": {"text": event["trigger"]["text"]},
        "args": [{"role": a["role"], "text": a["text"]} for a in event.get("args", [])],
    }]}
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def load_few_shot(train_path, k=3, seed=SEED):
    with open(train_path, "r", encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]
    random.Random(seed).shuffle(items)
    examples, seen = [], set()
    for it in items:
        ev = it.get("event")
        if ev and ev.get("type") not in seen:
            examples.append((it["text"], ev))
            seen.add(ev["type"])
        if len(examples) >= k:
            break
    return examples


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


# ================= 事件还原（offset 由文本反查，与 api_few_shot.py 一致） =================
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
        events.append({"type": e.get("type") or "Unknown",
                       "trigger": {"text": tname, "start": ts, "end": te},
                       "args": args})
    return events


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return round(p, 4), round(r, 4), round(f1, 4)


# ================= 批量生成 =================
@torch.inference_mode()
def generate_answers(model, tokenizer, system_prompt, examples, texts, batch_size=BATCH_SIZE):
    model.eval()
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
    answers = []
    n_batch = (len(texts) + batch_size - 1) // batch_size
    t0 = time.time()
    pbar = tqdm(range(0, len(texts), batch_size), total=n_batch, desc="推理进度", unit="batch")

    for bi, i in enumerate(pbar):
        chunk = texts[i:i + batch_size]
        prompts = []
        for t in chunk:
            messages = [{"role": "system", "content": system_prompt}]
            for ex_text, ex_event in examples:
                messages.append({"role": "user", "content": build_user_content(ex_text)})
                messages.append({"role": "assistant", "content": build_answer(ex_event)})
            messages.append({"role": "user", "content": build_user_content(t)})

            ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
            )
            if len(ids) > MAX_LEN:              # 本数据集实测最长 1663，不会触发
                ids = ids[-MAX_LEN:]
            prompts.append(ids)

        maxlen = max(len(p) for p in prompts)
        input_ids = [[pad_id] * (maxlen - len(p)) + p for p in prompts]
        attn = [[0] * (maxlen - len(p)) + [1] * len(p) for p in prompts]
        enc = {"input_ids": torch.tensor(input_ids, dtype=torch.long, device=DEVICE),
               "attention_mask": torch.tensor(attn, dtype=torch.long, device=DEVICE)}

        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.05,
            eos_token_id=eos_ids,
            pad_token_id=pad_id,
        )
        new_tokens = out[:, enc["input_ids"].shape[1]:]
        answers.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))

        done = len(answers)
        el = time.time() - t0
        pbar.set_postfix(done=done, s_per_batch=f"{el/(bi+1):.1f}",
                         eta_min=f"{el/done*(len(texts)-done)/60:.1f}")

    el = time.time() - t0
    print(f"   生成完成：{len(texts)} 条，用时 {el/60:.1f} 分钟（{el/len(texts)*1000:.0f} ms/条）")
    return answers, el


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N_EVAL)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    script_dir = Path(__file__).resolve().parent
    model_path = script_dir / "Qwen3-0.6B"

    train_path = DATA_DIR / "train_extract.jsonl"
    eval_path = DATA_DIR / "dev_extract_clean.jsonl"      # ★ 带标注的抽取验证集
    schema_path = DATA_DIR / "event_schema.json"
    output_name = ("llm_few_shot.jsonl" if args.n == N_EVAL
                   else f"llm_few_shot_sample{args.n}.jsonl")
    output_path = ROOT_DIR / "output" / "bilstm_crf" / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"❌ 找不到本地模型目录: {model_path}")
        return

    schema = json.load(open(schema_path, encoding="utf-8"))
    system_prompt = build_system_prompt(schema)

    print(f"✅ 当前使用的设备: {DEVICE}")
    print("1. 加载模型与 few-shot 示例...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
    ).to(DEVICE)

    examples = load_few_shot(train_path)
    print(f"   已选 {len(examples)} 个示例，事件类型: {[e[1]['type'] for e in examples]}")

    print(f"2. 读取验证集前 {args.n} 条（{eval_path.name}）...")
    rows = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(rows) >= args.n:
                break
            if line.strip():
                rows.append(json.loads(line))
    n_gt = sum(1 for r in rows if r.get("event"))
    print(f"   句子数: {len(rows)} | 含事件: {n_gt}")

    print(f"3. 开始推理（batch={args.batch}, max_new_tokens={args.max_new_tokens}）...")
    answers, elapsed = generate_answers(
        model, tokenizer, system_prompt, examples, [r["text"] for r in rows],
        batch_size=args.batch,
    )

    print("4. 解析结果并计算指标...")
    stats = {"trig": [0, 0, 0], "arg": [0, 0, 0], "event": [0, 0, 0]}
    results = []
    bad_json = 0

    for r, ans in zip(rows, answers):
        text = r["text"]
        obj = extract_json(ans)
        if obj is None:
            bad_json += 1
            obj = {"events": []}
        preds = json_to_events(obj, text)
        gt = [r["event"]] if r.get("event") else []

        gt_t = Counter((g["trigger"]["text"], g["trigger"]["start"]) for g in gt)
        pr_t = Counter((e["trigger"]["text"], e["trigger"]["start"]) for e in preds)
        tp = sum((gt_t & pr_t).values())
        stats["trig"][0] += tp
        stats["trig"][1] += sum(pr_t.values()) - tp
        stats["trig"][2] += sum(gt_t.values()) - tp

        gt_a = Counter((a["role"], a["text"], a["start"]) for g in gt for a in g.get("args", []))
        pr_a = Counter((a["role"], a["text"], a["start"]) for e in preds for a in e.get("args", []))
        tp = sum((gt_a & pr_a).values())
        stats["arg"][0] += tp
        stats["arg"][1] += sum(pr_a.values()) - tp
        stats["arg"][2] += sum(gt_a.values()) - tp

        def key(e):
            return (e["trigger"]["text"], e["trigger"]["start"],
                    frozenset((a["role"], a["text"]) for a in e.get("args", [])))
        gt_e = Counter(key(g) for g in gt)
        pr_e = Counter(key(e) for e in preds)
        tp = sum((gt_e & pr_e).values())
        stats["event"][0] += tp
        stats["event"][1] += sum(pr_e.values()) - tp
        stats["event"][2] += sum(gt_e.values()) - tp

        results.append({"sent_id": r["sent_id"], "text": text,
                        "pred_events": preds, "true_events": gt, "raw_response": ans})

    t_p, t_r, t_f1 = prf(*stats["trig"])
    a_p, a_r, a_f1 = prf(*stats["arg"])
    e_p, e_r, e_f1 = prf(*stats["event"])

    metrics = {
        "Trigger": {"Precision": t_p, "Recall": t_r, "F1": t_f1},
        "Argument": {"Precision": a_p, "Recall": a_r, "F1": a_f1},
        "Event": {"Precision": e_p, "Recall": e_r, "F1": e_f1},
    }

    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.write(json.dumps({"global_metrics": metrics,
                                "meta": {"model": "Qwen3-0.6B (few-shot)", "n_eval": len(rows),
                                         "temperature": 0, "bad_json": bad_json,
                                         "elapsed_sec": round(elapsed, 1),
                                         "dataset": eval_path.name}},
                               ensure_ascii=False) + "\n")
        for res in results:
            f_out.write(json.dumps(res, ensure_ascii=False) + "\n")

    print("\n---- 预测示例 ----")
    shown = 0
    for res in results:
        if not res["true_events"] or shown >= 2:
            continue
        print(f"[{shown + 1}] {res['text'][:50]}")
        print(f"    GT  : {res['true_events'][0]['trigger']['text']!r}")
        if res["pred_events"]:
            pe = res["pred_events"][0]
            print(f"    PRED: {pe['trigger']['text']!r} "
                  f"args={[(a['role'], a['text']) for a in pe.get('args', [])][:4]}")
        else:
            print("    PRED: <无>")
        shown += 1

    print("\n" + "=" * 50)
    print("✅ Qwen3-0.6B Few-Shot 验证完成！")
    print(f" - Trigger F1 : {t_f1:.4f} (P: {t_p:.4f}, R: {t_r:.4f})")
    print(f" - Argument F1: {a_f1:.4f} (P: {a_p:.4f}, R: {a_r:.4f})")
    print(f" - Event F1   : {e_f1:.4f} (P: {e_p:.4f}, R: {e_r:.4f})")
    print(f" - JSON 解析失败: {bad_json}/{len(rows)}")
    print(f" - 耗时: {elapsed/60:.1f} 分钟（{elapsed/len(rows)*1000:.0f} ms/条）")
    print(f"\n📁 已保存至: {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
