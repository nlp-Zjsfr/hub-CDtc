# -*- coding: utf-8 -*-
"""
DeepSeek API Few-Shot 事件抽取验证（统一口径版）

相对旧版的改动（旧版数字不可用，见文末说明）：
  1. 【评估集】旧版读 dev_detect_clean.jsonl 前 N 行（混了大量无 GT 的负样本，
     前 200 行只有 39 条有标注）。现改为 dev_extract_clean.jsonl 前 200 条，
     全部带标注，与 BERT / biLSTM / LLM+LoRA 的评估口径一致。
  2. 【指标】旧版 `fn += len(true_events)` 把已命中的也计入 FN，导致
     Recall 上限只有 0.5、F1 上限只有 0.667。现改为 `fn = gt - tp`，
     并且和其它模型一样按 (text, start) 严格匹配（offset 由文本反查）。
  3. 【提示词】旧版只给了 3 个示例，没有列出 62 个论元角色名，模型只能瞎猜
     （DeepSeek 预测的角色只有 23% 落在真实 schema 内）。现把 event_schema.json
     的类型→角色映射写进 system prompt，两个模型完全共享同一份提示词。
  4. 【解码】显式 temperature=0（旧版没设，DeepSeek API 默认 1.0 采样，
     每次跑结果都不一样）；响应为空时重试。

用法：
    set DEEPSEEK_API_KEY=sk-xxx
    python api_few_shot.py
    python api_few_shot.py --n 200 --model_name deepseek-chat
"""

import os
import re
import json
import time
import random
import argparse
from pathlib import Path
from collections import defaultdict, Counter

import requests
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"

N_EVAL = 200
SEED = 42
TEMPERATURE = 0
MAX_TOKENS = 1024
SLEEP_BETWEEN = 0.3

JSON_SCHEMA = ('{"events":[{"type":"事件类型","trigger":{"text":"触发词"},'
               '"args":[{"role":"角色","text":"论元"}]}]}')


# ================= 提示词（与 llm_few_shot.py 完全一致） =================
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
    """固定种子挑 k 个不同事件类型的示例（两个脚本一致）。"""
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


# ================= 事件还原（offset 由文本反查，与 LLM 版一致） =================
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


# ================= API 调用 =================
def call_deepseek_api(api_key, system_prompt, messages, model_name, max_retries=4):
    url = "https://api.deepseek.com/anthropic/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model_name,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,     # ★ 关键：贪心，保证可复现
        "system": system_prompt,
        "messages": messages,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                for block in data.get("content", []):
                    if block.get("type") == "text" and block.get("text", "").strip():
                        return block["text"].strip()
                time.sleep(0.5)          # 空返回，重试
                continue
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            print(f"\n[API 错误] {resp.status_code}: {resp.text[:200]}")
            if resp.status_code in (400, 401, 403):
                return ""
        except Exception as e:
            print(f"\n[网络异常] {type(e).__name__}: {e}")
            time.sleep(1)
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", default=os.environ.get("DEEPSEEK_API_KEY", ""))
    parser.add_argument("--model_name", default="deepseek-flash")
    parser.add_argument("--n", type=int, default=N_EVAL)
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("❌ 请通过 --api_key 传入密钥，或设置环境变量 DEEPSEEK_API_KEY")

    train_path = DATA_DIR / "train_extract.jsonl"
    eval_path = DATA_DIR / "dev_extract_clean.jsonl"      # ★ 改成带标注的抽取验证集
    schema_path = DATA_DIR / "event_schema.json"
    output_name = ("api_few_shot.jsonl" if args.n == N_EVAL
                   else f"api_few_shot_sample{args.n}.jsonl")
    output_path = ROOT_DIR / "output" / "bilstm_crf" / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    schema = json.load(open(schema_path, encoding="utf-8"))
    system_prompt = build_system_prompt(schema)

    print("1. 准备 few-shot 示例...")
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

    print(f"3. 调用 API（model={args.model_name}, temperature={TEMPERATURE}）...")
    stats = {"trig": [0, 0, 0], "arg": [0, 0, 0], "event": [0, 0, 0]}
    results = []
    bad_json = 0
    t0 = time.time()

    for r in tqdm(rows, desc="API 预测"):
        text = r["text"]
        messages = []
        for ex_text, ex_event in examples:
            messages.append({"role": "user", "content": build_user_content(ex_text)})
            messages.append({"role": "assistant", "content": build_answer(ex_event)})
        messages.append({"role": "user", "content": build_user_content(text)})

        raw = call_deepseek_api(args.api_key, system_prompt, messages, args.model_name)
        obj = extract_json(raw)
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
                        "pred_events": preds, "true_events": gt, "raw_response": raw})
        time.sleep(SLEEP_BETWEEN)

    elapsed = time.time() - t0
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
                                "meta": {"model": args.model_name, "n_eval": len(rows),
                                         "temperature": TEMPERATURE, "bad_json": bad_json,
                                         "elapsed_sec": round(elapsed, 1),
                                         "dataset": eval_path.name}},
                               ensure_ascii=False) + "\n")
        for res in results:
            f_out.write(json.dumps(res, ensure_ascii=False) + "\n")

    print("\n" + "=" * 50)
    print("✅ API Few-Shot 验证完成！")
    print(f" - Trigger F1 : {t_f1:.4f} (P: {t_p:.4f}, R: {t_r:.4f})")
    print(f" - Argument F1: {a_f1:.4f} (P: {a_p:.4f}, R: {a_r:.4f})")
    print(f" - Event F1   : {e_f1:.4f} (P: {e_p:.4f}, R: {e_r:.4f})")
    print(f" - JSON 解析失败: {bad_json}/{len(rows)}")
    print(f" - 耗时: {elapsed/60:.1f} 分钟（{elapsed/len(rows)*1000:.0f} ms/条）")
    print(f"\n📁 已保存至: {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
