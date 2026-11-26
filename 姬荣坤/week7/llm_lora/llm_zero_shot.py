# -*- coding: utf-8 -*-
"""
LLM (Qwen3-0.6B) Zero-Shot 事件抽取
仅使用验证集的前 1000 条进行评估。
"""
import os
import re
import json
import torch
import random
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
MAX_NEW_TOKENS = 384
MAX_LEN = 1024
EVAL_SUBSET_SIZE = 1000  # 限制验证数量

SYSTEM_PROMPT = "请你作为一个语言提取助手，帮我提取下面语句中的信息。"
JSON_SCHEMA = ('{"events":[{"type":"事件类型","trigger":{"text":"触发词"},'
               '"args":[{"role":"角色","text":"论元"}]}]}')

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_user_content(text):
    return (f"{text}\n\n"
            f"要求：只输出一个JSON对象，不要输出任何解释、Markdown代码块或多余文字。\n"
            f"输出格式：{JSON_SCHEMA}；如果没有事件，输出 {{\"events\":[]}}。")

def build_prompt_ids(tokenizer, text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_content(text)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
    )

def extract_json(text):
    if not text: return None
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s).strip()
    start = s.find("{")
    if start < 0: return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
        else:
            if ch == '"': in_str = True
            elif ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try: return json.loads(s[start:i + 1])
                    except json.JSONDecodeError: return None
    return None

@torch.no_grad()
def generate_answers(model, tokenizer, texts):
    model.eval()
    try:
        gc = model.generation_config
        gc.do_sample = False
        gc.temperature = None
        gc.top_p = None
        gc.top_k = None
    except Exception: pass
    
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    answers = []
    try:
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="推理进度"):
            chunk = texts[i:i + BATCH_SIZE]
            prompts = [build_prompt_ids(tokenizer, t) for t in chunk]
            enc = tokenizer.pad({"input_ids": prompts}, padding=True, return_tensors="pt")
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            
            out = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
            )
            new_tokens = out[:, enc["input_ids"].shape[1]:]
            answers.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    finally:
        tokenizer.padding_side = old_side
    return answers

def calculate_f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return round(p, 4), round(r, 4), round(f1, 4)

def main():
    set_seed()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    
    model_path = os.path.abspath(os.path.join(script_dir, "Qwen3-0.6B"))
    dev_path = os.path.join(root_dir, "data", "processed", "dev_detect_clean.jsonl")
    gt_path = os.path.join(root_dir, "data", "processed", "dev_extract_clean.jsonl")
    
    output_dir = os.path.join(root_dir, "output", "bilstm_crf")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "llm_zero_shot.jsonl")

    print(f"✅ 当前使用的设备: {DEVICE}")
    print("1. 加载模型与数据...")
    if not os.path.exists(model_path):
        print(f"❌ 错误：找不到本地模型目录 '{model_path}'")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
    ).to(DEVICE)

    # 只读取前 EVAL_SUBSET_SIZE 条验证数据
    texts, sent_ids = [], []
    with open(dev_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= EVAL_SUBSET_SIZE: break
            if line.strip():
                item = json.loads(line)
                texts.append(item["text"])
                sent_ids.append(item["sent_id"])
    
    sent_id_set = set(sent_ids)
    gt_data = defaultdict(list)
    with open(gt_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item.get("sent_id") in sent_id_set and item.get("event"):
                    gt_data[item["sent_id"]].append(item["event"])

    print(f"   验证集样本数(截断后): {len(texts)}")
    print("2. 开始 Zero-Shot 推理...")
    answers = generate_answers(model, tokenizer, texts)

    print("3. 解析结果并计算指标...")
    stats = {"trig": {"tp":0, "fp":0, "fn":0}, "arg": {"tp":0, "fp":0, "fn":0}, "event": {"tp":0, "fp":0, "fn":0}}
    results = []

    for sent_id, text, ans in zip(sent_ids, texts, answers):
        obj = extract_json(ans) or {"events": []}
        pred_event = None
        if obj.get("events"):
            pred_event = obj["events"][0]
            
        true_events = gt_data.get(sent_id, [])

        if pred_event:
            p_trig = pred_event.get("trigger", {}).get("text", "")
            is_trig_match = any(gt["trigger"]["text"] == p_trig for gt in true_events)
            if is_trig_match: stats["trig"]["tp"] += 1
            else: stats["trig"]["fp"] += 1

            pred_args = {(a.get("role"), a.get("text")) for a in pred_event.get("args", [])}
            gt_args = set()
            for gt in true_events:
                for a in gt.get("args", []):
                    gt_args.add((a["role"], a["text"]))
            
            tp_a = len(pred_args.intersection(gt_args))
            stats["arg"]["tp"] += tp_a
            stats["arg"]["fp"] += len(pred_args) - tp_a
            stats["arg"]["fn"] += len(gt_args) - tp_a

            is_event_match = any(
                gt["trigger"]["text"] == p_trig and 
                {(a["role"], a["text"]) for a in gt.get("args", [])} == pred_args
                for gt in true_events
            )
            if is_event_match: stats["event"]["tp"] += 1
            else: stats["event"]["fp"] += 1
        else:
            pass

        if true_events:
            stats["trig"]["fn"] += len(true_events)
            stats["event"]["fn"] += len(true_events)
            if not pred_event:
                for gt in true_events: stats["arg"]["fn"] += len(gt.get("args", []))

        results.append({"sent_id": sent_id, "text": text, "pred_event": pred_event, "true_events": true_events})

    t_p, t_r, t_f1 = calculate_f1(stats["trig"]["tp"], stats["trig"]["fp"], stats["trig"]["fn"])
    a_p, a_r, a_f1 = calculate_f1(stats["arg"]["tp"], stats["arg"]["fp"], stats["arg"]["fn"])
    e_p, e_r, e_f1 = calculate_f1(stats["event"]["tp"], stats["event"]["fp"], stats["event"]["fn"])

    metrics = {
        "Trigger": {"Precision": t_p, "Recall": t_r, "F1": t_f1},
        "Argument": {"Precision": a_p, "Recall": a_r, "F1": a_f1},
        "Event": {"Precision": e_p, "Recall": e_r, "F1": e_f1}
    }

    with open(output_path, 'w', encoding='utf-8') as f_out:
        f_out.write(json.dumps({"global_metrics": metrics}, ensure_ascii=False) + "\n")
        for res in results:
            f_out.write(json.dumps(res, ensure_ascii=False) + "\n")

    print("\n" + "="*50)
    print("✅ Zero-Shot 验证完成！全局指标如下：")
    print(f" - Trigger F1 : {t_f1:.4f} (P: {t_p:.4f}, R: {t_r:.4f})")
    print(f" - Argument F1: {a_f1:.4f} (P: {a_p:.4f}, R: {a_r:.4f})")
    print(f" - Event F1   : {e_f1:.4f} (P: {e_p:.4f}, R: {e_r:.4f})")
    print(f"\n📁 详细结果已保存至: {output_path}")
    print("="*50)

if __name__ == "__main__":
    main()