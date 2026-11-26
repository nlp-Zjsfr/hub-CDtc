# -*- coding: utf-8 -*-
"""
week7 抽取实验：换模型 / API / 数据集，只改命令行参数。

一个脚本干完：读数据 → 构造提示词 → 调模型 → 解析 → 评测 → 落盘。
（仿 week06 参考答案：控制变量只换参数，跑两遍再对比。）

用法：
    # 1. 离线冒烟（不联网，用内置小样本验证脚本能跑）
    python extract.py --dataset cmeee --provider mock --n 16

    # 1b. 真·离线实验：本地 Qwen3-0.6B（不联网、不要 key，可写进报告）
    python extract.py --dataset duee_fin --provider local --n 50

    # 2. DuEE-Fin 金融事件抽取：换 API / 换模型 / 换提示词策略
    python extract.py --dataset duee_fin --provider deepseek --model deepseek-chat --mode zero
    python extract.py --dataset duee_fin --provider deepseek --model deepseek-chat --mode few
    python extract.py --dataset duee_fin --provider qwen --model qwen-plus --mode zero

    # 3. 换数据集（同一套代码，提示词由 schema 自动生成）
    python extract.py --dataset cmeee --provider deepseek --mode zero

环境变量（任选一种）：
    DEEPSEEK_API_KEY / DASHSCOPE_API_KEY / ZHIPU_API_KEY / OPENAI_API_KEY
    只用一个厂商时直接设通用变量：LLM_API_KEY

输出（都在 out/ 下）：
    <tag>_log.json    汇总指标 + 耗时 + token + 格式合规率
    <tag>_pred.json   逐条预测（给 compare.py 用）
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent      # 脚本所在的 week7/ 目录

# ── 数据 / 配置 ──────────────────────────────────────────────────────────

PROVIDERS = {
    # provider: (base_url, 默认模型, 该厂商支持的环境变量名列表)
    "deepseek": ("https://api.deepseek.com", "deepseek-chat",
                 ["DEEPSEEK_API_KEY", "LLM_API_KEY"]),
    "qwen": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus",
             ["DASHSCOPE_API_KEY", "LLM_API_KEY"]),
    "glm": ("https://open.bigmodel.cn/api/paas/v4", "glm-4-flash",
            ["ZHIPU_API_KEY", "LLM_API_KEY"]),
    "moonshot": ("https://api.moonshot.cn/v1", "moonshot-v1-8k",
                 ["MOONSHOT_API_KEY", "LLM_API_KEY"]),
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini",
               ["OPENAI_API_KEY", "LLM_API_KEY"]),
    "ollama": ("http://localhost:11434/v1", "qwen2.5:7b", ["OLLAMA_API_KEY"]),
    "local": ("", "Qwen3-0.6B", []),   # 本地模型：不联网、不要 key
    "mock": ("", "mock", []),      # 离线假模型，只为打通流程，数字无意义
}

# CMeEE 的 9 类实体（官方类别）
CMEEE_TYPES = ["检查", "治疗", "疾病", "药物", "症状", "解剖部位", "身体部位", "科室", "临床表现"]


def find_processed():
    """仿 week06 的 find_code_root：向上逐级找 week7 教学项目已构建好的数据。"""
    for p in [ROOT] + list(ROOT.parents):
        for rel in ("week7-信息抽取/code/data/processed", "code/data/processed"):
            cand = p / rel
            if (cand / "dev_extract.jsonl").exists():
                return cand
    return None


def read_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_json(obj, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_dataset(name, n=0):
    """
    返回 (rows, schema, has_trigger)

    rows 统一格式：
      {"id","sent_id","text",
       "events":[{"type","trigger":{"text"}, "args":[{"role","text"}]}]}
    """
    if name == "cmeee":
        # CMeEE 用作业包内置的小样本（离线可跑；完整数据需自行下载）
        raw = read_jsonl(ROOT / "data" / "cmeee_mini.jsonl")
        rows, schema, has_trigger = [], {t: [] for t in CMEEE_TYPES}, False
        for i, r in enumerate(raw):
            args = []
            for a in r.get("spans") or []:
                args.append({"role": a["role"], "text": a["text"]})
            rows.append({"id": r.get("id", f"cmeee_{i}"),
                         "sent_id": r.get("id", f"cmeee_{i}"),
                         "text": r["text"],
                         "events": [{"type": "实体集合", "trigger": {"text": ""},
                                     "args": args}] if args else []})
        return (rows[:n] if n else rows), schema, has_trigger

    # DuEE-Fin：从 week7 教学项目读（与 week06 从公共目录读数据的方式一致）
    base = find_processed()
    if base is None:
        raise FileNotFoundError(
            "找不到 DuEE-Fin 数据。期望在 week7-信息抽取/code/data/processed/ 下。"
            "离线练习请用 --dataset cmeee（自带小样本）。")
    schema = json.loads((base / "event_schema.json").read_text(encoding="utf-8"))
    schema = {k: v for k, v in schema.items() if not k.startswith("__")}
    rows = []
    for r in read_jsonl(base / "dev_extract.jsonl"):
        ev = r.get("event") or {}
        trig = ev.get("trigger") or {}
        args = [{"role": a.get("role", ""), "text": a.get("text", "")}
                for a in ev.get("args") or [] if a.get("text")]
        rows.append({"id": r.get("id", ""), "sent_id": r.get("sent_id", ""),
                     "text": r.get("text", ""),
                     "events": [{"type": ev.get("type", ""),
                                 "trigger": {"text": trig.get("text", "")},
                                 "args": args}] if ev.get("type") else []})
    return (rows[:n] if n else rows), schema, True


# ── 提示词（由 schema 生成，换数据集不用改代码）────────────────────────────

def system_prompt(schema, has_trigger):
    if has_trigger:
        lines = "\n".join(f"- {t}：" + "、".join(roles) for t, roles in sorted(schema.items()))
        return ("你是金融事件抽取专家。输入是一句金融文本，抽取其中发生的一个事件。\n"
                f"事件类型与论元角色（共{len(schema)}类）：\n{lines}\n"
                "要求：类型必须逐字照抄；触发词和论元必须是原文中逐字截取的连续片段；"
                "原文没有的角色不要猜；不要输出字符下标（程序用 text.find 对齐）。\n"
                '只输出一个 JSON 对象，不要解释、不要代码块：\n'
                '{"events":[{"type":"类型","trigger":{"text":"触发词"},'
                '"arguments":[{"role":"角色","argument":"论元文本"}]}]}')
    lines = "\n".join(f"- {t}" for t in sorted(schema))
    return ("你是中文医疗实体识别专家。抽取句子中出现的所有实体。\n"
            f"实体类别（共{len(schema)}类）：\n{lines}\n"
            "要求：类别逐字照抄；实体必须是原文逐字截取的连续片段；不要输出字符下标。\n"
            '只输出一个 JSON 对象，不要解释、不要代码块：\n'
            '{"entities":[{"type":"类别","text":"实体文本"}]}')


def build_messages(row, schema, has_trigger, mode, shots):
    msgs = [{"role": "system", "content": system_prompt(schema, has_trigger)}]
    if mode == "few":
        for ex in shots:
            msgs.append({"role": "user", "content": ex["text"]})
            msgs.append({"role": "assistant", "content": answer_text(ex, has_trigger)})
    msgs.append({"role": "user", "content": row["text"]})
    return msgs


def answer_text(row, has_trigger):
    """把金标行变成 few-shot 的标准答案文本。"""
    if not has_trigger:
        ents = [{"type": a["role"], "text": a["text"]}
                for e in row["events"] for a in e["args"]]
        return json.dumps({"entities": ents}, ensure_ascii=False)
    evs = [{"type": e["type"], "trigger": {"text": e["trigger"]["text"]},
            "arguments": [{"role": a["role"], "argument": a["text"]} for a in e["args"]]}
           for e in row["events"]]
    return json.dumps({"events": evs}, ensure_ascii=False)


def pick_shots(rows, k=4):
    """从训练集抽 few-shot 示例（这里直接用 dev 的前几条做演示；正式实验应从 train 抽）。"""
    return [r for r in rows if r["events"]][:k]


# ── 调用模型 ─────────────────────────────────────────────────────────────

def find_pretrain():
    """向上逐级找课程公共的 pretrain_models/Qwen3-0.6B（仿 find_processed）。"""
    for p in [ROOT] + list(ROOT.parents):
        cand = p / "pretrain_models" / "Qwen3-0.6B"
        if (cand / "config.json").exists():
            return cand
    return None


_LOCAL = {}


def call_local(msgs, max_tokens):
    """本地 Qwen3-0.6B 生成：不联网、不要 key，和 API 共用同一套提示词与评测。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if "model" not in _LOCAL:
        path = find_pretrain()
        if path is None:
            raise FileNotFoundError(
                "找不到本地模型 Qwen3-0.6B。请把模型放到课程根目录 "
                "pretrain_models/Qwen3-0.6B/ 下（config.json 必须在），"
                "或先用 --provider mock 验证流程。")
        dt = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        tok = AutoTokenizer.from_pretrained(str(path))
        try:
            m = AutoModelForCausalLM.from_pretrained(str(path), dtype=dt)
        except TypeError:      # 老版本 transformers 只认 torch_dtype
            m = AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=dt)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m.to(dev)
        m.eval()
        _LOCAL.update(model=m, tok=tok, dev=dev)

    tok, m, dev = _LOCAL["tok"], _LOCAL["model"], _LOCAL["dev"]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = m.generate(**enc, max_new_tokens=max_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return text, "", 0, 0


def call_api(msgs, provider, model, max_tokens):
    """调一次模型；返回 (文本, 错误, prompt_tokens, completion_tokens)。"""
    if provider == "mock":
        # 假模型：只为验证流水线，数字无意义
        return '{"events": []}', "", 0, 0
    if provider == "local":
        return call_local(msgs, max_tokens)

    base_url, default_model, env_keys = PROVIDERS[provider]
    key = ""
    for k in env_keys:
        key = os.getenv(k, "").strip()
        if key:
            break
    if not key:
        raise EnvironmentError(
            f"没找到 {provider} 的 API key。请设置环境变量：{', '.join(env_keys)} "
            f"（PowerShell: $env:{env_keys[0]}='sk-xxx'）")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("缺少 openai SDK：pip install openai") from exc

    client = OpenAI(api_key=key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model or default_model,
        messages=msgs,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    usage = getattr(resp, "usage", None)
    return (resp.choices[0].message.content or "").strip(), "", \
           getattr(usage, "prompt_tokens", 0) if usage else 0, \
           getattr(usage, "completion_tokens", 0) if usage else 0


# ── 解析输出 ─────────────────────────────────────────────────────────────

def parse_output(raw, text, has_trigger):
    """
    模型输出 → 标准事件列表。

    容错三件事（实测各有 5%~20% 概率发生）：
      剥 markdown 围栏 / 从废话里抠出 JSON / 括号或截断时尽力补。
    修不回来就记 parse_fail，**该行照样写进预测文件（events=[]）**，
    不能删行 —— 删了分母变小，F1 会虚高。
    """
    info = {"json_ok": False, "repaired": False}
    s = (raw or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
        info["repaired"] = True
    start = s.find("{")
    if start >= 0:
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
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    s = s[start:i + 1]
                    break
    obj = None
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        fixed = re.sub(r",\s*([}\]])", r"\1", s)
        for ch in reversed("}]"):
            if fixed.count(ch) > fixed.count("{" if ch == "}" else "["):
                fixed += ch
        try:
            obj = json.loads(fixed)
            info["repaired"] = True
        except json.JSONDecodeError:
            return [], info
    info["json_ok"] = True

    def align(span):
        idx = text.find(span or "")
        return {"text": span, "start": idx, "end": idx + len(span) if idx >= 0 else -1}

    if not has_trigger:
        ents = obj.get("entities") or obj.get("spans") or []
        args = []
        for e in ents:
            if not isinstance(e, dict):
                continue
            atext = str(e.get("text", "")).strip()
            if atext:
                args.append({"role": str(e.get("type", "")).strip(), **align(atext)})
        return ([{"type": "实体集合", "trigger": {"text": "", "start": -1, "end": -1},
                  "args": args}] if args else []), info

    evs = obj.get("events") or []
    out = []
    for e in evs:
        if not isinstance(e, dict):
            continue
        trig = e.get("trigger") or {}
        trig_text = trig.get("text", "") if isinstance(trig, dict) else str(trig)
        trig_text = str(trig_text).strip()
        if not trig_text:
            continue
        args = []
        for a in e.get("arguments") or e.get("args") or []:
            if not isinstance(a, dict):
                continue
            atext = str(a.get("argument", a.get("text", ""))).strip()
            if atext:
                args.append({"role": str(a.get("role", "")).strip(), **align(atext)})
        out.append({"type": str(e.get("type", "")).strip(),
                    "trigger": align(trig_text), "args": args})
    return out, info


# ── 评测（四级指标，与 week06 一样统一口径）──────────────────────────────

def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"p": round(p, 4), "r": round(r, 4), "f1": round(f, 4)}


def evaluate(gold, pred, has_trigger):
    """entity 是跨数据集可比的主指标；trigger/event 只在事件型数据集上有意义。"""
    pmap = {str(r["id"]) + "|" + str(r["sent_id"]): r for r in pred}
    en = tr = ev = (0, 0, 0)
    halluc, n_pred = 0, 0

    for g in gold:
        key = str(g["id"]) + "|" + str(g["sent_id"])
        p = pmap.get(key) or {"events": []}
        ge = g["events"]
        pe = p["events"]
        g_args = Counter((a["role"], a["text"]) for e in ge for a in e["args"])
        p_args = Counter((a["role"], a["text"]) for e in pe for a in e["args"])
        t = sum((g_args & p_args).values())
        en = (en[0] + t, en[1] + sum(p_args.values()) - t, en[2] + sum(g_args.values()) - t)

        if has_trigger:
            g_tr = Counter(e["trigger"]["text"] for e in ge if e["trigger"]["text"])
            p_tr = Counter(e["trigger"]["text"] for e in pe if e["trigger"]["text"])
            t = sum((g_tr & p_tr).values())
            tr = (tr[0] + t, tr[1] + sum(p_tr.values()) - t, tr[2] + sum(g_tr.values()) - t)
            # event 完全正确：类型 + 触发词 + 论元集合全对
            for e in ge:
                hit = any(
                    q["type"] == e["type"]
                    and q["trigger"]["text"] == e["trigger"]["text"]
                    and Counter((a["role"], a["text"]) for a in q["args"])
                    == Counter((a["role"], a["text"]) for a in e["args"])
                    for q in pe)
                ev = (ev[0] + int(hit), ev[1] + int(not hit), ev[2] + int(not hit))

        src = g["text"]
        for e in pe:
            for a in e["args"]:
                n_pred += 1
                if a["text"] and a["text"] not in src:
                    halluc += 1
            if e["trigger"]["text"]:
                n_pred += 1
                if e["trigger"]["text"] not in src:
                    halluc += 1

    res = {
        "n": len(gold),
        "entity": prf(*en),
        "hallucination_rate": round(halluc / n_pred, 4) if n_pred else 0.0,
    }
    if has_trigger:
        res["trigger"] = prf(*tr)
        res["event_exact"] = prf(*ev)
    return res


# ── 主流程 ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="week7 抽取实验：换模型 / API / 数据集")
    ap.add_argument("--dataset", default="duee_fin", choices=["duee_fin", "cmeee"])
    ap.add_argument("--provider", default="deepseek", choices=list(PROVIDERS))
    ap.add_argument("--model", default="", help="留空用该 provider 的默认模型")
    ap.add_argument("--mode", default="zero", choices=["zero", "few"])
    ap.add_argument("--n", default=0, type=int, help="只跑前 N 条（0=全量）")
    ap.add_argument("--max_tokens", default=512, type=int)
    ap.add_argument("--tag", default="", help="输出文件名前缀，默认自动生成")
    args = ap.parse_args()

    rows, schema, has_trigger = load_dataset(args.dataset, args.n)
    model = args.model or PROVIDERS[args.provider][1]
    tag = args.tag or f"{args.dataset}_{args.provider}_{model}_{args.mode}"
    shots = pick_shots(rows) if args.mode == "few" else []

    print(f"[{tag}] {len(rows)} 条样本，provider={args.provider}，model={model}，"
          f"mode={args.mode}，few_shot={len(shots)}")

    pred, stats = [], Counter()
    p_tok = c_tok = 0
    t0 = time.time()
    for i, r in enumerate(rows):
        msgs = build_messages(r, schema, has_trigger, args.mode, shots)
        try:
            raw, err, pt, ct = call_api(msgs, args.provider, model, args.max_tokens)
        except Exception as exc:      # noqa: BLE001
            raw, err, pt, ct = "", f"{type(exc).__name__}: {exc}", 0, 0
        events, info = parse_output(raw, r["text"], has_trigger)
        for k, v in info.items():
            if isinstance(v, bool):
                stats[k] += int(v)
        p_tok += pt
        c_tok += ct
        if err and i == 0:
            print(f"  [警告] 第一条调用失败：{err}（后续可能同样失败）")
        pred.append({"id": r["id"], "sent_id": r["sent_id"], "text": r["text"],
                     "events": events})
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  {i + 1}/{len(rows)}  (耗时 {time.time() - t0:.1f}s)", flush=True)

    metrics = evaluate(rows, pred, has_trigger)
    elapsed = time.time() - t0
    log = {
        "tag": tag,
        "dataset": args.dataset,
        "task_type": "event" if has_trigger else "entity",
        "provider": args.provider,
        "model": model,
        "mode": args.mode,
        "n": len(rows),
        "metrics": metrics,
        "json_valid": stats["json_ok"],
        "repaired": stats["repaired"],
        "parse_fail": len(rows) - stats["json_ok"],
        "prompt_tokens": p_tok,
        "completion_tokens": c_tok,
        "total_tokens": p_tok + c_tok,
        "elapsed_s": round(elapsed, 1),
        "is_mock": args.provider == "mock",
    }
    out = ROOT / "out"
    write_json(log, out / f"{tag}_log.json")
    write_json(pred, out / f"{tag}_pred.json")

    print("\n" + "=" * 66)
    print(f"结果：{tag}")
    print("=" * 66)
    e = metrics["entity"]
    print(f"  entity F1  : {e['f1']:.4f}  (P={e['p']:.4f} R={e['r']:.4f})")
    if has_trigger:
        print(f"  trigger F1 : {metrics['trigger']['f1']:.4f}")
        print(f"  event F1   : {metrics['event_exact']['f1']:.4f}")
    print(f"  幻觉率     : {metrics['hallucination_rate']:.4f}")
    print(f"  JSON 合法  : {stats['json_ok']}/{len(rows)}（修复 {stats['repaired']}）")
    print(f"  耗时/token : {elapsed:.1f}s / {p_tok + c_tok}")
    if args.provider == "mock":
        print("  ⚠ mock 后端：数字仅证明流水线通了，不能写进结论。")
    print(f"\n结果 → out/{tag}_log.json")


if __name__ == "__main__":
    main()
