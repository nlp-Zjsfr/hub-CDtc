# coding=utf-8
"""
Qwen3-0.6B 生成式 NER（conllpp）：原始模型 / LoRA / 全量微调

任务：一句英文 → JSON 实体列表 {"entities":[{"text":..,"type":..}]}
评测口径与 bert/train.py、lstm/train.py 完全一致：**词级 span 精确匹配**（类型+边界都对）。
生成式模型输出的是文本，所以要把 text 反查回句中的词区间（locate_spans），
查不回来的预测单独统计为 unmatched——这是生成式 NER 特有的失败模式。

用法：
  python llm/ner_llm.py --mode base                    # 原始模型，零样本（不训练）
  python llm/ner_llm.py --mode lora                    # LoRA 微调 + 评测
  python llm/ner_llm.py --mode full                    # 全量微调 + 评测
  python llm/ner_llm.py --mode lora --num_train 500 --epochs 1 --max_eval 50   # 冒烟
产物：output/llm/model/<mode>/、output/llm/log/{metrics,log,console}_<mode>.*
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

HOMEWORK_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = HOMEWORK_DIR / "data" / "conllpp"
MODEL_ROOT = HOMEWORK_DIR / "output" / "llm" / "model"
LOG_DIR = HOMEWORK_DIR / "output" / "llm" / "log"
BASE_MODEL_DIR = HOMEWORK_DIR / "pretrain_models" / "Qwen3-0.6B"

TYPES = ["PER", "LOC", "ORG", "MISC"]
IGNORE_INDEX = -100
SYSTEM_PROMPT = (
    "你是命名实体识别助手。从给定句子中抽取命名实体，实体类型只有四种："
    "PER(人名)、LOC(地名)、ORG(组织)、MISC(其他专名)。"
    "只输出 JSON，不要解释。格式：{\"entities\":[{\"text\":\"实体原文\",\"type\":\"PER\"}]}；"
    "没有实体时输出 {\"entities\":[]}。"
)


# ─────────────────────────── 数据 ───────────────────────────

def read_conll(path: Path) -> list[dict]:
    rows, tokens, tags = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if tokens:
                    rows.append({"words": tokens, "tags": tags})
                    tokens, tags = [], []
                continue
            parts = line.split()
            if len(parts) != 4 or parts[0] == "-DOCSTART-":
                continue
            tokens.append(parts[0])
            tags.append(parts[3])
    if tokens:
        rows.append({"words": tokens, "tags": tags})
    return rows


def spans_from_tags(tags: list[str]) -> list[tuple[int, int, str]]:
    """BIO → (start, end, type)，左闭右开；孤立 I- 当 B- 修复（与判别式脚本同口径）。"""
    spans, s, t = [], -1, None
    for i, tag in enumerate(tags + ["O"]):
        if tag.startswith("B-"):
            if s >= 0:
                spans.append((s, i, t))
            s, t = i, tag[2:]
        elif tag.startswith("I-"):
            if s < 0:
                s, t = i, tag[2:]
            elif t != tag[2:]:
                spans.append((s, i, t))
                s, t = i, tag[2:]
        else:
            if s >= 0:
                spans.append((s, i, t))
            s, t = -1, None
    return spans


def target_json(row: dict) -> str:
    """gold spans → 目标 JSON（按出现顺序，不漏不重）。"""
    ents = [{"text": " ".join(row["words"][s:e]), "type": t}
            for s, e, t in spans_from_tags(row["tags"])]
    return json.dumps({"entities": ents}, ensure_ascii=False)


def user_text(row: dict) -> str:
    return "句子：" + " ".join(row["words"])


class SFTDataset(Dataset):
    """prompt 部分 labels=-100，只对 JSON 输出算 loss（同 code/llm_lora 的做法）。"""

    def __init__(self, rows, tok, max_length=384):
        self.samples, self.n_truncated = [], 0
        for row in rows:
            prompt = tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": user_text(row)}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            p_ids = tok.encode(prompt, add_special_tokens=False)
            r_ids = tok.encode(target_json(row), add_special_tokens=False) + [tok.eos_token_id]
            ids = (p_ids + r_ids)[:max_length]
            labels = ([-100] * len(p_ids) + r_ids)[:max_length]
            if len(p_ids) + len(r_ids) > max_length:
                self.n_truncated += 1
            if all(x == IGNORE_INDEX for x in labels):
                continue
            self.samples.append({"input_ids": torch.tensor(ids, dtype=torch.long),
                                 "labels": torch.tensor(labels, dtype=torch.long)})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def collate_train(batch, pad_id):
    L = max(len(b["input_ids"]) for b in batch)
    ids, labels = [], []
    for b in batch:
        n = len(b["input_ids"])
        pad = L - n
        ids.append(torch.cat([b["input_ids"], torch.full((pad,), pad_id, dtype=torch.long)]))
        labels.append(torch.cat([b["labels"], torch.full((pad,), IGNORE_INDEX, dtype=torch.long)]))
    ids, labels = torch.stack(ids), torch.stack(labels)
    mask = (labels != IGNORE_INDEX) | (ids != pad_id)
    return {"input_ids": ids, "labels": labels, "attention_mask": mask.long()}


# ─────────────────────────── 生成与解析 ───────────────────────────

def parse_entities(text: str) -> list[dict]:
    """从模型输出里抠出 {"entities":[...]}；容忍 ```json 包裹、前后废话。"""
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return []
    raw = m.group(0)
    for cand in (raw, raw[: raw.rfind("}") + 1]):
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        ents = obj.get("entities", []) if isinstance(obj, dict) else []
        out = []
        for e in ents if isinstance(ents, list) else []:
            if isinstance(e, dict) and isinstance(e.get("text"), str):
                t = str(e.get("type", "")).upper().strip()
                if t in TYPES and e["text"].strip():
                    out.append({"text": e["text"].strip(), "type": t})
        return out
    return []


def _norm(s: str) -> str:
    return re.sub(r"[\s]+", "", s).lower()


def locate_spans(ents: list[dict], words: list[str]) -> tuple[list, int]:
    """把实体文本反查成词区间；查不回来的计入 unmatched。

    两轮匹配：先大小写敏感（避免 "Said" 命中前面的普通词 "said"），
    再退化为忽略大小写+空白的模糊匹配（容忍模型改大小写或加空格）。
    """
    n = len(words)
    used, spans, unmatched = [], [], 0

    def find(t: str, exact: bool):
        # size 上限用整句长度：conllpp 里有 10 个词以上的组织名，卡在 8 会误判为 unmatched
        for size in range(n, 0, -1):
            for i in range(0, n - size + 1):
                if any(not (i + size <= a or i >= b) for a, b in used):
                    continue
                seg = "".join(words[i:i + size])
                if (seg == t) if exact else (_norm(seg) == _norm(t)):
                    return (i, i + size)
        return None

    for e in ents:
        hit = find(e["text"], exact=True) or find(e["text"], exact=False)
        if hit is None:
            unmatched += 1
            continue
        used.append(hit)
        spans.append((hit[0], hit[1], e["type"]))
    return spans, unmatched


@torch.no_grad()
def generate_batch(model, tok, rows, device, max_new_tokens=96, batch_size=4):
    model.eval()
    tok.padding_side = "left"
    preds, lat = [], []
    for st in range(0, len(rows), batch_size):
        chunk = rows[st:st + batch_size]
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": user_text(r)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        t0 = time.perf_counter()
        out = model.generate(ids, attention_mask=mask, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tok.pad_token_id)
        dt = time.perf_counter() - t0
        new = out[:, ids.shape[1]:]
        for j, row in enumerate(chunk):
            txt = tok.decode(new[j], skip_special_tokens=True).strip()
            spans, unmatched = locate_spans(parse_entities(txt), row["words"])
            preds.append({"spans": spans, "unmatched": unmatched, "raw": txt})
            lat.append(dt / len(chunk))
    return preds, lat


def span_metrics(gold_rows, pred_rows) -> dict:
    per = {t: [0, 0, 0] for t in TYPES}
    tp = fp = fn = unmatched = 0
    for g, p in zip(gold_rows, pred_rows):
        gs = set(g)
        ps = set(p["spans"])
        unmatched += p["unmatched"]
        tp += len(gs & ps); fp += len(ps - gs); fn += len(gs - ps)
        for s in gs & ps:
            per[s[2]][0] += 1
        for s in ps - gs:
            per[s[2]][1] += 1
        for s in gs - ps:
            per[s[2]][2] += 1
    prf = lambda a, b, c: (a / (a + b) if a + b else 0.0, a / (a + c) if a + c else 0.0,
                           2 * a / (2 * a + b + c) if 2 * a + b + c else 0.0)
    P, R, F = prf(tp, fp, fn)
    return {"precision": P, "recall": R, "f1": F, "tp": tp, "fp": fp, "fn": fn,
            "unmatched_entities": unmatched,
            "per_type": {t: dict(zip(("precision", "recall", "f1"), prf(*v))) for t, v in per.items()}}


def eval_split(model, tok, rows, device, split_name, batch_size, limit=0):
    sub = rows[:limit] if limit else rows
    t0 = time.perf_counter()
    preds, lat = generate_batch(model, tok, sub, device, batch_size=batch_size)
    gold = [spans_from_tags(r["tags"]) for r in sub]
    met = span_metrics(gold, preds)
    met["split"] = split_name
    met["n"] = len(sub)
    met["ms_per_sentence"] = round(sum(lat) / max(len(lat), 1) * 1000, 1)
    met["elapsed_s"] = round(time.perf_counter() - t0, 1)
    sample = [{"text": " ".join(r["words"]), "gold": gold[i], "pred": preds[i]["spans"],
               "raw": preds[i]["raw"]} for i, r in enumerate(sub[:3])]
    return met, sample, preds, sub


# ─────────────────────────── 训练 ───────────────────────────

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def get_device(req=""):
    if req:
        return torch.device(req)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device, mode):
    """MPS 上 bf16 可用（Apple Silicon），省一半权重/激活显存。"""
    if device.type == "mps":
        return torch.bfloat16
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def train(model, tok, train_rows, args, device, pad_id):
    ds = SFTDataset(train_rows, tok, args.max_length)
    print(f"训练样本 {len(ds)} 条（截断 {ds.n_truncated} 条）", flush=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda b: collate_train(b, pad_id))
    params = [p for p in model.parameters() if p.requires_grad]
    n_train_p = sum(p.numel() for p in params)
    print(f"可训练参数 {n_train_p/1e6:.2f}M / {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)
    if args.opt == "adafactor":
        # 全量微调的内存解法：Adafactor 只存行列统计量，几乎不占显存
        # （AdamW 要为 598M 参数存两份 fp32 状态 ≈ 4.8GB，8GB 机器必 OOM）
        from transformers.optimization import Adafactor
        opt = Adafactor(params, lr=args.lr, scale_parameter=False,
                        relative_step=False, warmup_init=False)
    elif args.opt == "sgd":
        opt = SGD(params, lr=args.lr)
    else:
        opt = AdamW(params, lr=args.lr, weight_decay=0.0)
    model.train()
    history, step = [], 0
    for epoch in range(1, args.epochs + 1):
        tot, nb, t0 = 0.0, 0, time.perf_counter()
        opt.zero_grad()
        for i, batch in enumerate(loader, 1):
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        labels=batch["labels"].to(device))
            loss = out.loss
            (loss / args.grad_accum).backward()
            tot += float(loss.item()); nb += 1
            if i % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad(); step += 1
            if i % 20 == 0:
                print(f"  epoch {epoch} step {i}/{len(loader)} loss={tot/nb:.4f}", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        el = time.perf_counter() - t0
        print(f"epoch {epoch}/{args.epochs} | train_loss={tot/max(nb,1):.4f} | {el:.0f}s", flush=True)
        history.append({"epoch": epoch, "train_loss": round(tot / max(nb, 1), 4),
                        "optimizer_steps": step, "elapsed_s": round(el, 1)})
        if args.max_steps and step >= args.max_steps:
            break
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["base", "lora", "full"], default="base")
    ap.add_argument("--model_path", default=str(BASE_MODEL_DIR))
    ap.add_argument("--data_dir", default=str(DATA_DIR))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--opt", choices=["adamw", "sgd", "adafactor"], default="adamw")
    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--num_train", type=int, default=-1)
    ap.add_argument("--max_eval", type=int, default=0, help=">0 时只评测前 N 句（冒烟）")
    ap.add_argument("--eval_batch", type=int, default=4)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="")
    args = ap.parse_args()
    if args.lr is None:
        args.lr = {"base": 0.0, "lora": 2e-4, "full": 2e-5}[args.mode]

    set_seed(args.seed)
    device = get_device(args.device)
    dtype = pick_dtype(device, args.mode)
    data_dir = Path(args.data_dir)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run = f"{args.mode}" + (f"_n{args.num_train}" if args.num_train > 0 else "")
    print(f"设备 {device} | dtype {dtype} | 模式 {args.mode} | 基座 {args.model_path}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=dtype,
                                                trust_remote_code=True)
    model.config.use_cache = False

    train_rows = read_conll(data_dir / "train.txt")
    dev_rows = read_conll(data_dir / "dev.txt")
    test_rows = read_conll(data_dir / "test.txt")

    out_dir = MODEL_ROOT / run
    history, n_train_p, n_all_p = [], 0, 0

    if args.mode != "base":
        if args.num_train > 0:
            train_rows = train_rows[: args.num_train]
        if args.mode == "lora":
            from peft import LoraConfig, TaskType, get_peft_model
            cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.lora_r,
                             lora_alpha=args.lora_alpha,
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                             lora_dropout=0.05, bias="none")
            model = get_peft_model(model, cfg)
        model.to(device)
        n_all_p = sum(p.numel() for p in model.parameters())
        n_train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"训练 {len(train_rows)} 句 | 可训练 {n_train_p/1e6:.2f}M / 全部 {n_all_p/1e6:.1f}M "
              f"({n_train_p/max(n_all_p,1):.2%})", flush=True)
        t0 = time.perf_counter()
        history = train(model, tok, train_rows, args, device, tok.pad_token_id)
        train_s = time.perf_counter() - t0
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(out_dir))
        tok.save_pretrained(str(out_dir))
        print(f"已保存 → {out_dir}（{train_s/60:.1f} 分钟）", flush=True)
    else:
        model.to(device)
        train_s = 0.0
        n_all_p = sum(p.numel() for p in model.parameters())
        n_train_p = 0

    dev_met, dev_sample, _, _ = eval_split(model, tok, dev_rows, device, "dev",
                                          args.eval_batch, args.max_eval)
    print(f"dev  P={dev_met['precision']:.4f} R={dev_met['recall']:.4f} F1={dev_met['f1']:.4f} "
          f"| unmatched={dev_met['unmatched_entities']} | {dev_met['ms_per_sentence']}ms/句", flush=True)
    test_met, test_sample, _, _ = eval_split(model, tok, test_rows, device, "test",
                                            args.eval_batch, args.max_eval)
    print(f"test P={test_met['precision']:.4f} R={test_met['recall']:.4f} F1={test_met['f1']:.4f} "
          f"| unmatched={test_met['unmatched_entities']} | {test_met['ms_per_sentence']}ms/句", flush=True)

    metrics = {"run": run, "mode": args.mode, "config": vars(args),
               "train_s": round(train_s, 1), "params_m": round(n_all_p / 1e6, 2),
               "trainable_m": round(n_train_p / 1e6, 3), "history": history,
               "dev": dev_met, "test": test_met,
               "samples": {"dev": dev_sample, "test": test_sample}}
    with open(LOG_DIR / f"metrics_{run}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"【{run} 汇总】dev F1={dev_met['f1']:.4f} test F1={test_met['f1']:.4f} | "
          f"训练 {train_s/60:.1f} 分钟 | 日志 → {LOG_DIR / ('metrics_' + run + '.json')}", flush=True)


if __name__ == "__main__":
    main()
