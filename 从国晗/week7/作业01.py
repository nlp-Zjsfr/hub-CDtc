"""第七周：在同一批 DuEE-Fin 样本上比较规则、BERT 和 Qwen2.5。"""
import argparse
import hashlib
import json
import random
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parent
COURSE = ROOT / "课件代码"
PRETRAINED = ROOT.parent / "第六周_分类方法对比" / "models"
METHODS = ("rule", "bert", "qwen_zero", "qwen_few")
FEW_SHOT_TYPES = ("解除质押", "股东减持", "股份回购")


def normalized_text(text):
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def sample_id(row):
    return row["id"], row["sent_id"]


def event_types(rows):
    return dict(sorted(Counter(row["event"]["type"] for row in rows).items()))


def stratified_sample(rows, size, seed):
    """按事件类型尽量保持原比例，每类至少保留一条。"""
    groups = defaultdict(list)
    for row in rows:
        groups[row["event"]["type"]].append(row)
    if not len(groups) <= size <= len(rows):
        raise ValueError(f"采样数必须在 {len(groups)} 到 {len(rows)} 之间")
    if size == len(rows):
        return sorted(rows, key=sample_id)

    extra = size - len(groups)
    remaining = len(rows) - len(groups)
    shares = {kind: divmod(extra * (len(group) - 1), remaining)
              for kind, group in groups.items()}
    counts = {kind: share[0] + 1 for kind, share in shares.items()}
    by_remainder = sorted(groups, key=lambda kind: (-shares[kind][1], kind))
    for kind in by_remainder[:size - sum(counts.values())]:
        counts[kind] += 1

    rng = random.Random(seed)
    sample = []
    for kind in sorted(groups):
        group = groups[kind]
        sample.extend(rng.sample(group, counts[kind]))
    rng.shuffle(sample)
    return sample


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_data(args):
    from evaluate_events import load_jsonl

    train_path = args.data / "train_extract.jsonl"
    dev_path = args.data / "dev_extract_clean.jsonl"
    full_train, full_dev = map(load_jsonl, (train_path, dev_path))
    train_by_text = {}
    for row in full_train:
        train_by_text.setdefault(normalized_text(row["text"]), row)
    train_doc_ids = {row["id"] for row in full_train}
    dev_by_text = {}
    for row in full_dev:
        text = normalized_text(row["text"])
        if row["id"] not in train_doc_ids and text not in train_by_text:
            dev_by_text.setdefault(text, row)

    train = stratified_sample(list(train_by_text.values()), args.train_size, args.seed)
    dev = stratified_sample(list(dev_by_text.values()), args.test_size, args.seed)
    manifest = {
        "seed": args.seed,
        "source_sha256": {path.name: file_sha256(path) for path in
                          (train_path, dev_path, args.data / "event_schema.json")},
        "candidate_counts": {"train": len(full_train), "dev": len(full_dev)},
        "deduplicated_counts": {"train": len(train_by_text), "dev": len(dev_by_text)},
        "train_by_type": event_types(train),
        "test_by_type": event_types(dev),
        "train_ids": [list(sample_id(row)) for row in train],
        "test_ids": [list(sample_id(row)) for row in dev],
    }
    dest = args.output / "split.json"
    if dest.exists() and json.loads(dest.read_text(encoding="utf-8")) != manifest:
        raise ValueError("输出目录已有另一份数据划分，请指定新的 --output")
    save_json(dest, manifest)
    save_jsonl(args.output / "gold.jsonl", dev)
    return train, dev


def rule_predictions(train, inputs, schema):
    from rule.run_rule import (
        MANUAL_TRIGGERS, build_company_dict, build_trigger_dict, extract_one, merge_trigger_dict,
    )

    start = perf_counter()
    triggers = merge_trigger_dict(build_trigger_dict(train), MANUAL_TRIGGERS)
    companies = build_company_dict(train)
    setup_s = perf_counter() - start
    start = perf_counter()
    results = []
    for row in inputs:
        event = extract_one(row["text"], schema, triggers, companies)
        results.append([event] if event else [])
    return results, {"train_s": 0, "setup_s": setup_s, "inference_s": perf_counter() - start,
                     "batch_size": 1, "device": "cpu"}


def bert_predictions(args, train, inputs, schema):
    import torch
    from torch.utils.data import DataLoader
    from transformers import BertTokenizerFast, get_linear_schedule_with_warmup

    from bert.train import build_optimizer, run_epoch, set_seed
    from event_data import BioDataset, build_joint_schema, collate_bio, decode_bio_spans
    from event_data import illegal_bio_stats, load_enums, spans_to_event
    from models import build_model, predict_char_tags

    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    args.decoder = "linear"
    args.hard_constraint = False
    args.dropout = 0.1
    args.lr = 2e-5
    args.head_lr_mult = 5.0
    args.weight_decay = 0.01
    args.max_grad_norm = 1.0
    args.grad_accum = 1
    scheme = build_joint_schema(schema)
    tokenizer = BertTokenizerFast.from_pretrained(args.bert_path, local_files_only=True)
    train_set = BioDataset(train, tokenizer, scheme, args.max_length)
    # 课件数据类会生成对齐标签，使用空事件占位以免推理接触真实标签。
    test_set = BioDataset(inputs, tokenizer, scheme, args.max_length)
    train_loader = DataLoader(train_set, batch_size=args.bert_batch, shuffle=True, collate_fn=collate_bio)
    test_loader = DataLoader(test_set, batch_size=args.bert_batch, shuffle=False, collate_fn=collate_bio)
    model = build_model(args, scheme).to(device)
    optimizer = build_optimizer(model, args)
    steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, steps // 10, steps)
    print(f"BERT 在 {len(train)} 条训练样本上训练 {args.epochs} 轮，共 {steps} 步。", flush=True)
    start = perf_counter()
    history = []
    for epoch in range(1, args.epochs + 1):
        epoch_start = perf_counter()
        loss = run_epoch(model, train_loader, optimizer, scheduler, device, args, epoch)
        record = {"epoch": epoch, "train_loss": loss, "elapsed_s": perf_counter() - epoch_start}
        history.append(record)
        print(f"第 {epoch} 轮 train loss={loss:.4f}，耗时 {record['elapsed_s']:.1f}s", flush=True)
    train_s = perf_counter() - start
    enums = load_enums(args.data / "event_schema.json")
    start = perf_counter()
    all_tags, _ = predict_char_tags(model, test_loader, scheme["id2label"], device)
    results = []
    for row, tags in zip(inputs, all_tags, strict=True):
        event = spans_to_event(decode_bio_spans(tags), row["text"], enums=enums)
        results.append([event] if event else [])
    return results, {"train_s": train_s, "inference_s": perf_counter() - start,
                     "batch_size": args.bert_batch, "device": str(device), "epochs": history,
                     "steps": steps, "max_length": args.max_length,
                     "bert_layers": args.bert_layers or 12, "model_dir": str(args.bert_path),
                     "char_length_over_limit": test_set.n_truncated,
                     "illegal_bio": illegal_bio_stats(all_tags)}


def find_span(text, piece):
    start = text.find(piece) if isinstance(piece, str) and piece else -1
    return {"text": piece, "start": start, "end": start + len(piece)} if start >= 0 else None


def parse_generation(raw, text, schema):
    """严格读取 JSON，检查事件、角色及原文片段。失败时保留空预测。"""
    counts = Counter()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        counts["invalid_json"] += 1
        return [], counts

    entries = obj.get("events") if isinstance(obj, dict) else None
    if not isinstance(entries, list) or len(entries) > 1:
        counts["invalid_structure"] += 1
        return [], counts
    if not entries:
        return [], counts
    event = entries[0]
    if not isinstance(event, dict) or not isinstance(event.get("arguments"), list):
        counts["invalid_structure"] += 1
        return [], counts

    event_type = event.get("type")
    if not isinstance(event_type, str) or event_type not in schema:
        counts["invalid_fields"] += 1
        return [], counts
    trigger = find_span(text, event.get("trigger"))
    if trigger is None:
        counts["outside_source"] += 1
        return [], counts

    arguments = []
    seen = set()
    for arg in event["arguments"]:
        if not isinstance(arg, dict) or arg.get("role") not in schema[event_type]:
            counts["invalid_fields"] += 1
            continue
        span = find_span(text, arg.get("text"))
        if span is None:
            counts["outside_source"] += 1
            continue
        key = arg["role"], span["text"]
        if key not in seen:
            seen.add(key)
            arguments.append({"role": arg["role"], **span})
    return [{"type": event_type, "trigger": trigger, "args": arguments}], counts


def qwen_prompt(schema):
    roles = "\n".join(f"{event_type}：{'、'.join(names)}" for event_type, names in schema.items())
    return (
        "你负责从金融公告原文中抽取一个事件。type 必须是下表中的事件类型。"
        "trigger 是指出该事件的原文短语。arguments 是论元列表。"
        "只填该类型的合法 role，trigger 和论元 text 必须逐字出现在原文中，缺失角色省略。"
        "只输出 JSON 对象，结构为 "
        '{"events":[{"type":"事件类型","trigger":"触发词","arguments":[{"role":"角色","text":"原文片段"}]}]}。'
        '没有事件时输出 {"events":[]}。不要解释，不要代码块。\n'
        f"事件类型和角色：\n{roles}"
    )


def training_examples(train, schema):
    selected = []
    for event_type in FEW_SHOT_TYPES:
        choices = []
        for row in train:
            if row["event"]["type"] != event_type:
                continue
            event = row["event"]
            item = {"type": event_type, "trigger": event["trigger"]["text"],
                    "arguments": [{"role": a["role"], "text": a["text"]} for a in event["args"]]}
            answer = json.dumps({"events": [item]}, ensure_ascii=False, separators=(",", ":"))
            events, problems = parse_generation(answer, row["text"], schema)
            if events and not problems and len(item["arguments"]) >= 2:
                choices.append((row, answer))
        if not choices:
            raise ValueError(f"训练样本里没有适合的少样本示例：{event_type}")
        selected.append(min(choices, key=lambda item: (len(item[0]["text"]), sample_id(item[0]))))
    return selected


def load_qwen(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_path, local_files_only=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        args.qwen_path, local_files_only=True, torch_dtype=dtype).to(device).eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    return model, tokenizer, device


def qwen_predictions(args, train, inputs, schema, backend, few_shot):
    import torch

    model, tokenizer, device = backend
    examples = training_examples(train, schema) if few_shot else []
    prefix = [{"role": "system", "content": qwen_prompt(schema)}]
    for row, answer in examples:
        prefix.extend([{"role": "user", "content": f"原文：{row['text']}"},
                       {"role": "assistant", "content": answer}])

    start = perf_counter()
    results, details = [], []
    totals = Counter()
    eos = model.generation_config.eos_token_id
    stop_ids = set(eos if isinstance(eos, list) else [eos])
    for first in range(0, len(inputs), args.qwen_batch):
        batch = inputs[first:first + args.qwen_batch]
        prompts = [tokenizer.apply_chat_template(
            prefix + [{"role": "user", "content": f"原文：{row['text']}"}],
            tokenize=False, add_generation_prompt=True) for row in batch]
        encoded = tokenizer(prompts, return_tensors="pt", add_special_tokens=False, padding=True)
        prompt_counts = encoded["attention_mask"].sum(dim=1).tolist()
        encoded = encoded.to(device)
        with torch.inference_mode():
            generated = model.generate(**encoded, do_sample=False, max_new_tokens=args.max_new_tokens,
                                       pad_token_id=tokenizer.pad_token_id)
        continuations = generated[:, encoded["input_ids"].shape[1]:].tolist()
        for row, new_ids, prompt_count in zip(batch, continuations, prompt_counts, strict=True):
            stop = next((i for i, token in enumerate(new_ids) if token in stop_ids), None)
            count = len(new_ids) if stop is None else stop + 1
            raw = tokenizer.decode(new_ids[:count], skip_special_tokens=True)
            events, diagnostics = parse_generation(raw, row["text"], schema)
            totals.update(diagnostics)
            totals.update({"prompt_tokens": prompt_count, "completion_tokens": count,
                           "hit_token_limit": int(stop is None and count == args.max_new_tokens)})
            results.append(events)
            details.append({"raw": raw, "diagnostics": dict(diagnostics),
                            "prompt_tokens": prompt_count, "completion_tokens": count})
        done = len(results)
        if done % (args.qwen_batch * 5) == 0 or done == len(inputs):
            print(f"Qwen {'few' if few_shot else 'zero'} {done}/{len(inputs)} | "
                  f"{perf_counter() - start:.1f}s", flush=True)
    meta = {"train_s": 0, "inference_s": perf_counter() - start, "batch_size": args.qwen_batch,
            "device": str(device), "dtype": str(model.dtype), "model_dir": str(args.qwen_path),
            "max_new_tokens": args.max_new_tokens,
            "few_shot_examples": [{"id": row["id"], "sent_id": row["sent_id"],
                                   "type": row["event"]["type"]} for row, _ in examples],
            "generation": dict(totals)}
    return results, details, meta


def write_result(name, gold, results, meta, output, details=None):
    from evaluate_events import evaluate

    if details is None:
        details = [{} for _ in gold]
    predictions = [
        {"id": row["id"], "sent_id": row["sent_id"], "events": events, **detail}
        for row, events, detail in zip(gold, results, details, strict=True)
    ]
    metrics = evaluate(gold, predictions, verbose=False)
    metrics["run"] = {**meta, "empty_predictions": sum(not entry for entry in results),
                      "ms_per_sample": 1000 * meta["inference_s"] / len(gold)}
    save_jsonl(output / f"pred_{name}.jsonl", predictions)
    save_json(output / f"metrics_{name}.json", metrics)
    print(f"{name} | trigger={metrics['trigger']['f1']:.4f} | "
          f"argument={metrics['argument']['f1']:.4f} | event={metrics['event_exact']['f1']:.4f}", flush=True)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("必须是正整数")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--course", type=Path, default=COURSE)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=positive_int, default=3200)
    parser.add_argument("--test-size", type=positive_int, default=240)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--bert-path", type=Path, default=PRETRAINED / "bert-base-chinese")
    parser.add_argument("--qwen-path", type=Path, default=PRETRAINED / "Qwen2.5-0.5B-Instruct")
    parser.add_argument("--epochs", type=positive_int, default=2)
    parser.add_argument("--bert-layers", type=int, choices=range(13), default=0, help="0 表示完整 12 层")
    parser.add_argument("--max-length", type=positive_int, default=256)
    parser.add_argument("--bert-batch", type=positive_int, default=24)
    parser.add_argument("--qwen-batch", type=positive_int, default=8)
    parser.add_argument("--max-new-tokens", type=positive_int, default=256)
    args = parser.parse_args()
    if not (args.course / "common" / "evaluate_events.py").is_file():
        parser.error(f"找不到评测脚本，请通过 --course 指定课件 code 目录：{args.course}")
    sys.path[:0] = [str(args.course / "common"), str(args.course)]
    from event_data import load_schema

    args.output.mkdir(parents=True, exist_ok=True)
    train, gold = prepare_data(args)
    schema = load_schema(args.data / "event_schema.json")
    inputs = [{"id": row["id"], "sent_id": row["sent_id"], "text": row["text"]} for row in gold]
    print(f"固定样本：训练 {len(train)} 条，评测 {len(gold)} 条，{len(event_types(gold))} 类。", flush=True)

    backend = None
    for name in (method for method in METHODS if method in args.methods):
        if name == "rule":
            results, meta = rule_predictions(train, inputs, schema)
            details = None
        elif name == "bert":
            results, meta = bert_predictions(args, train, inputs, schema)
            details = None
        else:
            if backend is None:
                backend = load_qwen(args)
            results, details, meta = qwen_predictions(args, train, inputs, schema, backend, name == "qwen_few")
        write_result(name, gold, results, meta, args.output, details)


if __name__ == "__main__":
    main()
