"""第六周作业：在相同的 TNEWS 样本上比较六种本地分类方法。"""

import argparse
import gc
import hashlib
import json
import random
import time
import unicodedata
from collections import Counter
from pathlib import Path

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from transformers import Adafactor, get_linear_schedule_with_warmup, set_seed


ROOT = Path(__file__).resolve().parent
# 沿用课件 data/train.json 的标签编号。
LABELS = [
    "故事", "文化", "娱乐", "体育", "财经", "房产", "汽车", "教育",
    "科技", "军事", "旅游", "国际", "证券", "农业", "电竞",
]
LABEL_IDS = list(range(len(LABELS)))
FAILED = -1
BERT = "bert-base-chinese"
QWEN = "Qwen2.5-0.5B-Instruct"
METHODS = {
    "rule": "关键词规则",
    "svm": "TF-IDF + LinearSVC",
    "bert": "BERT 微调",
    "zero": "Qwen2.5 零样本",
    "lora": "Qwen2.5 LoRA SFT",
    "full": "Qwen2.5 全量 SFT",
}
SYSTEM = "将新闻标题归入以下一个类别，只输出类别名。" + "、".join(LABELS) + "。"


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def title_key(sentence):
    return "".join(unicodedata.normalize("NFKC", sentence).split()).casefold()


def unique_titles(rows, blocked):
    """同文异标的标题全部移除，同文同标保留第一条。"""
    labels = {}
    for row in rows:
        labels.setdefault(title_key(row["sentence"]), set()).add(row["label"])
    seen = set(blocked)
    unique = []
    for row in rows:
        key = title_key(row["sentence"])
        if key and key not in seen and len(labels[key]) == 1:
            unique.append(row)
            seen.add(key)
    return unique


def stratified_sample(rows, size, seed):
    if size == len(rows):
        return rows
    selected, _ = train_test_split(
        rows, train_size=size, stratify=[row["label"] for row in rows], random_state=seed,
    )
    if {row["label"] for row in selected} != set(LABEL_IDS):
        raise ValueError(f"抽样 {size} 条后有类别缺失，请增大抽样数。")
    return selected


def prepare_data(args):
    train_file, eval_file = (ROOT / "data" / f"{name}.json" for name in ("train", "val"))
    train_bytes, eval_bytes = train_file.read_bytes(), eval_file.read_bytes()
    raw_train, raw_eval = json.loads(train_bytes), json.loads(eval_bytes)
    train_keys = {title_key(row["sentence"]) for row in raw_train}
    clean_train = unique_titles(raw_train, set())
    clean_eval = unique_titles(raw_eval, train_keys)
    train = stratified_sample(clean_train, args.train_size, args.seed)
    test = stratified_sample(clean_eval, args.test_size, args.seed)
    assert train_keys.isdisjoint(title_key(row["sentence"]) for row in test)
    manifest = {
        "dataset": "课件 TNEWS train.json / val.json",
        "train_sha256": hashlib.sha256(train_bytes).hexdigest(),
        "val_sha256": hashlib.sha256(eval_bytes).hexdigest(),
        "raw_train": len(raw_train),
        "raw_val": len(raw_eval),
        "val_overlaps_train": sum(title_key(row["sentence"]) in train_keys for row in raw_eval),
        "clean_train": len(clean_train),
        "clean_val": len(clean_eval),
        "train_ids": [row["idx"] for row in train],
        "test_ids": [row["idx"] for row in test],
        "train_distribution": dict(Counter(LABELS[row["label"]] for row in train)),
        "test_distribution": dict(Counter(LABELS[row["label"]] for row in test)),
        "seed": args.seed,
        "epochs": args.epochs,
        "device": args.device,
    }
    return train, test, manifest


def rule(text, fallback):
    # 课件基线。多个词命中时，按固定标签顺序取第一个。
    return next((i for i, label in enumerate(LABELS) if label in text), fallback)


def baselines(method, train, titles):
    y = [row["label"] for row in train]
    if method == "rule":
        fallback = Counter(y).most_common(1)[0][0]
        return [rule(title, fallback) for title in titles], None, {"fallback": LABELS[fallback]}

    started = time.perf_counter()
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), max_features=300_000)
    features = vectorizer.fit_transform([row["sentence"] for row in train])
    model = LinearSVC(C=1, random_state=42).fit(features, y)
    train_seconds = time.perf_counter() - started
    return model.predict(vectorizer.transform(titles)).tolist(), None, {"train_seconds": train_seconds}


def make_bert_batch(tokenizer, titles, device):
    return tokenizer(titles, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)


def bert_experiment(train, titles, args):
    path = ROOT / "models" / BERT
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=len(LABELS)).to(args.device)

    def collate(rows):
        batch = make_bert_batch(tokenizer, [row["sentence"] for row in rows], args.device)
        batch["labels"] = torch.tensor([row["label"] for row in rows], device=args.device)
        return batch

    loader = DataLoader(train, batch_size=32, shuffle=True, collate_fn=collate)
    stats = fit(model, loader, args, "bert", 3e-5)
    started = time.perf_counter()
    model.eval()
    predictions = []
    with torch.inference_mode():
        for offset in range(0, len(titles), 32):
            batch = make_bert_batch(tokenizer, titles[offset:offset + 32], args.device)
            predictions.extend(model(**batch).logits.argmax(dim=-1).tolist())
    stats["predict_seconds"] = time.perf_counter() - started
    return predictions, None, stats


def qwen_prompt(tokenizer, title):
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": title}],
        tokenize=True, add_generation_prompt=True,
    )


def make_sft_samples(tokenizer, rows):
    samples = []
    for row in rows:
        prompt_ids = qwen_prompt(tokenizer, row["sentence"])
        answer_ids = tokenizer.encode(LABELS[row["label"]], add_special_tokens=False)
        answer_ids.append(tokenizer.eos_token_id)
        # Prompt 和 padding 的目标设为 -100。第一个答案 token 仍参与损失。
        samples.append((prompt_ids + answer_ids, [-100] * len(prompt_ids) + answer_ids))
    return samples


def sft_batch(rows, pad_id, device):
    length = max(len(ids) for ids, _ in rows)
    tokens, labels, mask = [], [], []
    for ids, targets in rows:
        pad_count = length - len(ids)
        tokens.append(ids + [pad_id] * pad_count)
        labels.append(targets + [-100] * pad_count)
        mask.append([1] * len(ids) + [0] * pad_count)
    return {
        "input_ids": torch.tensor(tokens, device=device),
        "attention_mask": torch.tensor(mask, device=device),
        "labels": torch.tensor(labels, device=device),
    }


def sft_loss(model, batch):
    # 只有答案位置需要词表 logits，避免为提示词计算并保存大矩阵。
    base = model.get_base_model() if hasattr(model, "peft_config") else model
    hidden = base.model(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False,
    ).last_hidden_state
    targets = batch["labels"][:, 1:]
    supervised = targets.ne(-100)
    logits = base.get_output_embeddings()(hidden[:, :-1][supervised])
    return F.cross_entropy(logits, targets[supervised])


def fit(model, loader, args, method, learning_rate):
    parameters = [p for p in model.parameters() if p.requires_grad]
    if method == "full":
        # 分解二阶状态，使全量微调能在 8 GB 显存内保留 FP32 主权重。
        optimizer = Adafactor(parameters, lr=learning_rate, relative_step=False, scale_parameter=False)
    else:
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate, fused=args.device == "cuda")

    steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, steps // 10), steps)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    started = time.perf_counter()
    history = []
    for epoch in range(1, args.epochs + 1):
        losses = []
        for step, batch in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
                loss = model(**batch).loss if method == "bert" else sft_loss(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            if step == 1 or step % 50 == 0 or step == len(loader):
                print(f"{method} epoch {epoch}/{args.epochs} step {step}/{len(loader)} loss={losses[-1]:.4f}", flush=True)
        avg_loss = sum(losses) / len(losses)
        if not torch.isfinite(torch.tensor(avg_loss)):
            raise RuntimeError(f"{method} 出现非有限损失，停止评测。")
        history.append(avg_loss)
    return {
        "train_seconds": time.perf_counter() - started,
        "peak_train_gib": torch.cuda.max_memory_allocated() / 1024**3 if args.device == "cuda" else None,
        "trainable_parameters": sum(p.numel() for p in parameters),
        "optimizer": optimizer.__class__.__name__,
        "batch_size": loader.batch_size,
        "learning_rate": learning_rate,
        "train_loss_by_epoch": history,
    }


def qwen_experiment(method, train, titles, args):
    path = ROOT / "models" / QWEN
    tokenizer = AutoTokenizer.from_pretrained(path)
    inference_dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    # 全量微调用 FP32 主权重。LoRA 仅对少量适配参数求梯度。
    load_dtype = torch.float32 if method == "full" else inference_dtype
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=load_dtype).to(args.device)
    stats = {}
    if method == "lora":
        from peft import LoraConfig, TaskType, get_peft_model

        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, config)
    if method == "full":
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if method in ("lora", "full"):
        loader = DataLoader(
            make_sft_samples(tokenizer, train), batch_size=16, shuffle=True,
            collate_fn=lambda rows: sft_batch(rows, tokenizer.pad_token_id, args.device),
        )
        stats = fit(model, loader, args, method, 2e-4 if method == "lora" else 2e-5)
        model.zero_grad(set_to_none=True)
        if method == "full":
            model.gradient_checkpointing_disable()

    model.to(dtype=inference_dtype).eval()
    tokenizer.padding_side = "left"  # 生成时让每条输入最后一个位置都是真实 token。
    prompts = [qwen_prompt(tokenizer, title) for title in titles]
    generated = []
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(prompts), 16):
            batch = tokenizer.pad(
                {"input_ids": prompts[offset:offset + 16]}, padding=True, return_tensors="pt",
            ).to(args.device)
            tokens = model.generate(
                **batch, do_sample=False, max_new_tokens=8,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                temperature=None, top_p=None, top_k=None,
            )
            new_tokens = tokens[:, batch["input_ids"].shape[1]:]
            generated.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
            if offset == 0 or (offset + 16) % 160 == 0:
                print(f"{method} eval {len(generated)}/{len(titles)}", flush=True)
    stats["predict_seconds"] = time.perf_counter() - started
    label_to_id = {label: i for i, label in enumerate(LABELS)}
    predictions = [label_to_id.get(text.strip(), FAILED) for text in generated]
    return predictions, generated, stats


def evaluate(test, predictions, generated):
    if len(predictions) != len(test) or (generated is not None and len(generated) != len(test)):
        raise ValueError("测试样本和预测条数不一致。")
    y_true = [row["label"] for row in test]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predictions, labels=LABEL_IDS, zero_division=0,
    )
    matrix = confusion_matrix(y_true, predictions, labels=LABEL_IDS + [FAILED])
    rows = []
    for i, (sample, prediction) in enumerate(zip(test, predictions)):
        row = {
            "id": sample["idx"], "title": sample["sentence"], "true": LABELS[sample["label"]],
            "predicted": LABELS[prediction] if prediction != FAILED else None,
        }
        if generated is not None:
            row["response"] = generated[i]
        rows.append(row)
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "macro_f1": float(f1.mean()),
        "parse_failures": predictions.count(FAILED),
        "predicted_classes": len(set(predictions) - {FAILED}),
        "per_class": {
            label: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(n)}
            for label, p, r, f, n in zip(LABELS, precision, recall, f1, support)
        },
        "confusion_rows": LABELS,
        "confusion_columns": LABELS + ["解析失败"],
        "confusion_matrix": matrix[:-1].tolist(),
        "predictions": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train-size", type=int, default=3200)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    if min(args.train_size, args.test_size, args.epochs) < 1:
        parser.error("样本数和轮次必须为正数。")
    if args.device == "cuda" and not torch.cuda.is_bf16_supported():
        parser.error("CUDA 实验需要支持 BF16 的 GPU，可用 --device cpu 运行。")
    torch.set_num_threads(4)

    train, test, manifest = prepare_data(args)
    titles = [row["sentence"] for row in test]
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "split.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        parser.error("输出目录已有另一组实验，请为修改后的参数指定 --output。")
    write_json(manifest_path, manifest)
    for method in args.methods:
        set_seed(args.seed)
        random.seed(args.seed)
        started = time.perf_counter()
        if method in ("rule", "svm"):
            predictions, generated, stats = baselines(method, train, titles)
        elif method == "bert":
            predictions, generated, stats = bert_experiment(train, titles, args)
        else:
            predictions, generated, stats = qwen_experiment(method, train, titles, args)
        stats["wall_seconds"] = time.perf_counter() - started
        if method not in ("rule", "svm"):
            model_name = BERT if method == "bert" else QWEN
            revision_path = ROOT / "models" / model_name / "revision.txt"
            stats["base_model"] = model_name
            stats["base_revision"] = revision_path.read_text(encoding="ascii").strip()
        result = evaluate(test, predictions, generated)
        result.update(method=METHODS[method], **stats)
        write_json(args.output / f"{method}.json", result)
        print(f"{METHODS[method]} Accuracy={result['accuracy']:.4f} Macro-F1={result['macro_f1']:.4f}", flush=True)
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
