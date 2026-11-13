"""相同提示、贪心解码、严格 JSON/字段评分；不使用外部模型当裁判。"""
import json
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader

from common import ROOT, config, load_tokenizer, read_json, read_jsonl, write_json, write_jsonl, sha256
from modeling import inference_model
from sft_data import SFTDataset, SFTCollator, prompt_ids
from train import score, autocast

KEYS = ["order_id", "product", "intent", "problem", "urgent"]
INTENTS = {"return", "replace", "refund", "query_shipping"}
PROBLEMS = {"damaged", "wrong_size", "wrong_color", "missing_part", "late_delivery", "not_needed"}


def grade(text, target):
    def unique_keys(pairs):
        if len(pairs) != len({k for k, _ in pairs}):
            raise ValueError("duplicate JSON keys")
        return dict(pairs)
    try:
        obj = json.loads(text, object_pairs_hook=unique_keys)
    except (ValueError, TypeError):
        obj = None
    schema = (isinstance(obj, dict) and set(obj) == set(KEYS)
              and (obj["order_id"] is None or isinstance(obj["order_id"], str))
              and (obj["product"] is None or isinstance(obj["product"], str))
              and isinstance(obj["intent"], str) and obj["intent"] in INTENTS
              and isinstance(obj["problem"], str) and obj["problem"] in PROBLEMS and type(obj["urgent"]) is bool)
    # 不替非法 JSON 去掉代码围栏，不自动补字段或改类型；避免评估时掩盖输出错误。
    fields = {k: bool(isinstance(obj, dict) and k in obj and type(obj[k]) is type(target[k]) and obj[k] == target[k]) for k in KEYS}
    return {"json_valid": isinstance(obj, dict), "schema_valid": bool(schema),
            "exact_match": bool(schema and all(fields.values())), "fields": fields}


@torch.inference_mode()
def generate_rows(model, tokenizer, rows, cfg, batch_size=None):
    model.eval()
    device = next(model.parameters()).device
    eos = tokenizer.eos_token_id
    out = []
    batch_size = batch_size or cfg["generation_batch_size"]
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        prompts = [prompt_ids(tokenizer, r) for r in batch]
        width = max(map(len, prompts))
        ids = torch.full((len(batch), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for i, p in enumerate(prompts):
            ids[i, -len(p):] = torch.tensor(p, device=device)
            mask[i, -len(p):] = 1
        with autocast(device):
            generated = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=cfg["max_new_tokens"],
                do_sample=False, use_cache=True, eos_token_id=eos, pad_token_id=tokenizer.pad_token_id,
                repetition_penalty=1.0)
        for row, output in zip(batch, generated):
            new = output[width:].tolist()
            ended = eos in new
            if ended:
                new = new[:new.index(eos)]
            text = tokenizer.decode(new, skip_special_tokens=False).strip()
            record = {"id": row["id"], "input": row["messages"][1]["content"],
                      "prediction": text, "eos_stop": ended, "output_tokens": len(new),
                      "unseen_product": row.get("meta", {}).get("unseen_product", False)}
            if "target" in row:
                record.update({"target": row["target"], **grade(text, row["target"])})
            out.append(record)
        if len(rows) > batch_size:
            print(f"Generated {len(out)}/{len(rows)}", flush=True)
    return out


def summarize(predictions):
    n = len(predictions)
    result = {"samples": n, **{k: sum(r[k] for r in predictions)/n for k in ("json_valid", "schema_valid", "exact_match", "eos_stop")},
              "field_accuracy": {k: sum(r["fields"][k] for r in predictions)/n for k in KEYS},
              "mean_output_tokens": sum(r["output_tokens"] for r in predictions)/n}
    for name, predicate in (("unseen_products", lambda r:r["unseen_product"]),
                            ("seen_products_or_null", lambda r:not r["unseen_product"]),
                            ("missing_order", lambda r:r["target"]["order_id"] is None),
                            ("missing_product", lambda r:r["target"]["product"] is None)):
        sub = [r for r in predictions if predicate(r)]
        result[name] = {"samples": len(sub), "exact_match": sum(r["exact_match"] for r in sub)/len(sub) if sub else None}
    return result


def evaluate():
    cfg = config()
    training = read_json(ROOT / "results" / "training.json")
    assert training["config"] == cfg, "当前配置与完成的训练记录不同"
    assert training["template_sha256"] == sha256(ROOT / "chat_template.jinja")
    adapter = ROOT / "models" / "best_adapter" / "adapter_model.safetensors"
    assert training["best_adapter_sha256"] == sha256(adapter)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    tokenizer = load_tokenizer(cfg)
    model = inference_model(cfg, device)
    test = read_jsonl(ROOT / "data" / "test.jsonl")
    loaders = {s: DataLoader(SFTDataset(read_jsonl(ROOT / "data" / f"{s}.jsonl"), tokenizer, cfg["max_length"]),
                 batch_size=cfg["micro_batch_size"], collate_fn=SFTCollator(tokenizer.pad_token_id), num_workers=0) for s in ("val", "test")}
    results, all_predictions = {}, {}
    for name in ("base", "sft"):
        context = model.disable_adapter() if name == "base" else nullcontext()
        with context:
            losses = {s: score(model, loader, device) for s, loader in loaders.items()}
            print(name, "test answer NLL", losses["test"]["answer_nll"], flush=True)
            predictions = generate_rows(model, tokenizer, test, cfg)
        all_predictions[name] = predictions
        results[name] = {"losses": losses, "generation": summarize(predictions)}
        write_jsonl(ROOT / "results" / f"{name}_test_predictions.jsonl", predictions)
        print(name, json.dumps(results[name]["generation"], ensure_ascii=False), flush=True)
    write_json(ROOT / "results" / "metrics.json", {"best_epoch": training["best_epoch"], "adapter_sha256": sha256(adapter),
        "test_sha256": sha256(ROOT / "data" / "test.jsonl"), "decoding": {"do_sample": False, "max_new_tokens": cfg["max_new_tokens"],
        "eos_token": tokenizer.eos_token, "batch_size": cfg["generation_batch_size"], "padding_side": "left"},
        "metric_policy": "strict JSON, exact field names and types, no output repair; all 5 fields correct for exact match",
        "results": results})
    # 固定展示测试集前 4 条，并额外保存首个失败案例（如果存在），不挑选漂亮回答。
    indices = list(range(4))
    errors = [i for i, r in enumerate(all_predictions["sft"]) if not r["exact_match"]]
    if errors and errors[0] not in indices:
        indices.append(errors[0])
    write_json(ROOT / "results" / "examples.json", [
        {"id": test[i]["id"], "input": test[i]["messages"][1]["content"], "target": test[i]["target"],
         "base": all_predictions["base"][i]["prediction"], "sft": all_predictions["sft"][i]["prediction"],
         "sft_correct": all_predictions["sft"][i]["exact_match"]} for i in indices])


if __name__ == "__main__":
    evaluate()
