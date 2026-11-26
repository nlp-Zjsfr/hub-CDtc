"""中文命名实体识别方法对比。实体使用 [start, end) 字符偏移。"""
import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
TYPES = {"PER", "LOC", "ORG"}
MODELS = {
    "bert": "Davlan/bert-base-multilingual-cased-ner-hrl",
    "distilbert": "Davlan/distilbert-base-multilingual-cased-ner-hrl",
}


def entity(text, start, end, kind):
    return {"text": text[start:end], "start": start, "end": end, "type": kind}


def load_data(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("测试集为空")
    for row in rows:
        validate(row["text"], row["entities"])
    return rows


def validate(text, entities):
    if not isinstance(entities, list):
        raise ValueError("entities 必须为列表")
    for e in entities:
        s, t = e["start"], e["end"]
        if (type(s) is not int or type(t) is not int or not 0 <= s < t <= len(text)
                or e["type"] not in TYPES or text[s:t] != e["text"]):
            raise ValueError(f"无效实体: {e}")
    return entities


def rule_extractor():
    dictionary = json.loads((ROOT / "dictionary.json").read_text(encoding="utf-8"))
    def predict(text):
        candidates = []
        for kind, words in dictionary.items():
            for word in words:
                for match in re.finditer(re.escape(word), text):
                    candidates.append(entity(text, match.start(), match.end(), kind))
        candidates.sort(key=lambda e: e["end"] - e["start"], reverse=True)
        # 中文机构名的简单后缀规则（故意保留规则方法的局限性）
        for m in re.finditer(r"[\u4e00-\u9fff]{2,6}(?:大学|医院|银行|公司)", text):
            candidates.append(entity(text, m.start(), m.end(), "ORG"))
        # 优先词典实体，再补充与已有实体不重叠的规则候选
        result = []
        for e in candidates:
            if not any(e["start"] < x["end"] and x["start"] < e["end"] for x in result):
                result.append(e)
        return result
    return predict


def model_extractor(name, device):
    from transformers import pipeline
    ner = pipeline("token-classification", model=MODELS[name], aggregation_strategy="simple", device=device)
    def predict(text):
        result = []
        for item in ner(text):
            kind = item["entity_group"]
            if kind in TYPES:
                result.append(entity(text, int(item["start"]), int(item["end"]), kind))
        return result
    return predict


def api_extractor():
    # OpenAI 兼容 Chat Completions 接口；URL 必须为完整 endpoint。
    url = os.environ.get("NER_API_URL")
    key = os.environ.get("NER_API_KEY")
    model = os.environ.get("NER_API_MODEL")
    if not all((url, key, model)):
        raise ValueError("请配置 NER_API_URL、NER_API_KEY、NER_API_MODEL")
    if not url.startswith("https://"):
        raise ValueError("API URL 必须使用 HTTPS")
    def predict(text):
        prompt = ("抽取文本中的人名(PER)、地名(LOC)、组织机构名(ORG)。不要抽取其他类型。"
                  "使用最长、非重叠实体，不猜测文本外的内容。仅输出JSON对象，格式为"
                  '{"entities":[{"text":"张三","type":"PER"}]}。'
                  "重复出现的同一实体只需列一次。没有实体时返回空列表。")
        body = {"model": model, "temperature": 0, "messages": [
            {"role": "system", "content": prompt}, {"role": "user", "content": text}]}
        req = Request(url, data=json.dumps(body).encode("utf-8"), headers={
            "Content-Type": "application/json", "Authorization": "Bearer " + key}, method="POST")
        with urlopen(req, timeout=90) as response:
            content = json.load(response)["choices"][0]["message"]["content"].strip()
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.I)
        items = json.loads(content)["entities"]
        if not isinstance(items, list):
            raise ValueError("API 返回的 entities 不是列表")
        result = []
        for item in items:
            word, kind = item["text"], item["type"]
            if not isinstance(word, str) or not word or kind not in TYPES or word not in text:
                raise ValueError("API 返回了无效或不存在于原文的实体")
            # 使用原文定位，避免模型生成的字符偏移不准确。
            for match in re.finditer(re.escape(word), text):
                result.append(entity(text, match.start(), match.end(), kind))
        return result
    return predict


def entity_set(entities):
    return {(e["start"], e["end"], e["type"]) for e in entities}


def evaluate(rows, predict):
    tp = fp = fn = 0
    details = []
    begin = time.perf_counter()
    for i, row in enumerate(rows, 1):
        predicted = validate(row["text"], predict(row["text"]))
        gold, pred = entity_set(row["entities"]), entity_set(predicted)
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
        details.append({"id": i, "text": row["text"], "gold": row["entities"],
                        "prediction": predicted, "exact_match": gold == pred})
    elapsed = time.perf_counter() - begin
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    metrics = {"samples": len(rows), "tp": tp, "fp": fp, "fn": fn,
               "precision": round(p, 6), "recall": round(r, 6), "f1": round(f1, 6),
               "total_seconds": round(elapsed, 6), "ms_per_sample": round(elapsed * 1000 / len(rows), 3)}
    return metrics, details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", choices=["rule", "bert", "distilbert", "api"],
                        default=["rule", "bert", "distilbert"])
    parser.add_argument("--data", type=Path, default=ROOT / "test_data.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--device", type=int, default=-1, help="-1=CPU, 0=第一张CUDA GPU")
    args = parser.parse_args()
    rows = load_data(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for method in dict.fromkeys(args.methods):
        detail_file = args.output / f"{method}_predictions.jsonl"
        detail_file.unlink(missing_ok=True)
        label = MODELS.get(method, os.environ.get("NER_API_MODEL", "") if method == "api" else "dictionary+regex")
        try:
            if method == "rule":
                predict = rule_extractor()
            elif method == "api":
                predict = api_extractor()
            else:
                predict = model_extractor(method, args.device)
            metrics, details = evaluate(rows, predict)
            detail_file.write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in details) + "\n", encoding="utf-8")
            result = {"method": method, "model": label, "status": "ok", **metrics, "error": ""}
            print(f"{method}: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}, {metrics['ms_per_sample']:.3f} ms/sample")
            del predict
        except Exception as exc:
            # 不将失败或缺少密钥的实验记作 0 分。
            result = {"method": method, "model": label, "status": "failed", "error": type(exc).__name__}
            print(f"{method}: 未完成 ({type(exc).__name__})")
        results.append(result)
    columns = ["method", "model", "status", "samples", "tp", "fp", "fn", "precision", "recall", "f1", "total_seconds", "ms_per_sample", "error"]
    with (args.output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)
    print("结果目录:", args.output)
    return int(any(r["status"] != "ok" for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
