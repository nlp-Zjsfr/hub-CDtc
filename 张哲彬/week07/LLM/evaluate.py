"""
    调用deepseek大模型完成NER任务的评估，同时测试zero-shot和few-shot的效果。
    评估指标为实体级F1分数。
    评估数据集为人民日报NER测试集.
"""

import argparse
import json
import os
import pathlib

from openai import OpenAI
from seqeval.metrics import classification_report, f1_score

ROOT_DIR = pathlib.Path(__file__).parent.parent
DATA_PATH = ROOT_DIR / "peoples_daily"
TEST_PATH = DATA_PATH / "test.json"
LABELS_PATH = DATA_PATH / "label_names.json"

SYSTEM_PROMPT = """
你是一个中文命名实体识别标注器。请对输入文本逐字符标注。
每个字符必须对应一个标签，保持字符顺序，不增删字符。
实体首字符使用 B-PER、B-ORG 或 B-LOC，实体后续字符使用对应的 I 标签。
非实体字符标为 O。
只输出 JSON，格式为：
{"tokens":"原文","ner_tags":["标签1","标签2"]}
"""


def load_data(file_path):
    """
    加载字符级 NER 数据，并验证文本与标签逐字符对齐。
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{file_path} 顶层 JSON 必须是列表")

    texts = []
    labels = []
    for index, record in enumerate(data):
        if not isinstance(record, dict):
            raise ValueError(f"第 {index} 条记录必须是 JSON 对象")

        tokens = record.get("tokens")
        ner_tags = record.get("ner_tags")
        if not isinstance(tokens, list) or not all(
            isinstance(token, str) for token in tokens
        ):
            raise ValueError(f"第 {index} 条记录的 tokens 必须是字符串列表")
        if not isinstance(ner_tags, list) or not all(
            isinstance(tag, str) for tag in ner_tags
        ):
            raise ValueError(f"第 {index} 条记录的 ner_tags 必须是字符串列表")
        if len(tokens) != len(ner_tags):
            raise ValueError(
                f"第 {index} 条记录未对齐：{len(tokens)} 个 token，"
                f"{len(ner_tags)} 个标签"
            )

        texts.append("".join(tokens))
        labels.append(ner_tags)

    if not texts:
        raise ValueError(f"{file_path} 中没有测试样本")
    return texts, labels


def load_allowed_labels(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    if not isinstance(labels, list) or not all(
        isinstance(label, str) for label in labels
    ):
        raise ValueError(f"{file_path} 必须包含标签字符串列表")
    return set(labels)


def parse_prediction(content, expected_text, allowed_labels, sample_index):
    """解析并验证单条模型输出；拒绝格式或字符对齐错误，避免污染整体 F1。"""
    try:
        result = json.loads(content)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(
            f"第 {sample_index} 条模型输出不是合法 JSON"
        ) from error

    if not isinstance(result, dict):
        raise ValueError(f"第 {sample_index} 条模型输出必须是 JSON 对象")
    if result.get("tokens") != expected_text:
        raise ValueError(
            f"第 {sample_index} 条模型输出的 tokens 与输入文本不一致"
        )

    prediction = result.get("ner_tags")
    if not isinstance(prediction, list) or not all(
        isinstance(tag, str) for tag in prediction
    ):
        raise ValueError(f"第 {sample_index} 条模型输出的 ner_tags 必须是字符串列表")
    if len(prediction) != len(expected_text):
        raise ValueError(
            f"第 {sample_index} 条标签数量不匹配：文本 {len(expected_text)} 字符，"
            f"模型返回 {len(prediction)} 个标签"
        )

    invalid_labels = sorted(set(prediction) - allowed_labels)
    if invalid_labels:
        raise ValueError(
            f"第 {sample_index} 条模型输出包含未知标签：{invalid_labels}"
        )
    return prediction


def parse_args():
    parser = argparse.ArgumentParser(description="使用 DeepSeek 评估测试集 NER")
    parser.add_argument("--model", default="deepseek-flash", help="DeepSeek 模型名")
    parser.add_argument("--test-path", type=pathlib.Path, default=TEST_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("未设置 DEEPSEEK_API_KEY 环境变量")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )

    texts, labels = load_data(args.test_path)
    allowed_labels = load_allowed_labels(LABELS_PATH)
    invalid_gold_labels = sorted(
        {tag for sequence in labels for tag in sequence} - allowed_labels
    )
    if invalid_gold_labels:
        raise ValueError(f"测试集包含未知真实标签：{invalid_gold_labels}")

    predictions = []
    evaluated_labels = []
    errors = []
    for index, text in enumerate(texts):
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"请标注以下文本：\n{text}"},
            ],
            stream=False,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}},
        )

        output = response.choices[0].message.content
        try:
            prediction = parse_prediction(
                output, text, allowed_labels, sample_index=index
            )
        except ValueError as e:
            errors.append(f"第 {index} 条数据：{e}")
            continue
        predictions.append(prediction)
        evaluated_labels.append(labels[index])

    if errors:
        print(f"\n共有 {len(errors)} 条模型输出无效，已跳过：")
        for error in errors:
            print(f"- {error}")

    if not predictions:
        print("没有可用于评估的有效预测，无法计算 F1。")
    else:
        test_f1 = f1_score(
            evaluated_labels,
            predictions,
            average="micro",
            zero_division=0,
        )
        print(classification_report(
            evaluated_labels, predictions, digits=4, zero_division=0
        ))
        print(
            f"测试集整体实体级 micro-F1: {test_f1:.4f} "
            f"({len(predictions)}/{len(texts)} 条有效样本)"
            f"({len(errors)} 条无效样本)"
        )

if __name__ == "__main__":
    main()
