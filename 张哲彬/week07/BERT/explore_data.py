import json
import pathlib
import math
import matplotlib.pyplot as plt

ROOT_DIR = pathlib.Path(__file__).parent.parent
DATA_PATH = ROOT_DIR / "peoples_daily"

def find_max_length(lengths):
    """
    计算最大长度、98%分位数长度、95%分位数长度。
    lengths: list[int]，每条样本的长度
    """
    max_length = max(lengths)
    sorted_lengths = sorted(lengths)
    n = len(sorted_lengths)
    max_length98 = sorted_lengths[math.ceil(n * 0.98) - 1]
    max_length95 = sorted_lengths[math.ceil(n * 0.95) - 1]
    middle_length = sorted_lengths[math.ceil(n * 0.5) - 1]
    # 画出长度分布直方图
    # plt.hist(tokens_length_distribution, bins=50, color="skyblue", edgecolor="black")
    # plt.title("Token Length Distribution")
    # plt.xlabel("Token Length")
    # plt.ylabel("Frequency")
    # plt.axvline(max_length, color="red", linestyle="dashed", linewidth=1, label="Max Length")
    # plt.axvline(max_length98, color="blue", linestyle="dashed", linewidth=1, label="98% Quantile")
    # plt.axvline(max_length95, color="green", linestyle="dashed", linewidth=1, label="95% Quantile")
    # plt.legend()
    # plt.show()  
    return max_length, max_length98, max_length95, middle_length

def caculate_label_distribution(data):
    """
    统计全部标签类型及其出现次数，返回字典 {label: count}。
    """
    label_count = {}
    for record in data:
        labels = record.get("ner_tags", [])
        for label in labels:
            label_count[label] = label_count.get(label, 0) + 1
    return label_count

if __name__ == "__main__":
    with open(DATA_PATH / "train.json", encoding="utf-8") as f:
        train_data = json.load(f)
    label_distribution = caculate_label_distribution(train_data)
    tokens_length_distribution = [len(record.get("tokens", [])) for record in train_data]
    train_data_size = len(train_data)
    print("训练集样本数：", train_data_size)
    max_length, max_length98, max_length95, middle_length = find_max_length(tokens_length_distribution)
    print("训练集最大长度：", max_length)
    print("训练集98%分位数长度：", max_length98)
    print("训练集95%分位数长度：", max_length95)
    print("训练集长度中位数：", middle_length)
    tokens_length = 128
    percentage = sum(value < tokens_length for value in tokens_length_distribution) / len(tokens_length_distribution) * 100
    print(f"{tokens_length} 大于列表中 {percentage:.2f}% 的数字")
