import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["HF_HOME"] = "" \
# "D:/deeplearning"

from datasets import load_dataset

# 直接加载，它已经内置了 train/validation/test 的划分
dataset = load_dataset("sookiemonster/asrs-narratives-rebalance")
print(dataset)
print(dataset["train"][0]) 
print(dataset["train"].features["label"].names)