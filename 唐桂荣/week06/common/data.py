"""读取商品数据，按商品名去重，并构建所有实验共用的固定切分。"""

import json
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SPLIT_SEED = 42
VAL_FRACTION = 0.1


class Sample(TypedDict):
    idx: int
    sentence: str
    label: int


@dataclass
class DatasetSplits:
    train: list[Sample]
    val: list[Sample]
    test: list[Sample]
    labels: tuple[str, ...]


def normalize_product_name(name: str) -> str:
    """生成去重键。模型输入仍使用原始商品名。"""
    return "".join(unicodedata.normalize("NFKC", name).casefold().split())


def _read_groups(path: Path) -> dict[str, list[tuple[int, str, str]]]:
    groups = defaultdict(list)
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} JSON 格式错误") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} 应为 JSON 对象")
            name, category = row.get("product_name"), row.get("category")
            if not isinstance(name, str) or not isinstance(category, str) or not category.strip():
                raise ValueError(f"{path}:{line_number} 商品名和类别应为非空字符串")
            name_key = normalize_product_name(name)
            if not name_key:
                raise ValueError(f"{path}:{line_number} 商品名应为非空字符串")
            groups[name_key].append((line_number, name, category))
    return groups


def _keep_unambiguous_names(groups, excluded_names):
    """同名同类保留首条，同名异类整组排除。"""
    return [
        rows[0]
        for name, rows in groups.items()
        if name not in excluded_names and len({category for _, _, category in rows}) == 1
    ]


def load_splits(data_dir: Path = DATA_DIR) -> DatasetSplits:
    """返回固定切分及标签表，idx 为样本在原始文件中的行号（从 1 开始）。"""
    train_groups = _read_groups(data_dir / "train.jsonl")
    test_groups = _read_groups(data_dir / "test.jsonl")
    clean_train = _keep_unambiguous_names(train_groups, set())
    # 用原始训练集的去重键，覆盖因标签冲突而被移出训练集的商品名。
    clean_test = _keep_unambiguous_names(test_groups, train_groups.keys())

    labels = tuple(sorted({category for _, _, category in clean_train}))
    label_to_id = {name: index for index, name in enumerate(labels)}
    unknown_labels = {category for _, _, category in clean_test} - label_to_id.keys()
    if unknown_labels:
        raise ValueError(f"清洗后的测试集含训练集中未出现的类别：{sorted(unknown_labels)}")

    train_samples = [
        Sample(idx=idx, sentence=name, label=label_to_id[category])
        for idx, name, category in clean_train
    ]
    train, val = train_test_split(
        train_samples,
        test_size=VAL_FRACTION,
        random_state=SPLIT_SEED,
        stratify=[sample["label"] for sample in train_samples],
    )
    test = [
        Sample(idx=idx, sentence=name, label=label_to_id[category])
        for idx, name, category in clean_test
    ]
    return DatasetSplits(train=train, val=val, test=test, labels=labels)
