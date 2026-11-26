"""生成式作业的本地模型和课程数据路径。"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT.parent / "week7-信息抽取" / "code" / "data" / "processed"
QWEN_MODEL_DIR = ROOT / "qwen3-0.6b-base"


def resolve_model_path(model_path=None) -> Path:
    """只使用本地权重，缺文件时直接报错，不隐式联网下载。"""
    path = Path(model_path).expanduser().resolve() if model_path else QWEN_MODEL_DIR
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"缺少模型配置：{path / 'config.json'}")
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = path / name
        if index.is_file():
            shards = set(json.loads(index.read_text())["weight_map"].values())
            if shards and all((path / shard).is_file() and (path / shard).stat().st_size > 0
                              for shard in shards):
                return path
            raise FileNotFoundError(f"模型分片未补齐：{index}")
    if any(p.is_file() and p.stat().st_size > 0
           for name in ("model.safetensors", "pytorch_model.bin") for p in [path / name]):
        return path
    raise FileNotFoundError(f"缺少本地模型权重：{path}。请补齐 model.safetensors，或用 --model_path 指定完整目录。")
