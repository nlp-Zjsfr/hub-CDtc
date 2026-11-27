"""本地预训练模型定位：向上查找 pretrain_models/<模型名>，不静默联网。"""

from pathlib import Path

QWEN_DIR = "Qwen2.5-0.5B-Instruct"
PRETRAIN = "pretrain_models"


def _has_weights(p: Path) -> bool:
    if not p.is_dir():
        return False
    if any(p.glob("*.safetensors")):
        return True
    return (p / "pytorch_model.bin").exists()


def resolve(dir_name: str, extra_roots=None) -> Path:
    here = Path(__file__).resolve()
    candidates = []
    for root in (extra_roots or []):
        root = Path(root)
        candidates.append(root if root.name == dir_name else root / dir_name)
    for parent in here.parents:
        candidates.append(parent / PRETRAIN / dir_name)
    for c in candidates:
        if _has_weights(c):
            return c
    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"未找到本地模型 {dir_name}，尝试过：\n  {tried}")


def qwen_path() -> Path:
    return resolve(QWEN_DIR)


if __name__ == "__main__":
    print(f"{QWEN_DIR:<30} -> {qwen_path()}")
