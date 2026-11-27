"""
本地预训练模型定位（离线课堂必备）

为什么不能写死路径：
  同一个项目会存在于两套目录树里（母版 `项目材料准备/` 与交付副本 `八斗课程/weekN/`），
  脚本所在层级不同，`../pretrain_models` 在其中一套里必然失效。
  做法：从脚本所在目录**向上逐级查找** `pretrain_models/<模型名>`，
  并且要求目录里真的有权重文件——否则会拿到半下载的空目录（离线课堂最常见的坑）。

用法：
  from common.model_paths import qwen3_path, roberta_path
"""

from pathlib import Path

QWEN_DIR = "Qwen3-0.6B"
# 判别式主力基座：权重就放在项目根目录
ROBERTA_DIR = "nlp_roberta_backbone_base_std"
PRETRAIN = "pretrain_models"
CODE_ROOT = Path(__file__).resolve().parent.parent


def _has_weights(p: Path) -> bool:
    if not p.is_dir():
        return False
    if any(p.glob("*.safetensors")):
        return True
    return (p / "pytorch_model.bin").exists()


def resolve(dir_name: str, extra_roots=None) -> Path:
    """
    从当前文件向上逐级找 pretrain_models/<dir_name>。
    找不到就抛错并列出尝试过的路径——**不静默回退到联网下载**，保证断网也能用。
    """
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


def qwen3_path() -> Path:
    return resolve(QWEN_DIR)


def roberta_path() -> Path:
    """
    当前流水线唯一的判别式基座：ModelScope 中文 RoBERTa-base。

    权重直接放在项目根目录（`nlp_roberta_backbone_base_std/`，411 MB），
    不再要求塞进 pretrain_models/——离线课堂少一次搬文件。
    也兼容别人把它规范成 `pretrain_models/roberta-base-chinese/` 的布局。

    注意：这份 state_dict 带 `bert.` 前缀和 Paddle 风格 LayerNorm.gamma/beta。
    transformers 5.x 的 from_pretrained 会自动去掉前缀、把 gamma/beta 映射成 weight/bias；
    4.x 不会，只会打 warning 然后**随机初始化**。所以调用方必须校验 missing_keys（见 roberta/train_roberta.py）。
    """
    local = CODE_ROOT / ROBERTA_DIR
    if _has_weights(local):
        return local
    return resolve(ROBERTA_DIR)


if __name__ == "__main__":
    # 自检：课堂排查「模型找不到」时先跑这个。
    # 逐个 try：Qwen3 缺失不应该挡住院线要用的 RoBERTa 的自检。
    for _name, _fn in ((ROBERTA_DIR, roberta_path), (QWEN_DIR, qwen3_path)):
        try:
            print(f"{_name:<34} -> {_fn()}")
        except FileNotFoundError as _e:
            print(f"{_name:<34} -> [缺失] {_e}")
