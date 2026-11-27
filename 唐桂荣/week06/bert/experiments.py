"""九组配置及正式运行顺序。每个变体只覆盖基线的一个实验设置。"""

from dataclasses import dataclass, replace

MODEL_NAMES = ("bert-base-chinese", "roberta_base")


@dataclass(frozen=True)
class Experiment:
    description: str
    pool: str = "cls"
    encoder_lr: float = 2e-5
    max_length: int = 64
    fgm_eps: float = 0.0
    label_smoothing: float = 0.0
    focal_gamma: float = 0.0


baseline = Experiment("CLS 池化")
EXPERIMENTS = {
    "exp1": baseline,
    "exp2": replace(baseline, description="Mean 池化", pool="mean"),
    "exp3": replace(baseline, description="Max 池化", pool="max"),
    "exp4": replace(baseline, description="CLS + Mean 池化", pool="cls_mean"),
    "exp5": replace(baseline, description="编码器学习率 3e-5", encoder_lr=3e-5),
    "exp6": replace(baseline, description="最大长度 32", max_length=32),
    "exp7": replace(baseline, description="FGM（ε=1.0）", fgm_eps=1.0),
    "exp8": replace(baseline, description="Label Smoothing（ε=0.1）", label_smoothing=0.1),
    "exp9": replace(baseline, description="Focal Loss（γ=2）", focal_gamma=2.0),
}
RUN_ORDER = ("exp6", "exp2", "exp3", "exp4", "exp5", "exp7", "exp8", "exp9")
