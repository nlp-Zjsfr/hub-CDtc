"""模型输出 logits；CrossEntropyLoss 内部完成 log-softmax，无需提前 Softmax。"""
import torch
from torch import nn


class NewsClassifier(nn.Module):
    def __init__(self, input_dim=5000, hidden_dims=(128, 64), num_classes=10, dropout=0.0):
        super().__init__()
        layers, previous = [], input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(previous, hidden), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous = hidden
        layers.append(nn.Linear(previous, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def load_checkpoint(path, device="cpu"):
    # 仅加载本项目自己保存的 state_dict 与基础类型配置。
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    cfg = checkpoint["config"]
    model = NewsClassifier(checkpoint["input_dim"], cfg["hidden_dims"], checkpoint["num_classes"], cfg["dropout"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint
