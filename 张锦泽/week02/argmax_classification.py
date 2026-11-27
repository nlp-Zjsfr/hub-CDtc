# coding: utf8
"""
使用 PyTorch 训练一个五分类模型：五维向量中，最大值在第几维，就属于第几类。

例如：[0.1, 0.8, 0.3, 0.2, 0.5] 属于第 2 类。
对外展示的类别是 1～5；训练标签是 CrossEntropyLoss 要求的 0～4。
如果最大值并列，训练标签取第一个最大值的位置。
训练样本的取值范围是 [0, 1)，模型学习得到的预测可能存在误差。

依赖：pip install torch numpy matplotlib
运行：python week02/argmax_classification.py
脚本会先训练并保存模型、绘制训练曲线，再加载模型进行预测。
"""

import os

# 沿用示例中解决 OpenMP 库冲突的设置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


INPUT_SIZE = 5


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, INPUT_SIZE)  # 每个类别对应一个输出
        self.loss = nn.CrossEntropyLoss()

    # 有标签时返回 loss；没有标签时返回各类别的原始分数 logits
    def forward(self, x, y=None):
        logits = self.linear(x)  # (batch_size, 5) -> (batch_size, 5)
        if y is not None:
            # CrossEntropyLoss 内部会处理归一化，计算损失前不需要 softmax
            return self.loss(logits, y)
        return logits


# 生成一个样本：最大值的位置就是标签，标签范围为 0～4
def build_sample():
    x = np.random.random(INPUT_SIZE)
    y = int(np.argmax(x))
    return x, y


# 五个类别出现的概率相同；标签是形状为 (样本数,) 的整数张量
def build_dataset(total_sample_num):
    samples = [build_sample() for _ in range(total_sample_num)]
    x, y = zip(*samples)
    return torch.tensor(np.array(x), dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def evaluate(model, test_x, test_y):
    model.eval()
    with torch.no_grad():
        logits = model(test_x)
        predictions = torch.argmax(logits, dim=1)
        correct = (predictions == test_y).sum().item()
    accuracy = correct / len(test_y)
    print("正确预测个数：%d/%d，正确率：%.2f%%" % (correct, len(test_y), accuracy * 100))
    return accuracy


def main(output_dir=None):
    # 固定随机种子，便于复现实验
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    epoch_num = 30
    batch_size = 64
    train_sample = 5000
    test_sample = 1000
    learning_rate = 0.01

    model = TorchModel(INPUT_SIZE)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_x, train_y = build_dataset(train_sample)
    # 使用独立、固定的测试集，便于比较每轮训练的效果
    test_x, test_y = build_dataset(test_sample)
    counts = torch.bincount(test_y, minlength=INPUT_SIZE).tolist()
    print("测试集第 1～5 类的样本数：%s" % counts)

    log = []
    for epoch in range(epoch_num):
        model.train()
        total_loss = 0.0
        # 每轮打乱训练数据，并保留最后不足一个 batch 的样本
        indices = torch.randperm(train_sample)
        for start in range(0, train_sample, batch_size):
            batch_indices = indices[start : start + batch_size]
            x = train_x[batch_indices]
            y = train_y[batch_indices]
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            total_loss += loss.item() * len(batch_indices)

        mean_loss = total_loss / train_sample
        print("=========\n第 %d 轮平均 loss：%.6f" % (epoch + 1, mean_loss))
        accuracy = evaluate(model, test_x, test_y)
        log.append([accuracy, mean_loss])

    # 输出位置相对于脚本，而不是运行命令时所在的目录
    output_dir = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "argmax_model.bin"
    curve_path = output_dir / "training_curve.png"
    torch.save(model.state_dict(), model_path)

    fig, ax = plt.subplots()
    epochs = range(1, epoch_num + 1)
    ax.plot(epochs, [item[0] for item in log], label="accuracy")
    ax.plot(epochs, [item[1] for item in log], label="loss")
    ax.set_xlabel("Epoch")
    ax.legend()
    fig.tight_layout()
    fig.savefig(curve_path)
    plt.close(fig)
    print("模型已保存到：%s" % model_path)
    print("训练曲线已保存到：%s" % curve_path)
    return model_path


# 支持一个五维向量，或者由多个五维向量组成的列表
def predict(model_path, input_vec):
    model = TorchModel(INPUT_SIZE)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    inputs = torch.tensor(input_vec, dtype=torch.float32)
    if inputs.ndim == 1:
        inputs = inputs.unsqueeze(0)
    if inputs.ndim != 2 or inputs.shape[1] != INPUT_SIZE:
        raise ValueError("输入必须是一个五维向量，或由多个五维向量组成的列表")

    with torch.no_grad():
        probabilities = torch.softmax(model(inputs), dim=1)
        confidence, predictions = probabilities.max(dim=1)

    for vec, category, probability in zip(inputs.tolist(), predictions.tolist(), confidence.tolist()):
        print("输入：%s，预测类别：%d，概率值：%.6f" % (vec, category + 1, probability))
    return (predictions + 1).tolist()


if __name__ == "__main__":
    saved_model_path = main()
    test_vec = [
        [0.81889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
        [0.94963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
        [0.90797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
        [0.39349776, 0.59416669, 0.92579291, 0.41567412, 0.8358894],
        [0.1, 0.9, 0.3, 0.4, 0.5],
        [0.1, 0.2, 0.3, 0.9, 0.5],
    ]
    predict(saved_model_path, test_vec)
