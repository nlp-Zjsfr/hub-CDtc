# coding: utf-8
"""五维输入的五分类：哪个位置的数字最大，就属于第几类（内部标签 0~4，展示 1~5）。"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


MODEL_PATH = Path(__file__).with_name("torch_demo_cross_entropy.bin")
INPUT_SIZE = 5
CLASS_NUM = 5


class TorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(INPUT_SIZE, CLASS_NUM)
        # cross_entropy 接收原始 logits (N, 5) 和 long 类型类别编号 (N,)，内部含 softmax
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        logits = self.linear(x)
        if y is not None:
            return self.loss(logits, y)
        return logits


def build_sample():
    x = np.random.random(INPUT_SIZE).astype(np.float32)
    y = int(np.argmax(x))
    return x, y


def build_dataset(total_sample_num):
    X, Y = [], []
    for _ in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    X = torch.tensor(np.array(X), dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.long)
    return X, Y


def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        predictions = model(x).argmax(dim=1)
        accuracy = (predictions == y).float().mean().item()
    return accuracy


def main(model_path=MODEL_PATH):
    np.random.seed(42)
    torch.manual_seed(42)

    epoch_num = 30
    batch_size = 64
    train_sample_num = 5000
    learning_rate = 0.03

    model = TorchModel()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_x, train_y = build_dataset(train_sample_num)
    validation_x, validation_y = build_dataset(1000)

    print("训练输入形状：", tuple(train_x.shape), "标签形状：", tuple(train_y.shape))
    print(f"训练前验证准确率：{evaluate(model, validation_x, validation_y):.2%}")

    for epoch in range(epoch_num):
        model.train()
        total_loss = 0.0
        indices = torch.randperm(train_sample_num)
        for start in range(0, train_sample_num, batch_size):
            batch_indices = indices[start:start + batch_size]
            x = train_x[batch_indices]
            y = train_y[batch_indices]
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            total_loss += loss.item() * len(batch_indices)

        accuracy = evaluate(model, validation_x, validation_y)
        print(f"第 {epoch + 1:02d} 轮 | 平均交叉熵：{total_loss / train_sample_num:.4f}"
              f" | 验证准确率：{accuracy:.2%}")

    test_x, test_y = build_dataset(2000)
    print(f"最终独立测试准确率：{evaluate(model, test_x, test_y):.2%}")
    torch.save(model.state_dict(), model_path)
    print("权重已保存到：", Path(model_path).resolve())
    return model


def predict(model_path, input_vec):
    x = torch.tensor(input_vec, dtype=torch.float32)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != INPUT_SIZE:
        raise ValueError("请传入非空二维列表，每行恰好 5 个数字，例如 [[0.1, 0.8, 0.3, 0.2, 0.4]]。")
    if not torch.isfinite(x).all().item():
        raise ValueError("输入必须是有限数字，不能包含 NaN 或无穷大。")

    model = TorchModel()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1)

    for vec, prediction, row in zip(input_vec, predictions, probabilities):
        category = prediction.item() + 1
        print(f"输入：{vec} -> 预测第 {category} 类")
        print("第 1~5 类的概率：", [round(value, 4) for value in row.tolist()])
    return predictions, probabilities


if __name__ == "__main__":
    main()
    test_vec = [
        [0.9, 0.1, 0.2, 0.3, 0.4],
        [0.1, 0.9, 0.2, 0.3, 0.4],
        [0.1, 0.2, 0.9, 0.3, 0.4],
        [0.1, 0.2, 0.3, 0.9, 0.4],
        [0.1, 0.2, 0.3, 0.4, 0.9],
    ]
    predict(MODEL_PATH, test_vec)
