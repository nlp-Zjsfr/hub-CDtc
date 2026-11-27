"""手写数字十分类：使用 PyTorch 训练多层感知机。"""

import copy
import random

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
EPOCHS = 20
BATCH_SIZE = 32
HIDDEN_SIZES = [32, 16]
LEARNING_RATES = [0.1, 0.01]


def reset_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)


def build_model():
    return nn.Sequential(
        nn.Linear(64, HIDDEN_SIZES[0]),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZES[0], HIDDEN_SIZES[1]),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZES[1], 10),
    )


def evaluate(model, features, labels):
    model.eval()
    with torch.no_grad():
        scores = model(features)
        loss = nn.functional.cross_entropy(scores, labels).item()
        predictions = scores.argmax(dim=1)
        accuracy = (predictions == labels).float().mean().item()
    return loss, accuracy, predictions


def train_candidate(x_train, y_train, x_val, y_val, learning_rate):
    reset_seed()
    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(SEED), num_workers=0,
    )
    best_loss = float("inf")
    best_state = None
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for features, labels in loader:
            optimizer.zero_grad()
            scores = model(features)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()

        train_loss, _, _ = evaluate(model, x_train, y_train)
        val_loss, val_accuracy, _ = evaluate(model, x_val, y_val)
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % 5 == 0:
            print(f"lr={learning_rate:g}  epoch={epoch}/{EPOCHS}  "
                  f"训练loss={train_loss:.4f}  验证loss={val_loss:.4f}  "
                  f"验证准确率={val_accuracy:.2%}")

    model.load_state_dict(best_state)
    return model, best_loss, best_epoch


def main():
    torch.set_num_threads(2)
    reset_seed()
    digits = load_digits()
    features = digits.data.astype(np.float32) / 16.0
    labels = digits.target.astype(np.int64)

    # 分层划分训练集、验证集、测试集，比例约为 70%、15%、15%。
    train_indices, heldout_indices = train_test_split(
        np.arange(len(labels)), test_size=0.30,
        random_state=SEED, stratify=labels,
    )
    val_indices, test_indices = train_test_split(
        heldout_indices, test_size=0.50,
        random_state=SEED, stratify=labels[heldout_indices],
    )
    x_train = torch.from_numpy(features[train_indices])
    y_train = torch.from_numpy(labels[train_indices])
    x_val = torch.from_numpy(features[val_indices])
    y_val = torch.from_numpy(labels[val_indices])
    print(f"数据划分：训练 {len(train_indices)}，验证 {len(val_indices)}，"
          f"测试 {len(test_indices)}")

    # 只依据验证损失比较学习率及模型轮次。
    candidates = []
    for learning_rate in LEARNING_RATES:
        model, val_loss, epoch = train_candidate(
            x_train, y_train, x_val, y_val, learning_rate,
        )
        candidates.append((val_loss, model, learning_rate, epoch))
    _, selected_model, selected_lr, selected_epoch = min(
        candidates, key=lambda item: item[0],
    )

    # 模型选择结束后，使用测试集评估一次。
    x_test = torch.from_numpy(features[test_indices])
    y_test = torch.from_numpy(labels[test_indices])
    test_loss, _, predictions = evaluate(selected_model, x_test, y_test)
    correct = int((predictions == y_test).sum().item())
    print(f"\n选中学习率 {selected_lr:g}，第 {selected_epoch} 轮")
    print(f"测试损失：{test_loss:.4f}")
    print(f"测试准确率：{correct}/{len(y_test)} = {correct / len(y_test):.2%}")


if __name__ == "__main__":
    main()
