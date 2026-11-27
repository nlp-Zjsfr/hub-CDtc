# coding:utf8

# 解决 OpenMP 库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，最大值落在第几维，样本就属于第几类（类别编号 0~4）
这是一个5分类任务，损失使用交叉熵

"""


class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes=5):
        super(TorchModel, self).__init__()
        # 线性层：5维输入 -> 5个类别的原始分数(logits)
        # CrossEntropyLoss 内部会做 softmax，这里不要再加 sigmoid / softmax
        self.linear = nn.Linear(input_size, num_classes)
        self.loss = nn.functional.cross_entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测logits
    def forward(self, x, y=None):
        y_pred = self.linear(x)  # (batch_size, 5)
        if y is not None:
            return self.loss(y_pred, y)  # 交叉熵要求 y 是 LongTensor，形状 (batch_size,)
        else:
            return y_pred  # 输出5个类别的logits，预测时再argmax


# 生成一个样本：随机5维向量，标签=最大值所在的下标
def build_sample():
    x = np.random.random(5)
    y = int(np.argmax(x))  # 0~4
    return x, y


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)  # 多分类标签是整数，不要再包成 [y]
    return torch.FloatTensor(np.array(X)), torch.LongTensor(Y)


# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    # 统计测试集中5类各有多少
    class_counts = [(y == i).sum().item() for i in range(5)]
    print("本次预测集中各类样本数：", class_counts)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # (100, 5) logits
        y_pred_class = torch.argmax(y_pred, dim=-1)  # 每行取最大分数对应的类别
        for y_p, y_t in zip(y_pred_class, y):
            if int(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.01  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))
    print(model.state_dict())

    model.eval()
    with torch.no_grad():
        logits = model.forward(torch.FloatTensor(input_vec))  # (n, 5)
        probs = torch.softmax(logits, dim=-1)  # 转成各类概率，仅用于展示
        pred_class = torch.argmax(logits, dim=-1)
    for vec, cls, prob in zip(input_vec, pred_class, probs):
        print("输入：%s, 真实最大值维：%d, 预测类别：%d, 各类概率：%s" % (
            vec,
            int(np.argmax(vec)),
            int(cls),
            [round(p, 4) for p in prob.tolist()]
        ))


if __name__ == "__main__":
    main()
    test_vec = [[0.81889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
                [0.94963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
                [0.90797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
                [0.39349776, 0.59416669, 0.92579291, 0.41567412, 0.8358894]]
    predict("model.bin", test_vec)
