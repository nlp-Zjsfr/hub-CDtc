from sklearn.datasets import load_iris

iris = load_iris()
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

经典鸢尾花分类任务

"""


class TorchModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)  # 线性层
        self.activation = torch.softmax  # 激活函数使用softmax
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        logits = self.linear(x)  # (batch_size, 5)
        if y is not None:
            return self.loss(logits, y)  # 训练：原始分数计算损失
        else:
            return self.activation(logits, dim=1)  # 预测：转换为概率


def build_dataset():
    # 每类前40个用于训练，共120个
    X = np.concatenate([
        iris.data[0:40],
        iris.data[50:90],
        iris.data[100:140]
    ])
    Y = np.concatenate([
        iris.target[0:40],
        iris.target[50:90],
        iris.target[100:140]
    ])

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.long)
    Y = nn.functional.one_hot(Y, num_classes=3).float()
    return X, Y


def build_test_set():
    # 每类后10个用于验证，共30个
    X = np.concatenate([
        iris.data[40:50],
        iris.data[90:100],
        iris.data[140:150]
    ])
    Y = np.concatenate([
        iris.target[40:50],
        iris.target[90:100],
        iris.target[140:150]
    ])

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.long)
    Y = nn.functional.one_hot(Y, num_classes=3).float()
    return X, Y


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    x, y = build_test_set()
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if np.argmax(y_p) == np.argmax(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    # 配置参数
    train_sample = 120
    epoch_num = 100  # 训练轮数
    batch_size = 10  # 每次训练样本个数
    learning_rate = 0.01  # 学习率
    # 建立模型
    model = TorchModel(4,3)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset()
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            #取出一个batch数据作为输入   train_x[0:20]  train_y[0:20] train_x[20:40]  train_y[20:40]
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "myModel2.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return

if __name__ == '__main__':
    main()