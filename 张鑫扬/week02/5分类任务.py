"""

分类任务：x是一个5维向量，哪个维度数值最大，则属于哪一类，
如[0.8,0.1,0.1,0.2,0.3] -> [1,0,0,0,0]

"""
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


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, input_size)  # 线性层
        self.activation = torch.softmax  # 激活函数使用softmax
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        logits = self.linear(x)  # (batch_size, 5)
        if y is not None:
            return self.loss(logits, y)  # 训练：原始分数计算损失
        else:
            return self.activation(logits, dim=1)  # 预测：转换为概率

# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 分类任务，随机生成一个5维向量，哪一维的值最大，就属于哪一类
def build_sample():
    x = np.random.random(5)
    y = np.zeros(5)
    y[np.argmax(x)] = 1
    return x,y

def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    # print(X)
    # print(Y)
    return (
        torch.from_numpy(np.array(X, dtype=np.float32)),
        torch.from_numpy(np.array(Y, dtype=np.float32))
    )

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
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

def plot_parameters(model):
    weight = model.linear.weight.detach().cpu().numpy()
    bias = model.linear.bias.detach().cpu().numpy()

    fig, axes = plt.subplots(
        1, 2, figsize=(9, 5),
        gridspec_kw={"width_ratios": [5, 1]}
    )

    # 共用颜色范围，便于比较权重和偏置
    limit = max(float(abs(weight).max()), float(abs(bias).max()))
    limit = max(limit, 1e-8)

    def draw(ax, data, title, xlabel):
        rows, cols = data.shape
        im = ax.imshow(
            data, cmap="coolwarm",
            vmin=-limit, vmax=limit,
            aspect="auto"
        )

        for row in range(rows):
            for col in range(cols):
                ax.text(
                    col, row, f"{data[row, col]:.3f}",
                    ha="center", va="center"
                )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Output class")
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        return im

    draw(axes[0], weight, "Weight (5 × 5)", "Input feature")
    im = draw(axes[1], bias.reshape(-1, 1), "Bias", "")
    axes[1].set_xticklabels(["bias"])

    fig.tight_layout()
    fig.colorbar(im, ax=axes.tolist(), label="Parameter value")
    plt.show()

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
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
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
    torch.save(model.state_dict(), "myModel1.bin")
    # 画出训练好的模型参数参数
    plot_parameters(model)
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return

if __name__ == '__main__':
    main()