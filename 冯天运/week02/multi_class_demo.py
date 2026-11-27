# coding:utf8
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_multiclass.bin")


# 多分类模型：5维向量，最大值在第几维就算第几类（0~4）
# 没用 CrossEntropyLoss，用 5 个 sigmoid 输出 + mse 逼近 one-hot 标签
class MultiClassModel(nn.Module):
    def __init__(self, input_size, num_class):
        super(MultiClassModel, self).__init__()
        self.linear = nn.Linear(input_size, num_class)
        self.activation = torch.sigmoid
        self.loss = nn.functional.mse_loss

    # 有标签就返回损失，没标签就返回预测概率
    def forward(self, x, y=None):
        x = self.linear(x)            # 线性层：5维 -> 5维
        y_pred = self.activation(x)   # sigmoid 把每维压到 (0,1)
        if y is not None:
            return self.loss(y_pred, y)   # 有标签就计算 mse 损失
        return y_pred


# 类别索引转 one-hot，如 2 -> [0,0,1,0,0]
def to_onehot(y, num_class):
    vec = [0.0] * num_class   # 先全部置 0
    vec[y] = 1.0              # 把对应类别那一维置 1
    return vec


# 造一个样本：随机5维向量，标签是最大值所在的维度
def build_sample():
    x = np.random.random(5)
    # 遍历找出最大值在第几个位置（这就是 argmax 在做的事）
    y = 0
    for i in range(5):
        if x[i] > x[y]:
            y = i
    return x, to_onehot(y, 5)


# 批量造样本，返回特征矩阵 X 和 one-hot 标签 Y
def build_dataset(n):
    X, Y = [], []
    for _ in range(n):
        x, y = build_sample()   # 造一个样本
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.FloatTensor(Y)    # 把列表转成模型要的张量


# 在 100 个随机样本上算准确率
def evaluate(model):
    model.eval()                          # 切到评估模式
    x, y = build_dataset(100)             # 造 100 个测试样本
    correct = 0
    with torch.no_grad():                 # 不记录梯度，省内存
        y_pred = model(x)                 # 模型输出每个样本 5 个概率
        for k in range(len(y_pred)):      # 逐个样本判断
            # 预测类：在 5 个概率里找最大的那一维
            pred_class = 0
            for j in range(5):
                if y_pred[k][j] > y_pred[k][pred_class]:
                    pred_class = j
            # 真实类：在 one-hot 标签里找为 1 的那一维
            true_class = 0
            for j in range(5):
                if y[k][j] > 0.5:
                    true_class = j
            if pred_class == true_class:
                correct += 1
    acc = correct / 100
    print("正确率：%f" % acc)
    return acc


def main():
    epoch_num = 50  # 训练轮数
    batch_size = 20 # 每次训练样本个数,根据GPU配置，一般默认32 64 128
    train_sample = 500 # 每轮训练总共训练的样本总数
    learning_rate = 0.01 # 学习率

    model = MultiClassModel(5, 5) #建立模型
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate) #选择优化器
    #help(torch.optim.Adamz)
    log = []

    # 先生成整份训练集
    train_x, train_y = build_dataset(train_sample)
    for epoch in range(epoch_num):
        model.train()
        losses = []
        # 按 batch 取数据训练
        for i in range(train_sample // batch_size):
            x = train_x[i * batch_size: (i + 1) * batch_size]
            y = train_y[i * batch_size: (i + 1) * batch_size]
            loss = model(x, y)   # 前向传播，算损失
            loss.backward()       # 反向传播，算梯度
            optim.step()          # 用梯度更新参数
            optim.zero_grad()     # 清空梯度，准备下一批
            losses.append(loss.item())
        print("第%d轮 loss:%f" % (epoch + 1, np.mean(losses)))
        log.append([evaluate(model), np.mean(losses)])

    # 保存训练好的权重
    torch.save(model.state_dict(), MODEL_PATH)
    # 画准确率和损失曲线
    plt.plot([l[0] for l in log], label="acc")
    plt.plot([l[1] for l in log], label="loss")
    plt.legend()
    plt.show()


def predict(model_path, input_vec):
    model = MultiClassModel(5, 5)
    model.load_state_dict(torch.load(model_path, weights_only=True))  # 载入训练好的权重
    model.eval()
    with torch.no_grad():
        res = model(torch.FloatTensor(input_vec))   # 预测每个样本 5 个概率
    for k in range(len(res)):
        # 在 5 个概率里找最大的那一维作为预测类别
        c = 0
        for j in range(5):
            if res[k][j] > res[k][c]:
                c = j
        print("输入：%s, 预测类别：%d" % (input_vec[k], int(c)))


if __name__ == "__main__":
    main()
    # predict(MODEL_PATH, [[0.1, 0.9, 0.3, 0.2, 0.5],
    #                       [0.8, 0.2, 0.6, 0.4, 0.1]])
