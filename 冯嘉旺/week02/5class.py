# coding:utf8

# 解决 OpenMP 库冲突问题
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，它最大数字在第几维就属于第几类

"""


class TorchModel(nn.Module):
    def __init__(self, input_size, class_num=5):
        super().__init__()
        # 输出维度为 class_num(5)：每个类别各有一个得分(logit)
        self.linear = nn.Linear(input_size, class_num)
        # 损失改为交叉熵：
        # CrossEntropyLoss 内部自带 log_softmax，输入必须是未归一化的 logits，
        # 标签必须是 0~4 的 long 类型类别下标
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # 这里不再接 sigmoid：softmax 由损失函数内部完成
        logits = self.linear(x)  # (batch_size, 5)
        if y is not None:
            return self.loss(logits, y)
        return logits


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，它最大数字在第几维就属于第几类（0-4）
def build_sample():
    x = np.random.random(5)
    # 标签是最大值所在维度(0~4)。
    # PyTorch 的类别下标必须从 0 开始；若要按第几维展示，在打印时再 +1
    return x, int(np.argmax(x))


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    # 标签用 LongTensor(整型类别下标)，形状为 (N,)；
    # 所以上面是 Y.append(y) 而不是 Y.append([y])，(N,1) 会让 CrossEntropyLoss 报维度错误
    return torch.FloatTensor(X), torch.LongTensor(Y)


def evaluate(model):
    model.eval()
    x, y = build_dataset(1000)
    with torch.no_grad():
        logits = model(x)  # (100, 5) 每个类别的原始得分
        # 取得分最大的维度作为预测类别
        pred = torch.argmax(logits, dim=1)  # (100,)
    # 整体张量比较：预测类别 == 真实类别
    correct = (pred == y).sum().item()
    acc = correct / len(y)
    print("正确个数：%d, 正确率：%f" % (correct, acc))
    return acc


def main():
    # 配置参数
    epoch_num = 50  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 2000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.01  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程(与二分类完全一致：前向 -> 反向 -> 更新 -> 梯度清零)
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            # 取出一个batch数据作为输入   train_x[0:20]  train_y[0:20] train_x[20:40]  train_y[20:40]
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
    # 打印权重矩阵：理想情况下近似单位矩阵的若干倍，第 i 行第 i 列数值最大
    print("linear.weight:\n%s" % model.linear.weight.data)
    torch.save(model.state_dict(), "model_5class.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    # 重建模型时输出类别数必须是 5，否则加载权重会报 shape 不匹配
    model = TorchModel(5)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(input_vec))  # (N,5)
        # 多分类先 softmax 得到 5 类概率分布。softmax 单调，argmax(概率) 与 argmax(logits) 等价
        probs = torch.softmax(logits, dim=1)
    for vec, p in zip(input_vec, probs):
        cls = int(torch.argmax(p))
        print("输入：%s, 预测：第%d维, 置信度：%f" % (vec, cls + 1, float(p[cls])))


if __name__ == "__main__":
    main()
    test_vec = [
        [0.81889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
        [0.94963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
        [0.90797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
        [0.39349776, 0.59416669, 0.92579291, 0.41567412, 0.8358894],
    ]
    predict("model_5class.bin", test_vec)
