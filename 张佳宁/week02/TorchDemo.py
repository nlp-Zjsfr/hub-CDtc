# coding:utf8

# 解决 OpenMP 库冲突问题
import os
from unicodedata import category

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务

"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层
        self.loss = nn.functional.cross_entropy  # loss函数采用均方差损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        y_pred = self.linear(x)  # (batch_size, input_size) -> (batch_size, 5)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，如果第一个值大于第五个值，认为是正样本，反之为负样本
def build_sample():
    x = np.random.random(5)
    # if x[0] > x[4]:
    #     return x, 1
    # else:
    #     return x, 0
    # 找的随机数组中最大数值的索引
    i = np.argmax(x)
    # 获得一个长度为5 所有元素都为0数组
    y = np.zeros(5)
    # 将对应的索引改为1
    y[i] = 1
    return x, y


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    # print(X,'======')
    # print(Y,'======')
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    return X, Y


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    category_list = [0, 0, 0, 0, 0]
    print("各类统计为：")
    for categories in y:
        for i, value in enumerate(categories):
            if value == 1:
                category_list[i] += 1
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        correct = 0
        wrong_list = [0, 0, 0, 0, 0]
        for y_p, y_t in zip(y_pred, y):
            i_p = torch.argmax(y_p)
            i_t = torch.argmax(y_t)
            if i_p == i_t:
                correct += 1
            else:
                wrong_list[i_t] += 1

        # 计算正确率
        accuracy = correct / len(y)
        for i, value in enumerate(category_list):
            wrong_count = wrong_list[i]
            t = 0
            if value > 0:
                t = (value - wrong_count) / value
            else:
                t = 0
            print(
                f"{i + 1}类：{value}个，预测正确: {value - wrong_count}，预测错误：{wrong_count}，正确率为：{t:.2%}"
            )
        print(f"正确数量：{correct}")
        print(f"正确率：{accuracy:.2%}")
    return accuracy


# 获取当前 Python 文件所在的目录
BASE_DIR = Path(__file__).resolve().parent

# 拼接模型保存路径
model_path = BASE_DIR / "model.bin"


def main():
    # 配置参数
    epoch_num = 30  # 训练轮数
    batch_size = 25  # 每次训练样本个数
    train_sample = 2500  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.1  # 学习率
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
    # # 保存模型
    # torch.save(model.state_dict(), "model.bin")

    # 保存模型
    torch.save(model.state_dict(), model_path)
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        i_p = torch.argmax(res).item()
        i_t = np.argmax(vec).item()
        print(f"输入：{vec}")
        print(f"真实类别：第 {i_t + 1} 类")
        print(f"预测类别：第 {i_p + 1} 类")
        print(f"最高得分：{res[i_p].item()}")
        print(f"全部得分：{res.tolist()}")
        print("-------------------")


if __name__ == "__main__":
    # main()
    test_vec = [
        [0.81889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
        [0.94963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
        [0.90797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
        [0.39349776, 0.59416669, 0.92579291, 0.41567412, 0.8358894],
    ]
    predict(model_path, test_vec)
