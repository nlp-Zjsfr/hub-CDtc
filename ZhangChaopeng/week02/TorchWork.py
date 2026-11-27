import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
将 Sigmoid 修改为 多分类交叉熵任务
5分类任务：5维向量，最大值在第几维，就属于第几类
"""

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层
        self.loss = nn.CrossEntropyLoss()  # 损失函数为交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 5)
        if y is not None:
            return self.loss(x, y)  # 预测值和真实值计算损失
        else:
            return x  # 输出预测结果 (batch_size, 5)

# 生成样本：5维随机向量，最大值下标即为类别标签
def build_sample():
    x = np.random.random(5)
    y = np.argmax(x)  # 最大值的索引作为类别标签
    return x, y

# 批量生成样本数据集
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)  # 注意标签类型为 LongTensor 

# 评估函数，计算模型在测试集上的准确率
# def evaluate(model, X_test, Y_test):
#     model.eval()  # 设置模型为评估模式
#     with torch.no_grad():  # 不计算梯度
#         outputs = model(X_test)  # 获取模型输出
#         _, predicted = torch.max(outputs, 1)  # 获取预测类别
#         correct = (predicted == Y_test).sum().item()  # 计算正确预测的数量
#         accuracy = correct / Y_test.size(0)  # 计算准确率
#     return accuracy
def evaluate(model):
    model.eval()
    test_sample_num = 100
    X_test, Y_test = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        outputs = model(X_test)
        pred = torch.argmax(outputs, dim=1)
        for y_pred, y_true in zip(pred, Y_test):
            if y_pred == y_true:
                correct += 1
            else:
                wrong += 1
    accuracy = correct / (correct + wrong)
    print(f"本次预测集中共有 {correct} 个正确预测，{wrong} 个错误预测，准确率为 {accuracy:.2%}")
    return accuracy

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
    torch.save(model.state_dict(), "model.bin")
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

    # with torch.no_grad():  # 不计算梯度
    #     result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    # for vec, res in zip(input_vec, result):
    #     print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果

    # 编号对应电影名
    movie_map = {
        0: "动作片",
        1: "喜剧片",
        2: "恐怖片",
        3: "科幻片",
        4: "剧情片"
    }
    movie_names = ["动作片", "喜剧片", "恐怖片", "科幻片", "剧情片"]
    with torch.no_grad():
        logits = model(torch.FloatTensor(input_vec))
        pred = torch.argmax(logits, dim=1)
    for vec, res, logit in zip(input_vec, pred, logits):
        movie_name = movie_map[int(res)]
        # 模型输出logits
        logit_text = " | ".join([f"{movie_names[i]}:{logit[i]:.3f}" for i in range(5)])
        print(f"【模型输出Logits】：{logit_text}")
        print(f"用户偏好打分向量{np.round(vec,3)} → 预测最喜欢电影编号：{int(res)} ({movie_name})")


if __name__ == "__main__":
    # main()

    # 测试样本：用户对5部电影的偏好打分 [动作片,喜剧片,恐怖片,科幻片,剧情片]
    test_users_score = [
        [0.1, 0.85, 0.3, 0.5, 0.2],    # 用户A：喜剧片分数最高，预测喜剧片
        [0.92, 0.2, 0.15, 0.4, 0.3],   # 用户B：动作片分数最高，预测动作片
        [0.2, 0.1, 0.15, 0.1, 0.88],    # 用户C：剧情片分数最高，预测剧情片
        [0.4, 0.4, 0.9, 0.3, 0.2]
    ]
    predict("model.bin", test_users_score)
