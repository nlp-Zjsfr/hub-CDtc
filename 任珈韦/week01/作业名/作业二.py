# coding:utf8
import os
import sys

# ==================== 运行环境固定为 py312 ====================
# 本文件必须在 conda 环境 py312 (D:\anaconda3\envs\py312, python 3.12 + torch 2.6.0) 下运行。
# 如果当前解释器不是 py312 (比如系统默认的 base python 3.14、没装 torch 的那个)，
# 就自动用 py312 的解释器重新执行本脚本，这样双击 / IDE 默认配置 / 命令行都不会跑错环境。
# 注：PowerShell 执行策略为 Restricted，conda activate 用不了，所以这里直接按路径切换。
PY312_PYTHON = r"D:\anaconda3\envs\py312\python.exe"
REEXEC_FLAG = "COIN_PY312_REEXEC"

if (os.path.abspath(sys.executable).lower() != os.path.abspath(PY312_PYTHON).lower()
        and os.path.exists(PY312_PYTHON)
        and not os.environ.get(REEXEC_FLAG)):
    os.environ[REEXEC_FLAG] = "1"  # 防止切换失败时来回递归
    try:
        os.execv(PY312_PYTHON, [PY312_PYTHON, os.path.abspath(__file__)] + sys.argv[1:])
    except OSError:
        import subprocess
        sys.exit(subprocess.call([PY312_PYTHON, os.path.abspath(__file__)] + sys.argv[1:]))

print("[env] 当前解释器: %s" % sys.executable)

# 解决 OpenMP 库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt
"""
基于pytorch框架编写模型训练
实现一个自行构造的抛硬币(机器学习)任务
规律：把一次抛硬币过程的 5 个观测值组成 5 维向量 x
     x[0] = 抛掷力度    x[1] = 旋转速度    x[2] = 起始朝向
     x[3] = 空中时间    x[4] = 桌面平整度
硬币最后是正面朝上、反面朝上，还是立在桌面上，只由"抛掷力度"和"旋转速度"的差值决定：
     力度 - 旋转速度 >  0.1        -> 正面(类别0)
     旋转速度 - 力度 >  0.1        -> 反面(类别1)
     |力度 - 旋转速度| <= 0.1      -> 立起(类别2)  硬币几乎不翻转，立在桌面上，很少见
这是一个三分类任务：模型输出 正面 / 反面 / 立起 三种情况的概率
"""
# 三个类别的名字，下标就是类别编号
CLASS_NAMES = ["正面", "反面", "立起"]
CLASS_NUM = len(CLASS_NAMES)  # 类别数量 3
EDGE_GAP = 0.1  # 力度与转速的差距小于等于该值时，硬币立在桌面上
class TorchModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, class_num=CLASS_NUM):
        super(TorchModel, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)  # 隐层
        self.activation = torch.relu  # nn.ReLU() relu激活函数
        self.linear = nn.Linear(hidden_size, class_num)  # 输出层，3个类别各一个得分
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵损失(多分类)
    # 当输入真实标签，返回loss值；无真实标签，返回三种情况各自的概率
    def forward(self, x, y=None):
        x = self.activation(self.hidden(x))  # (batch_size, input_size) -> (batch_size, hidden_size)
        x = self.linear(x)  # (batch_size, hidden_size) -> (batch_size, 3)
        if y is not None:
            # 交叉熵内部自带softmax，所以直接喂原始得分(logits)和类别下标
            return self.loss(x, y)  # 预测值和真实值计算损失
        else:
            return torch.softmax(x, dim=-1)  # (batch_size, 3) -> 正面/反面/立起 的概率
# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量代表一次抛硬币，比较第1个值(力度)和第2个值(转速)的大小
def build_sample():
    x = np.random.random(5)
    gap = x[0] - x[1]  # 抛掷力度 - 旋转速度
    if gap > EDGE_GAP:
        return x, 0  # 力度远大于转速，硬币压向正面 -> 正面
    elif gap < -EDGE_GAP:
        return x, 1  # 转速远大于力度，硬币翻到了另一面 -> 反面
    else:
        return x, 2  # 两者接近，硬币几乎不翻转，立在桌面上 -> 立起
# 随机生成一批样本
# 三种样本按规律自然生成(正面、反面各约四成多，立起约两成)
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    # print(X)
    # print(Y)
    return torch.FloatTensor(np.array(X)), torch.LongTensor(Y)
# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    counts = [(y == i).sum().item() for i in range(CLASS_NUM)]
    print("本次预测集中共有%d个正面样本，%d个反面样本，%d个立起样本" % (counts[0], counts[1], counts[2]))
    correct, wrong = 0, 0
    hit = [0] * CLASS_NUM  # 记录每一类判断正确的个数
    prob_sum = np.zeros(CLASS_NUM)  # 记录预测概率的累计和，用来观察学到的概率分布
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)，得到每一类的概率
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            prob = [float(p) for p in y_p]  # 正面/反面/立起 三个概率
            prob_sum += np.array(prob)
            pred = int(np.argmax(prob))  # 概率最大的那一类就是预测结果
            if pred == int(y_t):
                correct += 1  # 判断正确
                hit[int(y_t)] += 1
            else:
                wrong += 1
    avg_prob = prob_sum / test_sample_num
    for i in range(CLASS_NUM):
        print("  %s：真实%d个，猜对%d个，平均预测概率%f"
              % (CLASS_NAMES[i], counts[i], hit[i], avg_prob[i]))
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)
def main():
    # 配置参数
    epoch_num = 30  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 2000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    hidden_size = 64  # 隐层维度
    learning_rate = 0.01  # 学习率
    # 建立模型
    model = TorchModel(input_size, hidden_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    counts = [(train_y == i).sum().item() for i in range(CLASS_NUM)]
    print("训练集共%d个样本：正面%d，反面%d，立起%d"
          % (train_sample, counts[0], counts[1], counts[2]))
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
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测，得到三种情况的概率
    for vec, res in zip(input_vec, result):
        prob = [float(p) for p in res]
        pred = int(np.argmax(prob))
        print("输入：%s, 预测类别：%s, 概率值：正面%f 反面%f 立起%f"
              % (vec, CLASS_NAMES[pred], prob[0], prob[1], prob[2]))  # 打印结果
if __name__ == "__main__":
    main()
    # 依次是：正面、反面、立起、正面
    test_vec = [[0.81889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
                [0.15229675, 0.81889086, 0.95758807, 0.95520434, 0.84890681],
                [0.52340000, 0.49890000, 0.13625847, 0.34675372, 0.19871392],
                [0.94963533, 0.55242560, 0.95758807, 0.95520434, 0.84890681]]
    predict("model.bin", test_vec)
