'''
多分类任务完成模型训练:任意5为向量输入，对最大值进行分类输出
'''
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
# ========== 1. 生成自定义规律的多分类数据集 ==========
def generate_data(num_samples=5000):
    # 生成指定数量的5维样本，数据自带“最大值维度即类别”规律
    X = np.random.randn(num_samples, 5)  # 生成标准正态分布的随机5维特征数据
    # 计算每行（单个样本）最大值所在的索引，索引值即为该样本的类别标签
    y = np.argmax(X, axis=1)# axis=1表示按行取最大值索引
    return X, y

# 调用函数生成数据集，50000个训练样本
X, y = generate_data(50000)
# 将numpy数组转换为PyTorch张量，适配模型计算
X_tensor = torch.FloatTensor(X)  # 特征数据转为浮点型张量
y_tensor = torch.LongTensor(y)   # 标签转为长整型张量，适配多分类任务

# ========== 2. 划分训练集和测试集 ==========
# 按3:1划分训练集和测试集，固定随机种子保证结果可复现
train_ratio = 0.75
num_samples = X_tensor.shape[0]

# 随机生成索引，划分训练集与测试集索引
indices = torch.randperm(num_samples)
train_num = int(num_samples * train_ratio)
train_indices = indices[:train_num]
test_indices = indices[train_num:]

# 提取训练集和测试集数据
X_train, y_train = X_tensor[train_indices], y_tensor[train_indices]
X_test, y_test = X_tensor[test_indices], y_tensor[test_indices]

# ========== 3. 构建分类神经网络模型 ==========
class RuleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):#定义输入维度输出分类
        super(RuleClassifier, self).__init__()
        # 隐藏层1：输入维度为5，隐藏层神经元100，激活函数用ReLU提升非线性表达能力
        self.fc1 = nn.Linear(input_dim, 100)
        self.relu = nn.ReLU()
        # 隐藏层2：输入100，输出50，再次增强模型拟合能力
        self.fc2 = nn.Linear(100, 50)
        # 输出层：输出维度等于类别数5，输出各类别原始得分
        self.fc3 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = self.fc1(x)  # 第一层全连接计算
        x = self.relu(x) # 激活函数，引入非线性变换
        x = self.fc2(x)  # 第二层全连接计算
        x = self.relu(x) # 再次激活
        x = self.fc3(x)  # 输出层计算，输出5类原始得分
        return x         # 返回前向传播结果

# 实例化模型，输入维度5，分类数量5
model = RuleClassifier(input_dim=5, num_classes=5)

# ========== 4. 定义损失函数与优化器 ==========
# 使用交叉熵损失，多分类标准损失函数，自动适配softmax概率计算
criterion = nn.CrossEntropyLoss()
# 选用Adam优化器，自适应调整学习率，加快模型收敛速度
optimizer = optim.Adam(model.parameters(), lr=0.05)

# ========== 5. 模型训练流程 ==========
def train_model(model, train_x, train_y, num_epochs=160):
    model.train()  # 切换模型为训练模式，启用训练相关机制
    for epoch in range(num_epochs):
        # 单轮全量数据训练
        output = model(train_x)          # 模型前向传播，输出原始预测值
        loss = criterion(output, train_y)  # 计算交叉熵损失，对比真实标签
        optimizer.zero_grad()           # 清空历史梯度，避免梯度累加
        loss.backward()                 # 反向传播，自动计算模型参数梯度
        optimizer.step()                # 根据梯度更新模型参数
        # 每10轮打印一次损失，监控训练进度
        if (epoch + 1) % 10 == 0:
            print(f"训练轮次：{epoch + 1}，损失值：{loss.item():.4f}")
            #print(loss)
    # 保存模型
    torch.save(model.state_dict(), "model_rule.bin")

# 启动模型训练
train_model(model, X_train, y_train)

# ========== 6. 模型评估，验证是否学会规律 ==========
def evaluate(model, test_x, test_y):
    model.eval()  # 切换为评估模式，关闭训练相关操作
    with torch.no_grad():  # 关闭梯度计算，提升推理速度、节省内存
        output = model(test_x)  # 模型前向传播，获取预测得分
        # softmax归一化，将原始得分转为概率值，概率总和为1
        probabilities = torch.softmax(output, dim=1)
        # 取概率最大的维度索引作为预测类别，严格遵循“最大维度即类别”规律
        pred_classes = torch.argmax(probabilities, dim=1)
    # 计算测试集准确率，判断模型是否掌握核心规律
    accuracy = torch.sum(pred_classes == test_y).item() / test_y.shape[0]
    print(f"\n模型测试集准确率：{accuracy:.4f}")
    return pred_classes, probabilities

# 执行测试集评估
pred, test_prob = evaluate(model, X_test, y_test)

# ========== 7. 单样本预测，验证规律匹配度 ==========
def predict_single_sample(model_path, sample):
    # 构造单个5维样本，手动设置最大值维度，测试模型预测结果
    #sample_tensor = torch.FloatTensor(sample).unsqueeze(0)  # 增加维度适配模型输入格式
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    model.eval()
    with torch.no_grad():
        #output = model(sample)        # 前向传播获取预测得分
        output = model.forward(torch.FloatTensor(sample).unsqueeze(0)) #模型预测
        probs = torch.softmax(output, dim=1)# softmax转为概率
        pred_label = torch.argmax(probs, dim=1).item() + 1  # 取最大维度索引作为类别
    return pred_label, probs.numpy()[0]

# 测试样本1：最大值在第3维，真实类别应为3
sample1 = [3.2, -0.5, 3.3, 2.6, -0.2]
#pred1, prob1 = predict_single_sample(model, sample1)
pred1,prob1 = predict_single_sample("model_rule.bin",sample1)
print(f"\n测试样本1：{sample1}")
print(f"预测类别：{pred1}，各类别概率：{prob1}")

# 测试样本2：最大值在第4维，真实类别应为4
sample2 = [-0.8, 0.2, 0.5, 2.3, 1.9]
#pred2, prob2 = predict_single_sample(model, sample2)
pred2, prob2 = predict_single_sample("model_rule.bin", sample2)
print(f"测试样本2：{sample2}")
print(f"预测类别：第{pred2}类，各类别概率：{prob2}")
