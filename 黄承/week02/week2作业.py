
# 解决 OpenMP 库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

# 第二周作业：尝试完成一个多分类任务的训练:一个随机向量，哪一维数字最大就属于第几类

"""
    多分类任务
        使用交叉熵
        softmax


"""


# 假设默认最大维数限制为 100
max_dim = 100


# 模型
class TorchModel(nn.Module):
    def __init__(self, input_size) -> None:
        super(TorchModel,self).__init__()
        # 线性层
        self.linear = nn.Linear(input_size,input_size)
        # loss 交叉熵适合分类
        self.loss = nn.functional.cross_entropy

    def forward(self,x,y = None):
        # 线性层计算
        x = self.linear(x)
        if y is not None:
            return self.loss(x,y),x # 输入真实值时，计算损失
        else:
            return x

# 随机生成测试样本
def generate_sample():
    
    # 找出数字最大的维数
    sample = np.random.random(max_dim)
    max_index = np.argmax(sample)
    
    return sample,max_index

# 测试generate_sample函数
def test_generate_sample():
    sample,max_index = generate_sample()
    max_value = max(sample)
    assert max_value == sample[max_index] 
    print(f"sample: {sample}, max_index: {max_index}, max_value: {max_value}")
    print("="*50)


# 批量生成数据集
def generate_sample_batch(batch_num):
    X = []
    Y = []
    for i in range(batch_num):
        x,y = generate_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X),torch.LongTensor(Y)


# 测试集评估模型准确率
def evaluate(model,x,y):
    model.eval()
    
    
    watch_loss = []
    correct,wrong = 0,0
    with torch.no_grad():
        loss,y_pred = model(x,y)
        watch_loss.append(loss.item())
        for y_p,y_t in zip(y_pred,y):
            if torch.argmax(y_p) == y_t:
                correct +=1
            else:
                wrong +=1
    print("正确预测数：%d,正确率：%f"%(correct,correct/(correct+wrong)))
    return correct/(correct+wrong),watch_loss
# 固定测试集训练评估

# 模型训练
def training(epoch,batch,train_sample,learning_rate = 0.01):
    """
        epoch 训练轮数
        batch 每个批次样本数
        train_sample 单轮训练样本数
        learning_rate 学习率
    """
    log = []
    # 定义模型
    model = TorchModel(max_dim)
    model.load_state_dict(torch.load("model.bin"))
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(),lr = learning_rate)
    # 训练数据集
   
    train_x,train_y = generate_sample_batch(train_sample)
    # 训练

    # 测试集
    test_sample_size = 100
    test_x,test_y = generate_sample_batch(test_sample_size)

    for epoch_index in range(epoch): # 轮
        model.train()
        watch_loss = [] # 监视损失率变化
        for batch_index in range(train_sample // batch): # 批次
            # 获取本批训练数据
            x = train_x[batch_index*batch:(batch_index+1)*batch]
            y = train_y[batch_index*batch:(batch_index+1)*batch]
            loss,_ = model(x,y)
            loss.backward() # 方向传播
            optim.step() # 更新权重 梯度下降
            optim.zero_grad() # 梯度清零
            watch_loss.append(loss.item())
        print("============\n第%d轮平均loss:%f"%(epoch_index+1,np.mean(watch_loss)))

        acc,evalu_watch_loss = evaluate(model,test_x,test_y)

        log.append([acc,float(np.mean(watch_loss)),float(np.mean(evalu_watch_loss))])

    torch.save(model.state_dict(),"model.bin")
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="train_loss")  # 画loss曲线
    plt.plot(range(len(log)), [l[2] for l in log], label="test_loss")  # 画测试loss曲线
    plt.legend()
    plt.show()

def predict(cnt):
    for i in range(cnt):
        test_sample_size = 100
        x_,y_ = generate_sample_batch(test_sample_size)
        model = TorchModel(max_dim)
        model.load_state_dict(torch.load("model.bin"))
        acc,evalu_watch_loss = evaluate(model,x_,y_)
        print(evalu_watch_loss)
   

# training( 40,2**15,2**20,0.000005)
predict(5)

    











