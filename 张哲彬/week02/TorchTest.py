import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class TestModel(nn.Module):
    def __init__(self, input_size):
        super(TestModel, self).__init__()
        self.linear1 = nn.Linear(input_size, 16)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(16, 5)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y = None):
        y_pred = self.linear1(x)
        # 交叉熵损失函数中已经包含了归一化处理，不需要sigmoid或softmax，一层线性网络可以不用激活函数
        y_pred = self.activation(y_pred)
        y_pred = self.linear2(y_pred)
        if y is not None:
            loss = self.loss(y_pred, y)
            return loss
        else:
            return y_pred

def test(data):
    #加载保存本地的模型
    model = TestModel(5)
    model.load_state_dict(torch.load("test_model.pth"))
    model.eval()
    with torch.no_grad():
        x = torch.tensor(data, dtype=torch.float32)
        y_pred = model(x, None)
        max_index = torch.argmax(y_pred, dim=1)
        return max_index

def evaluate(model):
    model.eval()
    test_x, test_y = build_dataset(100)
    with torch.no_grad():
        y_pred = model(test_x, None)
        max_index = torch.argmax(y_pred, dim=1)
        test_y = torch.argmax(test_y, dim=1)
        acc = (max_index == test_y).float().mean()
        return acc

def generate_sample():
    x = np.random.rand(5)
    # 输入5维向量中最大值的索引位置为1，其余为0
    y = np.zeros(5)
    y[np.argmax(x)] = 1
    return x, y

def build_dataset(dataset_num):
    train_x, train_y = [], []
    for i in range(dataset_num):
        x, y = generate_sample()
        train_x.append(x)
        train_y.append(y)
    return torch.tensor(train_x, dtype=torch.float32), torch.tensor(train_y, dtype=torch.float32)

def main():
    epoch_num = 50       #训练轮数
    batch_size = 20      #梯度更新一次的样本数
    dataset_num = 5000   #训练数据数量
    input_size = 5       #输入向量维度
    learning_rate = 0.01 #学习率
    train_x, train_y = build_dataset(dataset_num)
    model = TestModel(input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    result = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(dataset_num // batch_size):
            x = train_x[batch_index * batch_size:(batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size:(batch_index + 1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            watch_loss.append(loss.item())
        acc = evaluate(model)
        print(f"epoch: {epoch+1}, loss: {loss.item()}, acc: {acc}")
        result.append([acc, np.mean(watch_loss)])
    # 保存模型
    torch.save(model.state_dict(), "test_model.pth")
    plt.plot(range(epoch_num), [r[0] for r in result], label="acc")
    plt.plot(range(epoch_num), [r[1] for r in result], label="loss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
    test_data = np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                          [0.5, 0.4, 0.3, 0.2, 0.1],
                          [0.1, 0.3, 0.2, 0.4, 0.5]])
    result = test(test_data)
    print("test result:", result)   