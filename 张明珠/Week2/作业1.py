import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class TorchModel(nn.Module):
    def __init__ (self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)
        # self.activation = torch.softmax
        self.loss = nn.CrossEntropyLoss()

    def forward(self,x,y=None):
        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred,y)
        else:
            return y_pred

def build_sample():
    x = np.random.random(5)
    y = np.argmax(x)
    return x, y

def build_dateset(total_sample_size):
    X = []
    Y = []
    for i in range(total_sample_size):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

def evaluate(model):
    model.eval()
    test_sample_size = 50
    test_x, test_y = build_dateset(test_sample_size)

    with torch.no_grad():
        y_pred = model(test_x)
        correct = (torch.argmax(y_pred, dim=-1) == test_y).sum().item()
        accuracy = correct / test_sample_size
    return accuracy

def main():
    epoch_num = 50
    batch_size = 20
    tran_sample_size = 5000
    input_size = 5
    learning_rate = 0.1
    model = TorchModel(input_size)
    optim = torch.optim.Adam(model.parameters(), lr= learning_rate)
    train_x, train_y = build_dateset(tran_sample_size)
    test_acc_history   = []
    train_watch_loss= []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(tran_sample_size // batch_size):

            batch_x = train_x[batch_index * batch_size:(batch_index + 1) * batch_size]
            batch_y = train_y[batch_index * batch_size:(batch_index + 1) * batch_size]
            loss = model(batch_x, batch_y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        accuracy = evaluate(model)
        test_acc_history.append(accuracy)
        print("第%2d轮 | 训练loss: %.4f| 测试准确率: %.2f%%"% (epoch + 1, np.mean(watch_loss), accuracy * 100))
        train_watch_loss.append(np.mean(watch_loss))

    torch.save(model.state_dict(),'model.bin')

    plt.plot(range(len(train_watch_loss)), train_watch_loss, label='train loss')
    plt.plot(range(len(test_acc_history)), test_acc_history, label='test acc', color='green')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
    return

def predict(model_path, input_vec):
    input_size = 5
    model =TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))
    print(model.state_dict())

    model.eval()
    with torch.no_grad():
        y_pred = model.forward(torch.FloatTensor(input_vec))
    for vec, result in zip(input_vec, y_pred):
        pred_class = torch.argmax(result).item()
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, pred_class, result[pred_class]))

if __name__ == '__main__':
    # main()
    test_vec = [[0.81889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.39349776,0.59416669,0.92579291,0.41567412,0.8358894]]
    predict("model.bin", test_vec)
