import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 数据集
traindata = torchvision.datasets.MNIST(root='./data', train=True, transform=torchvision.transforms.ToTensor(),
                                       download=True)
trainloader = torch.utils.data.DataLoader(traindata, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                     transform=torchvision.transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.H1 = nn.Conv2d(1, 4, 5)  # 输入1 输出4，卷积核5*5 16*16->28*28->24*24
        # self.hh = nn.Conv1d(1, 4, 5);
        self.H2 = nn.AvgPool2d(2, 2)  # 池化 24*24 -> 12*12
        self.H3 = nn.Conv2d(4, 12, 5)  # 输入4 输出12 卷积核5*5 12*12->8*8
        self.H4 = nn.AvgPool2d(2, 2)  # 池化  8*8 ->4*4
        self.Output = nn.Linear(12 * 4 * 4, 10)

    def forward(self, x):
        x = F.relu(self.H1(x))
        x = self.H2(x)
        x = F.relu(self.H3(x))
        x = self.H4(x)
        x = x.view(-1, 12 * 4 * 4)
        x = self.Output(x)
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net()
net.to(device)
# 定义损失函数 采用交叉熵损失函数和随机梯度下降优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(3):
    running_loss = 0.0
    acc = 0.0
    # 按批次迭代训练模型
    for i, data in enumerate(trainloader, 0):
        # 从data中取出含有输入图像的张量inputs,标签张量labels
        inputs, labels = data[0].to(device), data[1].to(device)
        # 第一步将梯度清零
        optimizer.zero_grad()
        # 第二步将输入图像进入网络中,得到输出张量
        outputs = net(inputs)
        # 计算损失值
        loss = criterion(outputs, labels)
        # 进行反向传播和梯度更新
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i + 1) % 1000 == 0:
            # loss_avg = running_loss / (i + 1)
            # acc_avg = float(acc / ((i + 1) * 64))
            # print('Epoch', epoch + 1, ',step', i + 1, '| Loss_avg: %.4f' % loss_avg, '|Acc_avg:%.4f' % acc_avg)
            print('[%d,%5d] loss:%.3f' % (epoch + 1, i + 1, running_loss / i + 1))
            running_loss = 0.0
torch.save(net.state_dict(), './LeNet.pth')

net = Net()
net.to(device)
# 加载训练阶段保存好的模型的状态字典
net.load_state_dict(torch.load('./LeNet.pth'))
classes = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine')
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the 10K test images:%d %%' % (100 * correct / total))
# 为了更加细致的看一下模型在那些类别上表现更好，在那些类别上表现更差,我们分类别的进行准确率计算
# class_correct = list(0. for i in range(10))
# class_total = list(0. for i in range(10))
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data[0].to(device), data[1].to(device)
#         outputs = net(images)
#         _, predicted = torch.max(outputs, 1)
#         c = (predicted == labels).squeeze()
#         for i in range(4):
#             label = labels[i]
#             class_correct[label] += c[i].item()
#             class_total[label] += 1
# # 打印不同类别的准确率
# for i in range(10):
#     print('Accuracy of %5s: %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
