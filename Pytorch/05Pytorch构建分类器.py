"""
训练分类器的步骤
·1：使用torchvision 下载CIFAR10数据集
·2：定义卷积神经网络
·3：定义损失函数
·4：在训练集上训练模型
·5：在测试集上测试模型
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# 1:使用torchvision下载数据集
# 下载数据集并对图片进行调整,因为torchvision数据集的输出PILImage格式，数据域在[0,1],我们将其转换为标准数据域[-1,1]的张量格式
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# 迭代器loader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 构建展示图片的函数
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


"""
# 展示若干训练集图片
# 从数据迭代器中读取一张图片
dataiter = iter(trainloader)
images, labels = dataiter.next()

# 展示图片
imshow(torchvision.utils.make_grid(images))
# 打印标签
print(''.join('%5s' % classes[labels[j]] for j in range(4)))
"""


# 2定义卷积神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义连个卷积层
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 定义池化层
        self.pool = nn.MaxPool2d(2, 2)
        # 定义三个全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # 变换x的形状以适配全连接层的输入
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
net.to(device)
print(net)
# 3定义损失函数 采用交叉熵损失函数和随机梯度下降优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 4在训练集上训练模型，编写训练代码
for epoch in range(2):
    running_loss = 0.0
    # 按批次迭代训练模型
    for i, data in enumerate(trainloader, 0):
        # print('i=', i)
        # print('data=', data)
        # 从data中取出含有输入图像的张量inputs,标签张量labels
        inputs, labels = data[0].to(device),data[1].to(device)
        # 第一步将梯度清零
        optimizer.zero_grad()
        # 第二步将输入图像进入网络中,得到输出张量
        outputs = net(inputs)
        # 计算损失值
        loss = criterion(outputs, labels)
        # 进行反向传播和梯度更新
        loss.backward()
        optimizer.step()
        # 打印训练信息
        running_loss += loss.item()
        if (i + 1) % 2000 == 0:
            print('[%d,%5d] loss:%.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training')

# 设定报存位置
PATH = './cifar_net.pth'
# 保存模型的状态字典
torch.save(net.state_dict(), PATH)
# 5测试训练集
# 在测试集中取出一个批次的数据,做图像和标签的展示
# dataiter = iter(testloader)
# images, labels = dataiter.next()
# # 打印原始图片
# imshow(torchvision.utils.make_grid(images))
# # 打印真实的标签
# print('GroundTruth:', ''.join('%5s' % classes[labels[j]] for j in range(4)))
#
# # 加载模型并对测试图片进行预测
# 首先实例化模型的类对象
net = Net()
net.to(device)
# 加载训练阶段保存好的模型的状态字典
net.load_state_dict(torch.load(PATH))
# 利用模型对图片进行预测
# outputs = net(images)
# # 共有10个类别 采用模型计算出的概率最大作为预测的实列
# _, predicted = torch.max(outputs, 1)
# # 打印预测标签的结果
# print('Predicted: ', ''.join('%5s' % classes[predicted[j]] for j in range(4)))

# 全部测试集上表现
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device),data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the 10K test images:%d %%' % (100 * correct / total))
# 为了更加细致的看一下模型在那些类别上表现更好，在那些类别上表现更差,我们分类别的进行准确率计算
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device),data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
# 打印不同类别的准确率
for i in range(10):
    print('Accuracy of %5s: %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
