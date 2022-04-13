"""
构建神经网络的典型流程
·定义一个拥有可学习参数的神经网络
·遍历训练数据集
·处理输入数据使其流经神经网络
·计算损失值
·将网络参数的梯度进行反向传播
·以一定的规则更新网络的权重
"""
# torch.nn构建的神经网络只支持mini-batches的输入,不支持单一样本的输入
# 导入若干工具包
import torch
import torch.nn as nn
import torch.nn.functional as f


# 定义网络类
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义第一层卷积层,输入维度=1,输出维度=6，卷积核大小3*3
        self.conv1 = nn.Conv2d(1, 6, 3)
        # 定义第二层卷积层,输入维度=6，输出维度=16，卷积核大小3*3
        self.conv2 = nn.Conv2d(6, 16, 3)
        # 定义三层全连接神经网路
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 任意卷积层后面要加激活层，池化层
        x = f.max_pool2d(f.relu(self.conv1(x)), (2, 2))
        x = f.max_pool2d(f.relu(self.conv2(x)), 2)
        # 经过卷积层处理后，张量要进入全连接层，进入前需要调整张量的形状
        x = x.view(-1, self.num_flat_features(x))
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)
params = list(net.parameters())
print(len(params))
print(params[0].size())

inp = torch.randn(1, 1, 32, 32)
out = net(inp)
print(out)
print(out.size())

target = torch.randn(10)
target = target.view(1, -1)
criterion = nn.MSELoss()
loss = criterion(out, target)
print(loss)

print(loss.grad_fn)
print(loss.grad_fn.next_functions[0][0])
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])

# 反向传播
# 在python中执行反向传播非常简单,全部的操作就是loss.backword()
# 在执行反向传播之前，要先将梯度清零，否则梯度会在不同的批次数据之间累加

# Pytorch中首先执行梯度清零操作
net.zero_grad()
print('conv1.bias.grad before backward.')
print(net.conv1.bias.grad)

# 在Pytorch中实现一次反向传播
loss.backward()
print('conv1.bias.grad before backward.')
print(net.conv1.bias.grad)

# 更新网络参数
# 跟新参数最简单的算法就是SGD(随机梯度下降)
# 具体的算法公式表达式为：weigth = weight-learning_rate * gradient

# 首先用传统的Python代码来实现SGD
"""
learning_rate = 0.01
for p in net.parameters():
    p.data.sub_(p.grad.data * learning_rate)
"""

# 用Pytorch官方推荐的标准代码如下
# 首先导入优化器的包，optim中包含若若干常用的优化算法，比如SGD,Adam等

import torch.optim as optim

# 通过optim创建优化器对象
optimizer = optim.SGD(net.parameters(), lr=0.01)

#将优化器执行梯度清零操作
optimizer.zero_grad()
output = net(inp)
loss = criterion(output,target)

#对损失值执行反向传播的操作
loss.backward()
#将参数的更新通过一行标准代码来执行
optimizer.step()