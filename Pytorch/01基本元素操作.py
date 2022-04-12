import torch

# 创建一个没有初始化的矩阵
x = torch.empty(5, 3)
print(x)

# 创建一个有初始化的矩阵
x = torch.rand(5, 3)  # 标准高斯分布, 均值为0,方差为1
print(x)

# 创建一个全零矩阵并指定数据元素的类型为long
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# 之间通过数据创建张量
x = torch.tensor([2.5, 3.5])
print(x)

# 通过已有的一个张量创建相同尺寸的张量
# 利用news_methods方法得到一个张量
x = x.new_ones(5, 3, dtype=torch.double)
# 利用randn_like方法得到相同张量尺寸的一个新张量,并且采用随机初始化来对其赋值
y = torch.randn_like(x, dtype=torch.float)
print(x)
print(y)

# 得到张量尺寸 torch.size()函数本质返回的是一个tuple,因此它支持一切元组的操作
print(x.size())
a, b = x.size()
print('a=', a)
print('b=', b)

# 用类似numpy的方式对张量进行操作
print(x[:, 1])
print(x[:, :2])

# 改变张量的形状  tensor.view()操作需要确保数据元素的总数量不变
x = torch.randn(4, 4)
y = x.view(16)
# -1代表自动匹配个数
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())

# 如果张量中只有一个元素,可以用.item()将值取出，作为一个python number
x = torch.randn(1)
print(x)
print(x.item())
