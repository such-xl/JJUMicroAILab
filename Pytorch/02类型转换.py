import torch
import numpy as np
# torch Tensor 和Numpy array 共享底层内存空间，因此改变其中一个值,另一个也会随之被改变

# 将Torch Tensor 转换为 Numpy array
a = torch.ones(5)
print(a)
print('--------')
b = a.numpy()
print(b)
print('--------')
a.add_(1)
print(a)
print('--------')
print(b)
print('--------')

# 将Numpy array 转换为Torch Tensor
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

# 所有在CPU上的Tensors,除了CharTensor 都可以转换为Numpy array并可以反向转换
# 关于Cuda Tensor:Tensors可以用.to()方法来将其移动到任意设备上。
if torch.cuda.is_available():
    print("Hello CUDA")
    torch.cuda.set_device(0)
    x = torch.randn(1)
    # 定义一个设备对象，这里指定成CUDA，即使用GPU
    device = torch.device("cuda")
    # 直接在GPU上创建一个Tensor
    y = torch.ones_like(x, device=device)
    # 将在CPU上面的x张量移动到GPU上面
    x = x.to(device)
    z = x + y
    # z张量在GPU上
    print('z=', z)
    # 也可以将z转移到CPU上，并同时指定张量元素的数据类型
    print(z.to("cpu", torch.double))