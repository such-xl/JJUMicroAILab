import numpy as np
import torch

x1 = torch.ones(3, 3)
print(x1)

x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
print(y)

z = x * 2
print(z)
print(x.grad_fn)
print(y.grad_fn)
print(z.grad_fn)

z = y * y * 3
out = z.mean()
print(z, out)

# 关于方法requires_grad_()：该方法可以原地改变Tensor的属性，requires_grad的值,如果没有主动设定默认为False
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a*a).sum()
print(b)
print(b.grad_fn)
