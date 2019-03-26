# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 14:41:50 2019

@author: yanming
"""
# In[0]
import torch
import numpy as np
print(torch.__version__)
x = torch.tensor(2.0,requires_grad=True)
w = torch.tensor(5.0,requires_grad=True)
b = torch.tensor(4.0,requires_grad=True)
h = w*x+b
print(x.grad_fn)
print(w.grad_fn)
print(b.grad_fn)
print(h.grad_fn)
h.backward()
print(x.grad)
print(w.grad)
print(b.grad)
# In[0]

x = torch.rand(5, 5, requires_grad=True)
y = torch.rand(5,5,requires_grad=True)

z = torch.sum(x+y)  # 返回所有元素之和
print(z.grad_fn)
z.backward()
print(x.grad,y.grad)

z.backward()
print(x.grad,y.grad)

# In[1]
x1 = torch.from_numpy(2 * np.ones((2, 2), dtype=np.float32))
x1.requires_grad_(True)
w1 = torch.from_numpy(5 * np.ones((2, 2), dtype=np.float32))
w1.requires_grad_(True)
print("x1 =", x1)
print("w1 =", w1)

x2 = torch.mm(x1,w1)
w2 = torch.from_numpy(6 * np.ones((2, 2), dtype=np.float32))
w2.requires_grad_(True)
print("x2 =", x2)
print("w2 =", w2)

y = torch.mm(x2,w2)
Y = torch.from_numpy(10 * np.ones((2, 2), dtype=np.float32))
print("y =", y)
print("Y =", Y)

L = Y - y
print("L =", L)

L.backward(torch.ones(2, 2, dtype=torch.float))

print("x1.grad =", x1.grad)
print("w1.grad =", w1.grad)
print("w2.grad =", w2.grad)
# In[3]
x3 = torch.rand(5,3,requires_grad=True)
w3 = torch.rand(3,5,requires_grad=True)

y = w3.mm(x3)
print(y.grad_fn)
y.backward(torch.ones(3,3))
print(x3.grad)
# In[4]
x_1 = torch.rand(3,requires_grad=True)

y_1 = 2*x_1

print(y_1.grad_fn)
y_1.backward(0.1*torch.ones(3))
print(x_1.grad)
