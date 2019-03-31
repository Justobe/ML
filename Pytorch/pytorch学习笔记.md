### 2019.03.21 - 张量

这一章节主要是学习在pytorch中我们如何创建tensor。Tensor是torch中用于存储向量的重要的数据结构，并且方便进行GPU计算（numpy不支持GPU）。在pytorch-1.0之前的版本中，还有一个数据结构Variable。Variable和Tensor没有本质区别，都是存储张量，但是Variable可以用于计算梯度。在新版的pytorch-1.0中，将Variable被废弃了，我们统一使用Tensor保存张量，使用requires_grad=True来标记是否进行自动求导（在第二章中会详细介绍）

torch中创建张量很简单，和numpy中创建ndarray方式很像。

```python
import torch

torch.tensor([1,2,3]) #利用现有数组创建张量
torch.rand(5,3) # 创建 5x3维的随机矩阵
torch.empty(5,3) # 定义一个 没有被初始化的tensor，里面的数字是垃圾数
torch.randn_like(x, dtype=torch.float) #生成x大小尺寸的张量
torch.zeros(5,3) # 定义一个0矩阵 
torch.add(x, y, out=result) # 加法，并且把结果返回到某个变量里
torch.normal(mean, std, out=None) # 按照正态分布生成 tensor
```

**tensor基本操作：**

```python
torch.cat(seq, dim=0, out=None)  #按照指定维度拼接tensor
torch.chunk(tensor, chunks, dim=0)  #按照指定维度 对tensor进行切块
torch.squeeze(input)  #将tensor展开成1维
x.view(-1,1) #相当于numpy中的 reshape
torch.reshape(input, shape) #将input进行reshape
y.add_(x) #把x加到y上 torch中所有带“_”的函数都是inplace操作的
torch.add(input, value, out=None) #
b = a.numpy() # 获取某个tensor的numpy
b = torch.from_numpy(a) # 从numpy生成tensor
torch.mul(input, other, out=None) #乘法 矩阵对应位置相乘
torch.mm(input, other) #矩阵点乘
torch.round(input, out=None) #矩阵四舍五入
torch.ceil(input, out=None)  #向上取整，等于向下取整+1
torch.argmax(input, dim=None, keepdim=False) # 返回指定维度方向上的 最大值对应的index
torch.sigmoid(input, out=None) #sigmod函数
torch.tanh(input, out=None) #tanh函数
torch.abs(input, out=None) #取绝对值
```

---

### 2019.03.22 - 自动求导

深度学习的算法本质上是通过反向传播求导数，而PyTorch的autograd模块则实现了此功能。在Tensor上的所有操作，autograd都能为它们自动提供微分，避免了手动计算导数的复杂过程。

从0.4起, Variable 正式合并入Tensor, Variable 本来实现的自动微分功能，Tensor就能支持。我们的代码的运行环境均为pytorch-1.0

```python
import torch
#打印版本
torch.__version__

# 1.0.1
```

我们首先来看如何在pytorch中求简单的梯度

我们有如下的示例：

```python
import torch
import numpy as np

x = torch.tensor(2.0,requires_grad=True)
w = torch.tensor(5.0,requires_grad=True)
b = torch.tensor(4.0,requires_grad=True)
h = w*x+b

print(x.grad_fn) #grad_fn表示得到这个变量对应的操作
print(w.grad_fn) #是直接定义的来 还是通过加减得来
print(b.grad_fn)
print(h.grad_fn)

h.backward()   # pytorch中自动求导的函数
print(x.grad)  #对应某个变量的梯度
print(w.grad)
print(b.grad)

'''
None
None
None
<AddBackward0 object at 0x00000167ABA14B00>
tensor(5.)
tensor(2.)
tensor(1.)
'''
```



上面的代码对应的公式如下：
$$
h = wx + b
$$
首先我们定义三个标量 x，w，b，其值分别为2.0，5.0， 4.0。在我们声明变量的时候，使用`requires_grad=True`来标记，我们希望计算该变量的梯度，于是pytorch会将该变量放入计算图中。

然后我们使用backward()，计算所有标记`requires_grad=True`的变量的梯度。针对backward()函数的参数，我们后续会做详细的解释。

因此，h分别对x，w， b求梯度，其值分别为5, 2, 1

h.backward()函数是利用反向传播，对变量求解梯度。如果h是标量，那么就不需要使用参数。如果h是一个矢量，那么久需要传入一个矩阵/向量，传入向量的shape和y的shape是相同的。

```python
# In[4]
x_1 = torch.rand(3,requires_grad=True)

y_1 = 2*x_1

print(y_1.grad_fn)
y_1.backward(torch.ones(3)) # 传入的是[1,1,1]
print(x_1.grad)

'''
<MulBackward0 object at 0x00000167ABA14320>
tensor([2., 2., 2.])
'''
```

如果我们更改backward()传入参数，x_1的梯度会变成：

```python
x_1 = torch.rand(3,requires_grad=True)

y_1 = 2*x_1

print(y_1.grad_fn)
y_1.backward(0.1*torch.ones(3)) #[0.1,0.1,0.1]
print(x_1.grad)

'''
<MulBackward0 object at 0x00000167ABA14DD8>
tensor([0.2000, 0.2000, 0.2000])
'''
```

可以看出，在我们更改backward()传入的参数之后，x_1 的梯度发生了改变，变为之前的0.1倍。所以backward()中传入的参数，是对应分量梯度乘上的一个倍数。如[1,0.1,0.01]就是三个分量分别乘上1,0.1,0.01

最后给出一个完整的例子：

```python
x1 = torch.from_numpy(2 * np.ones((2, 2), dtype=np.float32))
x1.requires_grad_(True)
w1 = torch.from_numpy(5 * np.ones((2, 2), dtype=np.float32))
w1.requires_grad_(True)
print("x1 =", x1)
print("w1 =", w1)

x2 = x1.mm(w1)
w2 = torch.from_numpy(6 * np.ones((2, 2), dtype=np.float32))
w2.requires_grad_(True)
print("x2 =", x2)
print("w2 =", w2)
'''
x1 = tensor([[2., 2.],
        [2., 2.]], requires_grad=True)
w1 = tensor([[5., 5.],
        [5., 5.]], requires_grad=True)
'''

y = x2.mm(w2) # 矩阵点乘 相当于numpy中的dot
Y = torch.from_numpy(10 * np.ones((2, 2), dtype=np.float32))
print("y =", y)
print("Y =", Y)

'''
y = tensor([[240., 240.],
        [240., 240.]], grad_fn=<MmBackward>)
Y = tensor([[10., 10.],
        [10., 10.]])
'''

L = Y - y
print("L =", L)

'''
L = tensor([[-230., -230.],
        [-230., -230.]], grad_fn=<SubBackward0>)
'''
L.backward(torch.ones(2, 2, dtype=torch.float))

print("x1.grad =", x1.grad)
print("w1.grad =", w1.grad)
print("w2.grad =", w2.grad)
'''
x1.grad = tensor([[-120., -120.],
        [-120., -120.]])
w1.grad = tensor([[-48., -48.],
        [-48., -48.]])
w2.grad = tensor([[-40., -40.],
        [-40., -40.]])
'''
```
上述代码表示的公式为：
$$
x_2 = \omega_1x_1  \\
y=\omega_2x_2 \\
L = Y-y
$$

求导过程如下，由于矩阵中每个元素均相同，我就展示一个数字：
$$
\frac{\partial L}{\partial y} =  -1 \\
\frac{\partial L}{\partial \omega_2} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial \omega_2} \\
\frac{\partial L}{\partial \omega_1} =  \frac{\partial L}{\partial y} \frac{\partial y}{\partial x_2} \frac{\partial x_2}{\partial \omega_1}\\

\frac{\partial L}{\partial x_1} =  \frac{\partial L}{\partial y} \frac{\partial y}{\partial x_2} \frac{\partial x_2}{\partial x_1}
$$
根据上边的数值，我们将${\frac{\partial L}{\partial y}}$ 依次带入下边的公式中：
$$
\frac{\partial L}{\partial y} =  -1 \\
\frac{\partial L}{\partial \omega_2} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial \omega_2} = -1*(240/6) = -40 \\
\frac{\partial L}{\partial \omega_1} =  \frac{\partial L}{\partial y} \frac{\partial y}{\partial x_2} \frac{\partial x_2}{\partial \omega_1} = -1*(240/20)*(20/5) = -48\\

\frac{\partial L}{\partial x_1} =  \frac{\partial L}{\partial y} \frac{\partial y}{\partial x_2} \frac{\partial x_2}{\partial x_1} = -1*(240/20)*(20/2) = -120
$$

----

### 2019.03.26 - torch.nn包

torch.nn是专门为神经网络设计的模块化接口。nn构建于 Autograd之上，可用来定义和运行神经网络。

```python
import torch
import torch.nn as nn

```

**nn.Model**: torch中的container，也可视为我们常说的模型。是所有网络的基类，我们自己创建的模型也应该继承自该类。只要继承nn.Module，并实现它的forward方法，PyTorch会根据autograd，自动实现backward函数

```
class Net(nn.Module):
    def __init__(self):
    	pass
    def forward(self, x): 
 		pass
```

我们在\__init__()函数中，定义模型的各个层，然后再forward()中，定义输入前向计算的过程。

在torch.nn中定义了许多神经网络相关的模块，我们讲解一些常用的模块

**卷积与转置卷积：**

- **nn.Conv1d(in_channels, out_channels, kernel_size, stride=1**, padding=0, dilation=1, groups=1, bias=True)
  - 解释：一维卷积层
  - 示例：`nn.Conv1d(16, 33, 3, stride=2)` ，
  - 注意：在pytorch中，默认是使用的函数内置的卷积核。如果要使用自定义的卷积核要自行更改。**另外，在torch.nn包中，像kernel_size等参数可以传入int也可以传入tuple，如果传入int表示使用正方形卷积核，如果传入tuple则按照tuple的大小指定卷积核**
- **nn.Conv2d(in_channels, out_channels, kernel_size, stride=1,** padding=0, dilation=1, groups=1, bias=True)
  - 解释：二维卷积层，和一维的参数相同，不过kernel_size需要传入tuple。同理还有三位卷积层，不再赘述。
  - 示例：`nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))`
- **nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1,** padding=0, output_padding=0, groups=1, bias=True)
  - 解释：二维的转置卷积，关于转置卷积可以参考[这篇文章](ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True))。转置卷积可以看做是卷积的“逆过程”，其作用就是为了对图像进行上采样，扩大图像尺寸。但是对一个矩阵进行卷积再进行转置卷积并不能得到原有的数值。转置卷积只能使矩阵恢复到原有尺寸。
  - 示例：`nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))`

**池化与反池化（均以二维为例）：**

- **nn.MaxPool2d(kernel_size, stride=None,** padding=0, dilation=1, return_indices=False, ceil_mode=False)

  - 解释：二维池化层，使用max pooling
  - 示例：`nn.MaxPool2d(3, stride=2)`

- **nn.MaxUnpool2d(kernel_size, stride=None,** padding=0)

  - 解释：二维反池化，原理图如下：

    ![1553604847254](img\1553604847254.png)

  - 示例：`nn.MaxUnpool2d(2, stride=2)`

还有平均池化等，由于调用方式相同，原理较为简单，在这里不再赘述。除此之外，还有一些其他的池化操作，分数最大化，幂平均池化等，详见pytorch文档。

**激活函数：**

- **nn.RELU()**
- **nn.tanh()**
- **nn.Sigmoid()**

- **nn.Threshold(threshold, value,inplace=False)**
  - 解释：输入值小于阈值就会被value代替。
- **nn.Softmax()**
  - 解释：对n维输入张量运用Softmax函数，将张量的每个元素缩放到（0,1）区间且和为1。
- **nn.LogSoftmax()**
  - 解释：同softmax()函数

**nn中的其他重要结构：**

**全连接层：**

- nn.Linear(in_features, out_features, bias=True)
  - in_features - 每个输入样本的大小
  - out_features - 每个输出样本的大小
  - 解释：$y = w^Tx$  

**Dropout：**

- nn.Dropout(p=0.5, inplace=False)
  - 解释：随机将输入张量中部分元素设置为0。对于每次前向调用，被置0的元素都是随机的。
- nn.Dropout2d(p=0.5, inplace=False)
  - 解释：随机将输入张量中整个通道设置为0。对于每次前向调用，被置0的通道都是随机的。通常是一行或者一列被置为0

**损失函数：**

损失函数有很多种类型，例如**CrossEntropyLoss**或者**MSELoss**，我们只介绍一下中函数的使用方法

- **nn.CrossEntropyLoss(weight=None, size_average=True)**

  - 解释：交叉熵损失函数

  - 示例：

    ```python
    criterion = nn.CrossEntropyLoss()
    loss = criterion(x, y) #调用标准时也有参数
    ```

另外，在torch.nn包中，还有一个torch.nn.functional包，其和torch.nn中很多定义是相同的，例如nn.Conv2d和nn.functional.conv2d，nn.MaxPool2d和nn.functional.max_pool2d。

**区别就在于，torch.nn中定义的是模块，例如nn.Conv2d();而F.conv2d()是一个函数。以卷积为例，nn.Conv2d内部有一个\_init_()函数和forward()函数。在init()函数中定义了一些内置的Variable参数，然后再forward中调用了nn.functional.conv2d。因此，我们使用nn.functional用于简单的计算，用torch.nn中的函数进行来定义模型。**

**优化器：**

pytorch中的优化器位于torch.optim包中，但是在这里一并介绍，在后面直接开始搭建神经网络。

在反向传播计算完所有参数的梯度后，还需要使用优化方法来更新网络的权重和参数，在torch.optim中实现大多数的优化方法，例如RMSProp、Adam、SGD等

```python
out = net(input) # 这里调用的时候会打印出我们在forword函数中打印的x的大小
criterion = nn.MSELoss()
loss = criterion(out, y)
#新建一个优化器，SGD只需要要调整的参数和学习率
optimizer = torch.optim.SGD(net.parameters(), lr = 0.01)
# 先梯度清零(与net.zero_grad()效果一样)
optimizer.zero_grad() 
loss.backward()

#更新参数
optimizer.step()
```

---

### 2019.03.27 - 数据集加载

**制作自定义数据集：**

**Dataset:**

pytorch中提供了抽象类Dataset，用于方便地读取数据，将要使用的数据包装为Dataset。我们自定义的数据也都需要继承并且实现两个成员方法：

- \__len__()：返回数据集的总长度
- \__getitem__():返回数据集中的一个样本

```python
#定义一个数据集
class myDataset(Dataset):
    """ 数据集演示 """
    def __init__(self):
        """实现初始化方法，在初始化的时候将数据读载入"""
        pass
    def __len__(self):
        '''
        返回df的长度
        '''
        pass
    def __getitem__(self, idx):
        '''
        根据 idx 返回一列数据
        '''
        pass
```

**Dataloader:**

DataLoader为我们提供了对Dataset的读取操作，常用参数有：batch_size(每个batch的大小), shuffle(是否进行shuffle操作)

```python
dataset = myDataset()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

for idx, data in enumerate(dataloader):
    print(i,data)
```

我们以之前用于线性回归的数据为例，创建一个数据集，代码在dataset文件夹中：

首先我们先创建数据，数据形似y = kx+b

```python
def generate_data(num,filename):
    x = np.arange(1, num + 1)  # get axis x
    rand_k = 5 + (5 - 3) * np.random.random()  # get random k in range [1,3]
    rand_noise = np.random.uniform(-1, 1, num) * (num / 10)  # get random noise
    y = rand_k * x + rand_noise
    with open(filename, "w+") as f:
        f.write("x,Y\n")
        for i in range(num):
            f.write("{},{}\n".format(x[i],y[i]))

filename = "data.csv"
# generate_data(200,filename)
df = pd.read_csv(filename)
plt.figure(figsize=(5, 5))
plt.scatter(df['x'], df['Y'],s=2 )
plt.show()
```

数据如下所示：

![1553844277068](img\1553844277068.png)

创建我们的 dataset类，该类可以从csv文件读取数据，加载到类中。每次使用getitem()函数返回一组x，y数据

```python
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """ 数据集演示 """
    def __init__(self, X,Y):
        self.x = X
        self.Y = Y

    def __len__(self):
        """返回dataset的长度"""
        return len(self.x)

    def __getitem__(self, idx):
        """根据 idx 返回一列数据"""
        return self.x[idx], self.Y[idx]

```

加载数据集：

```python
data = MyDataset(df['x'].values,df['Y'].values)
dataloader = DataLoader(data,shuffle=True,batch_size=5)
```

遍历dataloader：

```python
for i, batch_data in enumerate(dataloader):
    print(i,batch_data)
```

![1553846466530](img\1553846466530.png)

i返回当前batch的序号，batch_data是一个list，由两个元组组成：

- 第一个元组有5个元素，分别代表五个样本的x值
- 第二个元组有5个元素，分别代表5个样本的Y值

以上是如何读取文本的数据，实际上图片数据是一样的。主要还是实现getitem()和 len()函数。**但是稍有不同的是，针对图片数据集，我们在getitem()函数中，使用PIL(Python image Library)读取完图片以后，要使用torchvision的transform库中的函数，将图像转换成tensor或者进行其他的一些操作**。在后边介绍torchvision和创建卷积神经网络的时候还会在介绍。

**使用pytorch自带数据集：**

pytorch自带的视觉数据集有很多，例如MNIST，CIFAR-10等，我们可以很方便的直接使用torchvision中提供的数据集来训练我们的模型。

```python
import torchvision.datasets as datasets
trainset = datasets.MNIST(root='./data', # 表示 MNIST 数据的加载的目录
                                      train=True,  # 表示是否加载数据库的训练集，false的时候加载测试集
                                      download=True, # 表示是否自动下载 MNIST 数据集
                                      transform=None) # 表示是否需要对数据进行预处理，none为不进行预处理
```

### 2019.03.29 - pytorch一元线性回归

我们使用pytorch来创建一个最简单的一元线性回归，以此来学习pytorch的模型构建，训练，评估，可视化等各个过程。

首先，我们使用的数据还是之前在数据集加载时使用的线性数据，我们将之前的数据存储到csv中，然后读取加载。

一个简单的一元线性回归模型 y = kx 定义如下：

linear_regression.py:

```python
import torch.nn as nn

class LinearRegression(nn.Module):

    def __init__(self):
        super(LinearRegression, self).__init__()
        self.lr = nn.Linear(1, 1)  # y = wx

    def forward(self, x):
        out = self.lr(x)
        return out

```

在LinearRegression/run.py文件中，我们编写加载数据和训练以及模型保存的代码：

数据加载及训练集划分：

```python
'''数据集划分'''
df = pd.read_csv(args.filename,header=0)
train_df = df.sample(frac=0.8,replace=False)  # 80%用于训练 20%用于测试
test_df = df[~df.index.isin(train_df.index)]  # 总集合中删去训练的数据
```

数据归一化。如果不进行归一化，则梯度会很大，loss无法收敛

```python
'''数据进行归一化 否则无法用于训练 梯度太大'''
test_df[['x']] = test_df[['x']].apply(max_min_scaler)
test_df[['Y']] = test_df[['Y']].apply(max_min_scaler)
train_df[['x']] = train_df[['x']].apply(max_min_scaler)
train_df[['Y']] = train_df[['Y']].apply(max_min_scaler)

train_X = train_df.loc[:, 'x'].values
train_Y = train_df.loc[:, 'Y'].values
test_X = test_df.loc[:, 'x'].values
test_Y = test_df.loc[:, 'Y'].values
```

加载数据集，定义损失函数和优化器：

```python
'''加载数据集和dataloader'''
data_set = MyDataset(train_X,train_Y)
dl = DataLoader(dataset=data_set,batch_size=args.batch_size,shuffle=True)


'''定义训练损失函数和优化器'''
linear_model = LinearRegression()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear_model.parameters(), lr = 0.01)
```

训练并保存模型：

 注意，在训练的时候，由于pytorch的广播机制，模型是可以介绍batch数据训练的，

```python
for i in range(args.epoches):
    avg_loss = 0
    for batch, batch_data in enumerate(dl):
        batch_x = torch.FloatTensor(list(batch_data[0])).reshape(-1,1)  #必须使用tensor
        batch_y = torch.FloatTensor(list(batch_data[1])).reshape(-1,1)  #转换为 5x1 最前面的数字是batch_size
        optimizer.zero_grad()  # 梯度清零
        out = linear_model(batch_x)  #
        loss = criterion(batch_y, out)

        avg_loss += loss.data  # 将该epoch中的loss 求和
        loss.backward()  # 反向传播
        optimizer.step()  # 参数更新
    if i % 10 == 0:
        print("Epoches: {} ,loss: {}".format(i, avg_loss))
torch.save(linear_model,"./result/linear_model.pkl")
```

模型评估及结果可视化：

```python
linear_model = torch.load("./result/linear_model.pkl")
linear_model.eval()
predict_y = linear_model(torch.FloatTensor(train_X).reshape(-1,1))

'''绘制图像'''
plt.figure(figsize=(5,5))
plt.title("Linear Regression")
plt.scatter(test_X.reshape(-1,1),test_Y.reshape(-1,1))
plt.plot(train_X.reshape(-1,1),predict_y.detach().numpy(),color='red',label='regression line')
plt.legend(['regression line', 'origin data'])
plt.xlabel("X")
plt.ylabel("Y")

plt.savefig("./result/predict.png")   # 必须先savefig再show
plt.show()                          # 因为show的时候会创建一个新的空白图片

```

![](E:\github\ML\Pytorch\img\predict.png)

### 2019.03.30 - pytorch LeNet-5 模型

今天对照着LeNet5的结构图，使用pytorch手动实现了一个模型，可以用于识别手写数据集MNIST，在测试集上的精度为0.9866。实现过程如下：

LeCun在论文中提出的LeNet的结构如下：

![img](img\lenet.png)

现在常用的结构为：

![](E:\github\ML\Pytorch\img\1351564-20180827204056354-1429986291.png)



**卷积层-池化层-卷积层-池化层-全连接-全连接-全连接-softmax**

首先我们来定义LeNet5的模型：

```python
import torch.nn as nn


class LeNet_5(nn.Module):
    def __init__(self):
        super(LeNet_5,self).__init__()
        self.conv1 = nn.Conv2d(1,6,5,padding=2)  # MNIST手写数据集是28x28的 但是LeNet5的输入是32x32的 需要padding
        self.pooling1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.pooling2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):                      # [batch_size,1,28,28]
        batch_size = x.size(0)   
        conv1_x = self.conv1(x)
        pooling1_x = self.pooling1(conv1_x)
        conv2_x = self.conv2(pooling1_x)
        pooling2_x = self.pooling2(conv2_x)
        fc1_x = self.fc1(pooling2_x.reshape(batch_size,-1))
        fc2_x = self.fc2(fc1_x)
        output = self.fc3(fc2_x)
        return output

```



在模型中我没有使用softmax,因为我在运行的代码中使用了CrossEntropyLoss（交叉熵损失函数），这个模块内置了softmax。

模型定义很简单，接下来我们在run.py中运行我们的函数,首先加载数据集：

我们使用pytorch torchvision中自带的数据库进行训练。同时，内置的mnist还将训练集和测试集进行了划分

```python
mnist_train = dset.MNIST(root="./dataset",train=True,download=False,transform=transforms.ToTensor())
mnist_test = dset.MNIST(root="./dataset",train=False,download=False,transform=transforms.ToTensor())

train_dl = DataLoader(mnist_train,batch_size=args.batch_size,shuffle=True)
test_dl = DataLoader(mnist_test,batch_size=args.batch_size,shuffle=True)
```

定义模型进行训练：

```python
# In[3] 训练
le_net = LeNet_5()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(le_net.parameters(),lr = 0.01)

for e in range(args.epoches):
    avg_loss = 0
    print("Epoch {}:".format(e))
    for i, batch_data in enumerate(train_dl):
        batch_image = batch_data[0]                    # [batch_size,1,28,28]
        batch_label = batch_data[1].squeeze_()         # [1,128] -> (128,)
        batch_num = batch_image.size(0)

        optimizer.zero_grad()
        out = le_net(batch_image).view(batch_num,-1)   # 不用args.batch_size的原因是 数据可能不是 128的整数倍
        loss = criterion(out,batch_label)              # 从而无法进行
        avg_loss += loss.data
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print("loss = {} ".format(loss.data))

torch.save(le_net,"result/lenet.pkl")
```

这段代码中有几个需要解释的地方：

-  训练的batch_image的尺寸是[128,1,28,28]，其中128是batch_size，即一下训练多少张图片，1是通道数，我们的灰度图是单通道的。28x28是MNIST图片的尺寸

- batch_data[1] 是批训练样本的标签，尺寸为[1,128]

- le_net(batch_image)返回的结果是每一个样本被预测为某一类的概率，尺寸为[128,1,10]

- 另外，由于我们使用的损失函数是CrossEntropyLoss()，其要求的输入是

  - Input：(N,C) where C = number of classes，N = batch_size
  - Target: (N) where each value is 0≤targets[i]≤C−1

  因此，我们必须将[1,128]尺寸的矩阵，reshape成长度为(128,)的向量，同时将[128,1,10]reshape成[128,10]，否则就会报错

  > RuntimeError: multi-target not supported at

  [这部分解释来源于stackoverflow有记录](<https://stackoverflow.com/questions/49206550/pytorch-error-multi-target-not-supported-in-crossentropyloss>)

评估模型：

```python
le_net = torch.load("result/lenet.pkl")
le_net.eval()

total = test_dl.sampler.num_samples                     #获取测试集dataloader的样本数
all_tp = 0
for _,(test_img,test_label) in enumerate(test_dl):
    batch_predict = le_net(test_img).squeeze_()         # 将 [5,1,10] 转换为 [5,10]
    predict_label = torch.max(batch_predict,1)          # pytorch中 判断相同用 equal(); a == b 返回的是每个元素对应值是否相同
    tp_num = (predict_label[1] == test_label).sum()     # 对应位置相同为1 不同为0 例如 [1,0,1,0,0]

    all_tp += tp_num.item()                             # 把0维的数  0-dim转换为 python 数值

test_acc = all_tp/total
print("Test Accurary: {}".format(round(test_acc,4)))



```

torch.max(x,1)表示记录矩阵x在行方向上，每一列的最大值，和对应列序号。torch.max(x,1)则记录列方向上，每一行的最大值，和对应行号。返回值是一个元组，第一个元素代表最大值组成的列表，第二个元素代表最大值对应列序号对应的列表。这里我们用来计算每个样本被预测的标签。

示例：

```shell
a = torch.rand(5,3)
print(a)
tensor([[0.7455, 0.1243, 0.2365],
        [0.6545, 0.7545, 0.3551],
        [0.0446, 0.4906, 0.3652],
        [0.5533, 0.7339, 0.0600],
        [0.4516, 0.8994, 0.1623]])
torch.max(a,1)
Out[6]: (tensor([0.7455, 0.7545, 0.4906, 0.7339, 0.8994]), tensor([0, 1, 1, 1, 1]))

```

完整代码在mycode/lenet_5下。由于pytorch还没有内置的可视化工具（虽然可以使用tensorboard或者自己绘制loss曲线绘制），所以这里就不放训练过程的可视化了。



