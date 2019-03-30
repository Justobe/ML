# In[1] 自定义加载数据
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mycode.dataset.mydataset import MyDataset
from torch.utils.data import DataLoader
import torchvision.datasets as dset


def generate_data(num,filename):
    x = np.arange(0, num)  # get axis x
    rand_k = 5 + (5 - 3) * np.random.random()  # get random k in range [1,3]
    rand_noise = np.random.uniform(-1, 1, num) * (num / 10)  # get random noise
    y = rand_k * x + rand_noise
    with open(filename, "w+") as f:
        f.write("x,Y\n")
        for i in range(num):
            f.write("{},{}\n".format(x[i],y[i]))


"""查看数据"""
filename = "data.csv"
generate_data(200,filename)
df = pd.read_csv(filename)
plt.figure(figsize=(5, 5))
plt.scatter(df['x'], df['Y'],s=2 )
plt.show()

data = MyDataset(df['x'].values,df['Y'].values)
dataloader = DataLoader(data,shuffle=True,batch_size=5)

for i, batch_data in enumerate(dataloader):
    print(i,batch_data)

# In[2] 加载mnist数据
import torchvision.datasets as datasets
# trainset = datasets.MNIST(root='./data', # 表示 MNIST 数据的加载的目录
#                                       train=True,  # 表示是否加载数据库的训练集，false的时候加载测试集
#                                       download=True, # 表示是否自动下载 MNIST 数据集
#                                       transform=None) # 表示是否需要对数据进行预处理，none为不进行预处理
#
