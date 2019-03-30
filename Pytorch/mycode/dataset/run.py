import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataset.mydataset import MyDataset
from torch.utils.data import DataLoader
import torchvision.datasets as dset

# In[1]
def generate_data(num,filename):
    x = np.arange(1, num + 1)  # get axis x
    rand_k = 5 + (5 - 3) * np.random.random()  # get random k in range [1,3]
    rand_noise = np.random.uniform(-1, 1, num) * (num / 10)  # get random noise
    y = rand_k * x + rand_noise
    with open(filename, "w+") as f:
        f.write("x,Y\n")
        for i in range(num):
            f.write("{},{}\n".format(x[i],y[i]))


"""查看数据"""
filename = "data.csv"
# generate_data(200,filename)
df = pd.read_csv(filename)
plt.figure(figsize=(20, 20))
plt.scatter(df['x'], df['Y'])
plt.show()

data = MyDataset(filename)
dataloader = DataLoader(data,shuffle=True,batch_size=5)

for i, batch_data in enumerate(dataloader):
    print(i,batch_data)

# In[2]
mnist_data = dset.mnist