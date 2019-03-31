# In[0]
from argparse import Namespace
import pandas as pd
import numpy as np
from mycode.dataset.mydataset import MyDataset
from torch.utils.data.dataloader import DataLoader
from mycode.LinearRegression.linear_regression import LinearRegression
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

args = Namespace(
    filename="../dataset/data.csv",
    batch_size=5,
    epoches=200
)


def max_min_scaler(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


'''数据集划分'''
df = pd.read_csv(args.filename,header=0)
train_df = df.sample(frac=0.8,replace=False)  # 80%用于训练 20%用于测试
test_df = df[~df.index.isin(train_df.index)]  # 总集合中删去训练的数据

'''数据进行归一化 否则无法用于训练 梯度太大'''
test_df[['x']] = test_df[['x']].apply(max_min_scaler)
test_df[['Y']] = test_df[['Y']].apply(max_min_scaler)
train_df[['x']] = train_df[['x']].apply(max_min_scaler)
train_df[['Y']] = train_df[['Y']].apply(max_min_scaler)

train_X = train_df.loc[:, 'x'].values
train_Y = train_df.loc[:, 'Y'].values
test_X = test_df.loc[:, 'x'].values
test_Y = test_df.loc[:, 'Y'].values

'''加载数据集和dataloader'''
data_set = MyDataset(train_X,train_Y)
dl = DataLoader(dataset=data_set,batch_size=args.batch_size,shuffle=True)

'''print data
for _, d in enumerate(dl):
    print(d)
'''

'''定义训练损失函数和优化器'''
linear_model = LinearRegression()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear_model.parameters(), lr = 0.01)


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
# In[1]
linear_model = torch.load("./result/linear_model.pkl")
linear_model.eval()
predict_y = linear_model(torch.FloatTensor(train_X).reshape(-1,1))

'''绘制图像'''
plt.figure(figsize=(5,5))
plt.title("Linear Regression")
plt.scatter(test_X.reshape(-1,1),test_Y.reshape(-1,1))
plt.plot(train_X.reshape(-1,1),predict_y.detach().numpy(),color='red',label='regression line')
plt.legend(['regression line', 'test data'])
plt.xlabel("X")
plt.ylabel("Y")

plt.savefig("./result/predict.png")   # 必须先savefig再show
plt.show()                          # 因为show的时候会创建一个新的空白图片






