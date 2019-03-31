# In[0]
from mycode.lenet_5.model import LeNet_5
import torchvision.datasets as dset
from torchvision import transforms
from argparse import Namespace
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
args = Namespace(
    data_dir="dataset",
    epoches=200,
    batch_size=256
)

# In[1] 数据加载
mnist_train = dset.MNIST(root="./dataset",train=True,download=False,transform=transforms.ToTensor())
mnist_test = dset.MNIST(root="./dataset",train=False,download=False,transform=transforms.ToTensor())
# In[2] 数据集划分
train_dl = DataLoader(mnist_train,batch_size=args.batch_size,shuffle=True)
test_dl = DataLoader(mnist_test,batch_size=args.batch_size,shuffle=True)
# In[3] 训练
le_net = LeNet_5()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(le_net.parameters(),lr = 0.01)

for e in range(args.epoches):
    avg_loss = 0
    print("Epoch {}:".format(e))
    for i, batch_data in enumerate(train_dl):
        batch_image = batch_data[0]                    # [batch_size,1,28,28]
        batch_label = batch_data[1].squeeze_()         # p[5,1] -> (5,)
        batch_num = batch_image.size(0)

        optimizer.zero_grad()
        out = le_net(batch_image).view(batch_num,-1)   # 不用args.batch_size的原因是 数据可能不是 128的整数倍
        loss = criterion(out,batch_label)              # 从而无法进行
        avg_loss += loss.data
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print("loss = {} ".format(loss.data))

# torch.save(le_net,"result/lenet.pkl")

# In[4] 加载模型

le_net = torch.load("result/lenet.pkl")
le_net.eval()

total = test_dl.sampler.num_samples
all_tp = 0
for _,(test_img,test_label) in enumerate(test_dl):
    batch_predict = le_net(test_img).squeeze_()         # 将 [5,1,10] 转换为 [5,10]
    predict_label = torch.max(batch_predict,1)          # pytorch中 判断相同用 equal(); a == b 返回的是每个元素对应值是否相同
    tp_num = (predict_label[1] == test_label).sum()     # 对应位置相同为1 不同为0 例如 [1,0,1,0,0]

    all_tp += tp_num.item()                             # 把0维的数  0-dim转换为 python 数值

test_acc = all_tp/total
print("Test Accurary: {}".format(round(test_acc,4)))




