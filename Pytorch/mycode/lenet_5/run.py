# In[0]

import torchvision.datasets as dset
from torchvision import transforms
from argparse import Namespace
from torch.utils.data import DataLoader
args = Namespace(
    data_dir="data/MNIST",
    epoches=1000,
)

# In[1] 数据加载
mnist_train = dset.MNIST(root="dataset",train=True,download=False,transform=transforms.ToTensor())
mnist_test = dset.MNIST(root="dataset",train=False,download=False,transform=transforms.ToTensor())
# In[2] 数据集划分
train_dl = DataLoader(mnist_train,batch_size=5,shuffle=True)
test_dl = DataLoader(mnist_test,batch_size=5,shuffle=True)

le_net =

for i in range(args.epoches):
    avg_loss = 0
    for _, batch_data in enumerate(test_dl):
        batch_image = batch_data[0]
        batch_label = batch_data[1]
