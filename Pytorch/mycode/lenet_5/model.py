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

    def forward(self, x):
        batch_size = x.size(0)
        conv1_x = self.conv1(x)
        pooling1_x = self.pooling1(conv1_x)
        conv2_x = self.conv2(pooling1_x)
        pooling2_x = self.pooling2(conv2_x)
        fc1_x = self.fc1(pooling2_x.reshape(batch_size,-1))
        fc2_x = self.fc2(fc1_x)
        output = self.fc3(fc2_x)
        return output

