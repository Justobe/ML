from torch.utils.data import Dataset


class MyDataset(Dataset):
    """ 数据集演示 """
    def __init__(self, filename,header=True):
        self.x = list()
        self.Y = list()
        with open(filename, 'r+') as f:
            if header:
                f.readline()                 #第一行是header
            for line in f.readlines():
                line = line.split(",")
                self.x.append(line[0])
                self.Y.append(line[1].strip("\n"))

    def __len__(self):
        """返回dataset的长度"""
        return len(self.x)

    def __getitem__(self, idx):
        """根据 idx 返回一列数据"""
        return self.x[idx], self.Y[idx]
