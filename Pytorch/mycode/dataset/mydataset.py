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
