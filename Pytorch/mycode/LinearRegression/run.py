from code.dataset.mydataset import MyDataset
from torch.utils.data.dataloader import DataLoader

data_set = MyDataset("../dataset/data.csv")

dl = DataLoader(dataset=data_set,batch_size=5,shuffle=True)

for _, d in enumerate(dl):
    print(d)
