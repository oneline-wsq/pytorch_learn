import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset=torchvision.datasets.CIFAR10("./CIFAR10",train=False,transform=torchvision.transforms.ToTensor())
dataloader=DataLoader(dataset,batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.linear1=Linear(196608,10)
    def forward(self,input):
        output=self.linear1(input)
        return output

tudui=Tudui()

for data in dataloader:
    imgs,targets=data
    print(imgs.shape)
    #output=torch.reshape(imgs,(1,1,1,-1))
    # 或者直接使用flatten，将其摊平成为一维数组
    output=torch.flatten(imgs)
    print(output.shape)
    output2=tudui(output)
    print(output2.shape)