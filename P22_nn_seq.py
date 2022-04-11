import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

import writer
from torch import nn


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        # self.conv1=nn.Conv2d(3,32,kernel_size=5,padding=2)
        # self.maxpool1=nn.MaxPool2d(2)
        # self.conv2=nn.Conv2d(32,32,kernel_size=5,padding=2)
        # self.maxpool2=nn.MaxPool2d(2)
        # self.conv3=nn.Conv2d(32,64,kernel_size=5,padding=2)
        # self.maxpool3=nn.MaxPool2d(2)
        # self.flatten=nn.Flatten()
        # self.linear1=nn.Linear(1024,64)
        # self.linear2=nn.Linear(64,10)
        self.model1=nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self,x):
        # x=self.conv1(x)
        # x=self.maxpool1(x)
        # x=self.conv2(x)
        # x=self.maxpool2(x)
        # x=self.conv3(x)
        # x=self.maxpool3(x)
        # x=self.flatten(x)
        # x=self.linear1(x)
        # x=self.linear2(x)
        x=self.model1(x)
        return x

tudui=Tudui()
print(tudui)
# 对网络结构进行一个简单的检验
input=torch.ones((64,3,32,32))
output=tudui(input)
print(output.shape)

writer=SummaryWriter("logs_seq")
writer.add_graph(tudui,input)

writer.close()


