import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10("./CIFAR10",train=False,transform=torchvision.transforms.ToTensor(),download=True)

dataloader=DataLoader(dataset,batch_size=64)

class Tudui(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.conv1=Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)
        # in_channels (int): Number of channels in the input image, 这里输入图像为彩色图像，2层
        # out_channels (int): Number of channels produced by the convolution，输出设为6层
        # kernel_size (int or tuple): Size of the convolving kernel

    def forward(self,x):
        x=self.conv1(x)
        return x

tudui=Tudui()
print(tudui) #Tudui((conv1): Conv2d(3, 6, kernel_size=(3, 3), stride=(1, 1)))
writer=SummaryWriter("nnconv2d")
step=0
for data in dataloader:
    imgs,targets=data
    output=tudui(imgs)
    print(imgs.shape) # torch.Size([64, 3, 32, 32]) ,64张图像
    print(output.shape) # torch.Size([64, 6, 30, 30])
    writer.add_images("conv2d_input",imgs,step)
    output=torch.reshape(output,(-1,3,30,30)) # 写-1,程序计算
    writer.add_images("conv2d_output",output,step)
    step=step+1

writer.close()