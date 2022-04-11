import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import writer
from torch import nn

dataset=torchvision.datasets.CIFAR10("./CIFAR10",train=False,transform=torchvision.transforms.ToTensor())
dataloader=DataLoader(dataset,batch_size=1)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
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
        x=self.model1(x)
        return x

loss=nn.CrossEntropyLoss()


tudui=Tudui()

for data in dataloader:
    imgs,targets=data
    outputs=tudui(imgs)
    result_loss=loss(outputs,targets) # 计算损失
    result_loss.backward() #计算梯度
    print("ok")