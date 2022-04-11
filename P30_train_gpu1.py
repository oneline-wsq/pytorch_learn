import torch
import torchvision

# 准备数据集
from torch import nn
from torch.utils.data import DataLoader
#from P27_model import * # *代表将里面的所有东西都引用过来

train_data=torchvision.datasets.CIFAR10("CIFAR10",train=True,transform=torchvision.transforms.ToTensor())
test_data=torchvision.datasets.CIFAR10("CIFAR10",train=False,transform=torchvision.transforms.ToTensor())

# length 长度
train_data_size=len(train_data)
test_data_size=len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 利用dataloader加载数据集
train_dataloader=DataLoader(train_data,batch_size=64)
test_dataloader=DataLoader(test_data,batch_size=64)

# 创建模型
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

# 创建网络模型
tudui=Tudui()
if torch.cuda.is_available():
    tudui=tudui.cuda()

# 损失函数
loss_fn=nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn=loss_fn.cuda()
# 优化器
learning_rate=0.01
optimizer=torch.optim.SGD(tudui.parameters(),lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step=0
# 记测试的次数
total_test_step=0
# 训练的轮数
epoch=10

for i in range(epoch):
    print("----------------第{}轮训练开始----------------".format(i+1))

    # 训练
    tudui.train() # 不写也可以运行，作用比较小，只对某些层起作用
    for data in train_dataloader:
        imgs,targets=data
        if torch.cuda.is_available():
            imgs=imgs.cuda()
            targets=targets.cuda()
        outputs=tudui(imgs)
        loss=loss_fn(outputs,targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step=total_train_step+1
        if total_train_step%100==0:
            print("训练次数：{}， loss:{}".format(total_train_step,loss.item()))

    # 测试步骤
    tudui.eval() # 不写也可以运行
    total_test_loss=0
    total_accuracy=0
    with torch.no_grad(): # with里面的代码没有梯度
        for data in test_dataloader:
            imgs,targets=data
            if torch.cuda.is_available():
                imgs=imgs.cuda()
                targets=targets.cuda()

            outputs=tudui(imgs)
            loss=loss_fn(outputs,targets)
            total_test_loss+=loss.item() # 加到整体的loss上
            accuracy=(outputs.argmax(1)==targets).sum() # 1代表横向比较
            total_accuracy+=accuracy
    print("整体测试集上的Loss:{}".format(total_test_loss))
    print("整体测试集上的准确率：{}".format(total_accuracy/test_data_size))
    total_test_step+=1
    torch.save(tudui,"tudui_{}.pth".format(i))
    print("模型已保存")
