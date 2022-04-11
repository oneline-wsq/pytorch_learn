import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# 准备的测试数据集


dataset_transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
test_set=torchvision.datasets.CIFAR10("./CIFAR10",train=False,transform=dataset_transform,download=True)
test_loader=DataLoader(dataset=test_set,batch_size=64,shuffle=True,num_workers=0,drop_last=False)

# 测试数据集中第一张图片
img,target=test_set[0]
print(img.shape)
print(target)

writer=SummaryWriter("dataLoader")
for epoch in range(2):
    step=0
    for data in test_loader:
        imgs,targets=data
        # print(imgs.shape)
        # print(targets)
        writer.add_images("Epoch: {}".format(epoch),imgs,step)
        step=step+1

writer.close()