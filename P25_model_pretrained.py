import torchvision

# train_data=torchvision.datasets.ImageNet("./ImageNet",split='train',download=True,transform=torchvision.transforms.ToTensor())
# 数据集太大了，而且官网不能直接访问，老师放弃下载了
from torch import nn

vgg16_False=torchvision.models.vgg16(pretrained=False)
vgg16_True=torchvision.models.vgg16(pretrained=True)

print(vgg16_True)

train_data=torchvision.datasets.CIFAR10('./CIFAR10',train=True,transform=torchvision.transforms.ToTensor())

# 如果想要加一层
vgg16_True.add_module('add_linear',nn.Linear(1000,10))
print(vgg16_True)

# 如果想要在里面加一层
vgg16_True.classifier.add_module('add_linear',nn.Linear(1000,10))
print(vgg16_True)

# 如果想要修改一层
vgg16_True.classifier[6]=nn.Linear(4096,10)
print(vgg16_True)