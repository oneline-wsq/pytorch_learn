# PyTorch深度学习快速入门教程（绝对通俗易懂！）【小土堆】

labels: CV, b站, pytorch, 深度学习, 编程
stars: ⭐⭐⭐⭐
开始-截止: March 22, 2022
状态: 进行中
类型: 科研



# P4: Python学习中的两大法宝函数

dir(): 打开，看见

help(): 说明书

相当于一个工具箱，dir打开工具箱；help里面工具的说明书；

![注意这里没有()](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled.png)

注意这里没有()

这里的输出有前后下划线，表示一种规范，这个变量不允许被篡改。

一个函数就相当于一个道具，使用help

在pycharm控制台，也可以写多行：

![Untitled](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%201.png)

![Untitled](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%202.png)





# P5，P6：Dataset类代码实战

```python
"""关于os"""
import os

dir_path="D:/learn_pytorch/数据集/dataset/train/ants_image"
img_path_list=os.listdir(dir_path) # 文件夹下的所有东西变为一个列表

```

![Untitled](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%203.png)

```python
from torch.utils.data import Dataset
import cv2
from PIL import Image
import os
#help(Dataset)

class MyData(Dataset):

    def __init__(self,root_dir,label_dir):
        # 初始化
        self.root_dir=root_dir
        self.label_dir=label_dir
        self.path=os.path.join(self.root_dir,self.label_dir) #将两个地址拼接起来
        self.img_path=os.listdir(self.path) # 将每张图片的名称转为列表

    def __getitem__(self, idx):
        # 改写getitem
        img_name=self.img_path[idx]
        img_item_path=os.path.join(self.root_dir,self.label_dir,img_name)
        img=Image.open(img_item_path)
        label=self.label_dir
        return img,label
    def __len__(self):
        return len(self.img_path)

root_dir="数据集/dataset/train"
ants_label_dir="ants_image"
bees_label_dir="bees_image"
ants_dataset=MyData(root_dir,ants_label_dir)
bees_dataset=MyData(root_dir,bees_label_dir)

train_dataset=ants_dataset+bees_dataset # 因为继承了父类，父类中+运算符重载了，所以这里直接使用就好
```





# P7: TensorBoard的使用(一)

想要看一个类的具体定义，按住ctrl键点击类名，就会跳到定义。

![Untitled](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%204.png)

绘制y=x

```python
from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter("logs") # 创建一个实例，“logs为文件地址”
# writer.add_image()

# 绘制 y=x
for i in range(100):
    writer.add_scalar("y=x",i,i)

writer.close()
```

如何打开tensorboard

![Untitled](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%205.png)

在pycharm的terminal中输入 `tensorboard —logdir=logs`

![可以通过参数设置不同的端口，避免和服务器上的其他人一样，冲突。](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%206.png)

可以通过参数设置不同的端口，避免和服务器上的其他人一样，冲突。





# P8: TensorBoard的使用(二)

参考：[PyTorch下的Tensorboard 使用 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/103630393)

Tensorboard的工作流程简单来说是：

- 将代码运行过程中的，某些你关心的数据保存在一个文件夹中：

这一步由代码中的writer完成。

- 再读取这个文件夹中的数据，用浏览器显示出来：

这一步通过再命令行运行tensorboard完成。

首先导入tensorboard

```python
from torch.utils.tensorboard import SummaryWriter
```

这里SummaryWriter的作用就是，将数据以特定的格式存储到刚刚提到的那个文件夹中。

首先我们将其***实例化***：

```python
writer=SummaryWriter("logs") # 创建一个实例，“logs”为文件地址
```

`writer.add_image()`

```python
from PIL import Image
img=Image.open(image_path)
print(type(img))
<class 'PIL.JpegImagePlugin.JpegImageFile'>
import numpy as np
img_array=np.array(img) # 转为np.array数据类型
print(type(img_array))
<class 'numpy.ndarray'>
```

这里输入的图片要是*`torch.Tensor, numpy.array, or string/blobname`*其中任意一种类型。

```python
from torch.utils.tensorboard import  SummaryWriter
import numpy as np
from PIL import Image

writer=SummaryWriter("logs")
image_path="数据集/dataset/train/ants_image/0013035.jpg"
img_PIL=Image.open(image_path)
img_array=np.array(img_PIL)
print(type(img_array)) # <class 'numpy.ndarray'>,虽然数据类型正确了，但还是报错，因为还需要确认长宽与通道数
print(img_array.shape) # (512, 768, 3)

writer.add_image("test",img_array,1,dataformats='HWC')
# y=2x
for i in range(100):
    writer.add_scalar("y=2x",3*i,i)

writer.close()
```

![Untitled](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%207.png)





# P10： Transforms的使用（一）

![Untitled](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%208.png)

<aside>
⭐ Python中的__call__()方法：
该方法的功能类似于在类中重载()运算符，使得类实例对象可以像调用普通函数呢样，以“对象名（）”的形式使用。

</aside>

比如：

```python
tensor_trans=transforms.ToTensor() # 这里是创建了一个类的实例对象
tensor_img=tensor_trans(img) # 直接以“对象名（）”的形式进行调用
```

![Untitled](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%209.png)





# P12，13: 常见的Transforms(一)（二）

## Python 中 **call**的用法

```python
class Person:
    def __call__(self, name):
        print("__call__"+"Hello"+name)
    def hello(self,name):
        print("hello"+name)

person=Person()
person("zhangsan") # __call__ 不用.的方式调用，而是直接用这种形式调用
person.hello("lisa")
```

可以用`ctrl+P`键显示需要什么参数。

总结：

关注输入输出参数数据类型；

多查看官方文档；

关注方法需要什么参数；

不知道返回值的时候

- print
- print(type())





# P14: torchvision中的数据集使用

```python
import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_set=torchvision.datasets.CIFAR10("./CIFAR10",train=True,transform=dataset_transform,download=True)
test_set=torchvision.datasets.CIFAR10("./CIFAR10",train=False,transform=dataset_transform,download=True)

# print(test_set[0]) # (<PIL.Image.Image image mode=RGB size=32x32 at 0x1BD63D7A8B0>, 3)
# print(test_set.classes)
#
# img,target=test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()

writer=SummaryWriter("p10")
for i in range(10):
    img,target=test_set[i]
    writer.add_image("test_set",img,i)

writer.close()
```





# P15: DataLoader的使用

dataset是一个数据集（一幅扑克牌），而dataloader则是从dataset中取数据

![Untitled](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%2010.png)

![Untitled](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%2011.png)

batch_size: 每次取几张；

shuffle: 每次是否洗牌（False，顺序一样）

num_workers: 多进程；默认为0；（但有时会有问题）

drop_last：比如100张牌，每次取3张，还剩下1张牌，是否舍去。

dataset中有getitem()，返回(img,target)

```python
dataset_transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
test_set=torchvision.datasets.CIFAR10("./CIFAR10",train=False,transform=dataset_transform,download=True)
```

![Untitled](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%2012.png)

而dataloader(batch_size=4)时，是从dataset中取4个数据：

img0,target0=dataset[0]

img1,target1=dataset[1]

img2,target2=dataset[2]

img3,target3=dataset[3]

上述四张图片与相应的targets作为一组返回。

用for循环将图片取出来：

```python
for data in test_loader:
    imgs,targets=data
    print(imgs.shape)
    print(targets)
```

![Untitled](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%2013.png)





# P17： 土堆说卷积操作

torch.nn是torch.nn.functional的一个封装，以方便使用。

![Untitled](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%2014.png)

### stride

![Untitled](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%2015.png)

```python
import torch
import torch.nn.functional as F
input=torch.tensor([[1,2,0,3,1],
                    [0,1,2,3,1],
                    [1,2,1,0,0],
                    [5,2,3,1,1],
                    [2,1,0,1,1]]) # 二维数组，首位有几个连续的[就是几维数组

kernel=torch.tensor([[1,2,1],
                     [0,1,0],
                     [2,1,0]])

input=torch.reshape(input,(1,1,5,5)) #input – input tensor of shape (minibatch , in_channels , iH , iW)
kernel=torch.reshape(kernel,(1,1,3,3))

output=F.conv2d(input,kernel,stride=1)
print(output)
# tensor([[[[10, 12, 12],
          [18, 16, 16],
          [13,  9,  3]]]])
```

### padding

![Untitled](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%2016.png)

```python
output2=F.conv2d(input,kernel,stride=1,padding=1)
print(output2)
```

![Untitled](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%2017.png)





# P18: 神经网络-卷积层

### 关于输入输出尺寸计算公式：

[Conv2d - PyTorch 1.11.0 documentation](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d)

![Untitled](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%2018.png)

这里的N是一个batch的size.

![Untitled](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%2019.png)

### 参数：

![Untitled](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%2020.png)

### dilation: 空洞卷积

[conv_arithmetic/README.md at master · vdumoulin/conv_arithmetic (github.com)](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)

![Untitled](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%2021.png)

### ⭐关于in_channels与out_channels的理解

in_channels为输入图像的通道数，一般彩色图像为3；out_channels为输入的通道数。

如下图，如果为2，则用两个卷积核进行卷积，两个卷积核可能不同，最后两个卷积核卷积后的输出形成一个最终的2个通道的输出。

![Untitled](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%2022.png)





# P19: 神经网络-最大池化的使用

![Untitled](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%2023.png)

### ceil_mode:

=Ture: 在不足3*3时，仍然保留。

![44061D1A-F465-4C67-8A5C-57645F73F0DD.jpeg](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/44061D1A-F465-4C67-8A5C-57645F73F0DD.jpeg)

```python
import torch
from torch import nn
from torch.nn import MaxPool2d

input=torch.tensor([[1,2,0,3,1],
                    [0,1,2,3,1],
                    [1,2,1,0,0],
                    [5,2,3,1,1],
                    [2,1,0,1,1]],dtype=torch.float32)
input=torch.reshape(input,(-1,1,5,5))
print(input.shape)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()
        self.maxpool1=MaxPool2d(kernel_size=3,ceil_mode=True)

    def forward(self,input):
        output=self.maxpool1(input)
        return output

tudui=Tudui()
output=tudui(input)
print(output)
# torch.Size([1, 1, 5, 5])
# tensor([[[[2., 3.],
#          [5., 1.]]]])
```





# P20: 非线性激活

### ReLU()

![Untitled](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%2024.png)

![Untitled](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%2025.png)

### ***inplace=False： 默认为False***

- 如果`inplace=True`:
  
    input=-1
    
    ReLU(input,inplace=True)
    
    Input=0
    
- 如果`inplace=False`:
  
    input=-1
    
    Output=ReLU(input,inplace=False)
    
    input=-1
    
    output=0
    





# P21: 神经网络-线性层及其他层介绍

```python
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
```





# P22: 神经网络-搭建小实战的Sequential的使用

![Untitled](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%2026.png)

![Untitled](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%2027.png)

### `nn.Sequential()`的用法

在不用nn.Sequential()之前，代码如下：

```python
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1=nn.Conv2d(3,32,kernel_size=5,padding=2)
        self.maxpool1=nn.MaxPool2d(2)
        self.conv2=nn.Conv2d(32,32,kernel_size=5,padding=2)
        self.maxpool2=nn.MaxPool2d(2)
        self.conv3=nn.Conv2d(32,64,kernel_size=5,padding=2)
        self.maxpool3=nn.MaxPool2d(2)
        self.flatten=nn.Flatten()
        self.linear1=nn.Linear(1024,64)
        self.linear2=nn.Linear(64,10)
       
    def forward(self,x):
        x=self.conv1(x)
        x=self.maxpool1(x)
        x=self.conv2(x)
        x=self.maxpool2(x)
        x=self.conv3(x)
        x=self.maxpool3(x)
        x=self.flatten(x)
        x=self.linear1(x)
        x=self.linear2(x)
        return x
```

使用nn.Sequential()后：

```python
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
```

代码简洁了很多。





# P23: 损失函数与反向传播

### loss的作用：

1. 计算实际输出和目标之间的差距
2. 为我们更新输出提供一定的依据（反向传播）

### 交叉熵：

[CrossEntropyLoss - PyTorch 1.11.0 documentation](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss)

![269A36F1-5667-488C-8588-91AD8A4361D7.jpeg](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/269A36F1-5667-488C-8588-91AD8A4361D7.jpeg)

```python
for data in dataloader:
    imgs,targets=data
    outputs=tudui(imgs)
    result_loss=loss(outputs,targets) # 计算损失
    result_loss.backward() #计算梯度
    print("ok")
```





# P24: 优化器

[torch.optim — PyTorch 1.11.0 documentation](https://pytorch.org/docs/stable/optim.html)

![Untitled](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%2028.png)

`optimizer.zero_grad()` 将上一步的梯度清零，以防对下面的运算造成影响，这一步一定要写。

```python
import torch.optim
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
# 设置优化器
optim=torch.optim.SGD(tudui.parameters(),lr=0.01)

for epoch in range(20):
    running_loss=0.0
    for data in dataloader:
        imgs,targets=data
        outputs=tudui(imgs)
        result_loss=loss(outputs,targets) # 计算损失
        optim.zero_grad()
        result_loss.backward()
        optim.step() # 对每个参数进行调优
        running_loss+=result_loss
    print("epoch[{}]: loss:{}".format(epoch,running_loss))
```

输出结果：

![Untitled](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%2029.png)

可以看到，loss随着迭代次数的增加在减小。





# P25: 现有网络模型的使用及修改

[Models and pre-trained weights - Torchvision 0.12 documentation](https://pytorch.org/vision/stable/models.html)

以VGG16为例: 

![Untitled](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%2030.png)

如果`pretrained=True`，相当于在这个数据集中已经训练好了；

progress：是否显示进度条

### 调用vgg-16:

```python
import torchvision

# train_data=torchvision.datasets.ImageNet("./ImageNet",split='train',download=True,transform=torchvision.transforms.ToTensor())
# 数据集太大了，而且官网不能直接访问，老师放弃下载了
from torch import nn

vgg16_False=torchvision.models.vgg16(pretrained=False)
vgg16_True=torchvision.models.vgg16(pretrained=True)
print(vgg16_True)
```

输出：

VGG(
(features): Sequential(
(0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(1): ReLU(inplace=True)
(2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(3): ReLU(inplace=True)
(4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
(5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(6): ReLU(inplace=True)
(7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(8): ReLU(inplace=True)
(9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
(10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(11): ReLU(inplace=True)
(12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(13): ReLU(inplace=True)
(14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(15): ReLU(inplace=True)
(16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
(17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(18): ReLU(inplace=True)
(19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(20): ReLU(inplace=True)
(21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(22): ReLU(inplace=True)
(23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
(24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(25): ReLU(inplace=True)
(26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(27): ReLU(inplace=True)
(28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(29): ReLU(inplace=True)
(30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
)
(avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
(classifier): Sequential(
(0): Linear(in_features=25088, out_features=4096, bias=True)
(1): ReLU(inplace=True)
(2): Dropout(p=0.5, inplace=False)
(3): Linear(in_features=4096, out_features=4096, bias=True)
(4): ReLU(inplace=True)
(5): Dropout(p=0.5, inplace=False)
(6): Linear(in_features=4096, out_features=1000, bias=True)
)
)

### 如果想要在vgg-16后面再加上一层：

```python
# 如果想要加一层
vgg16_True.add_module('add_linear',nn.Linear(1000,10))
print(vgg16_True)
```

输出：

VGG(
(features): Sequential(
(0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(1): ReLU(inplace=True)
(2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(3): ReLU(inplace=True)
(4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
(5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(6): ReLU(inplace=True)
(7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(8): ReLU(inplace=True)
(9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
(10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(11): ReLU(inplace=True)
(12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(13): ReLU(inplace=True)
(14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(15): ReLU(inplace=True)
(16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
(17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(18): ReLU(inplace=True)
(19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(20): ReLU(inplace=True)
(21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(22): ReLU(inplace=True)
(23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
(24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(25): ReLU(inplace=True)
(26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(27): ReLU(inplace=True)
(28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(29): ReLU(inplace=True)
(30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
)
(avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
(classifier): Sequential(
(0): Linear(in_features=25088, out_features=4096, bias=True)
(1): ReLU(inplace=True)
(2): Dropout(p=0.5, inplace=False)
(3): Linear(in_features=4096, out_features=4096, bias=True)
(4): ReLU(inplace=True)
(5): Dropout(p=0.5, inplace=False)
(6): Linear(in_features=4096, out_features=1000, bias=True)
)
(add_linear): Linear(in_features=1000, out_features=10, bias=True)
)

### 如果想要在classifier中加

```python
# 如果想要在里面加一层
vgg16_True.classifier.add_module('add_linear',nn.Linear(1000,10))
print(vgg16_True)
```

VGG(
(features): Sequential(
(0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(1): ReLU(inplace=True)
(2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(3): ReLU(inplace=True)
(4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
(5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(6): ReLU(inplace=True)
(7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(8): ReLU(inplace=True)
(9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
(10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(11): ReLU(inplace=True)
(12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(13): ReLU(inplace=True)
(14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(15): ReLU(inplace=True)
(16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
(17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(18): ReLU(inplace=True)
(19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(20): ReLU(inplace=True)
(21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(22): ReLU(inplace=True)
(23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
(24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(25): ReLU(inplace=True)
(26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(27): ReLU(inplace=True)
(28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(29): ReLU(inplace=True)
(30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
)
(avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
(classifier): Sequential(
(0): Linear(in_features=25088, out_features=4096, bias=True)
(1): ReLU(inplace=True)
(2): Dropout(p=0.5, inplace=False)
(3): Linear(in_features=4096, out_features=4096, bias=True)
(4): ReLU(inplace=True)
(5): Dropout(p=0.5, inplace=False)
(6): Linear(in_features=4096, out_features=1000, bias=True)
(add_linear): Linear(in_features=1000, out_features=10, bias=True)
)
(add_linear): Linear(in_features=1000, out_features=10, bias=True)
)

Process finished with exit code 0

### 想要修改某一层

修改前

![Untitled](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%2031.png)

```python
# 如果想要修改一层
vgg16_True.classifier[6]=nn.Linear(4096,101)
print(vgg16_True)
```

![Untitled](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%2032.png)





# P26: 网络模型的保存与读取

如何学习优秀的代码写法：看pytoch官网的代码和模板

一共有两种方式：

```python
import torch
import torchvision

vgg16=torchvision.models.vgg16(pretrained=False)
# 第一种保存方式
torch.save(vgg16,"vgg16_method1.pth")

# 第二种保存方式(官方推荐：因为内存小)
# 将vgg16中的参数保存成字典形式
torch.save(vgg16.state_dict(),"vgg16_method2.pth")
```

```python
import torch
# 对应保存方式1：加载模型
import torchvision

model=torch.load("vgg16_method1.pth")
print(model)

# 对应保存方式2：加载模型
model=torch.load("vgg16_method2.pth")
print(model)
# 如果要恢复成网络模型的形式
vgg16=torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
print(vgg16)
```





# P27-29: 完整的模型训练套路

首先构建网络，单独建一个文件：

```python
# 搭建神经网络
import torch
from torch import nn

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

if __name__ == '__main__':
    """很多人喜欢在这里测试这个网络的正确性"""
    tudui=Tudui()
    input=torch.ones(64,3,32,32)
    output=tudui(input) #torch.Size([64, 10])
    print(output.shape)
```

```python
import torch
import torchvision

# 准备数据集
from torch import nn
from torch.utils.data import DataLoader
from P27_model import * # *代表将里面的所有东西都引用过来

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

# 创建网络模型
tudui=Tudui()

# 损失函数
loss_fn=nn.CrossEntropyLoss()
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
```





# P30-31: 利用GPU训练

两种GPU训练方式：

## 第一种：

![Untitled](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%2033.png)

![Untitled](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%2034.png)

![Untitled](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%2035.png)

![Untitled](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%2036.png)

![Untitled](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%2037.png)

加上is_avaliable判断，是否能在GPU上跑，优先在GPU上跑。

![Untitled](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%2038.png)

## 第二种方法：

![注意实际上torch，t小写](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%2039.png)

注意实际上torch，t小写

![Untitled](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%2040.png)

![这里tudui，网络模型可以不用写赋值](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%2041.png)

这里tudui，网络模型可以不用写赋值

![同样，损失函数也可以不用写赋值](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%2042.png)

同样，损失函数也可以不用写赋值

![只有数据：图像与标签需要赋值](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%2043.png)

只有数据：图像与标签需要赋值

## 关于device的写法：

如果单张显卡：

![Untitled](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%2044.png)

这两者相同，没有区别

但是也有人这样写：

![Untitled](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%2045.png)





# P32: 完整的模型验证（测试，demo）套路

利用已经训练好的模型，然后给它提供输入。

路径中的点****“./”、“../”、“/”代表的含义：****

- **“./”：代表目前所在的目录**
- **“. ./”：代表上一层目录**
- **“/”：代表根目录**

比如访问imgs中的dogs，先../ 回到上一层目录，再访问

![Untitled](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%2046.png)

![Untitled](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%2047.png)

如果模型一开始是在GPU上训练，要将他映射到CPU上。





# P33: 看看开源项目

视频中的示例项目：

[https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

![Untitled](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%2048.png)

readme中，有这样一行，说训练模型，可以通过下面的一行指令实现：

![Untitled](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%2049.png)

![Untitled](PyTorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%20e749d/Untitled%2050.png)

如果有required=True, 则要给他参数

而像name, 有一个默认的参数

视频作者一般喜欢把required删掉，然后写default

[https://www.notion.so](https://www.notion.so)