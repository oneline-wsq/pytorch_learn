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