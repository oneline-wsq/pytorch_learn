import torch
import torchvision

vgg16=torchvision.models.vgg16(pretrained=False)
# 第一种保存方式
torch.save(vgg16,"vgg16_method1.pth")

# 第二种保存方式(官方推荐：因为内存小)
# 将vgg16中的参数保存成字典形式
torch.save(vgg16.state_dict(),"vgg16_method2.pth")