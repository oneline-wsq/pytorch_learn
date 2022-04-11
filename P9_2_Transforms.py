from PIL import  Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

image_path="数据集/dataset/train/ants_image/0013035.jpg"
img=Image.open(image_path)

writer=SummaryWriter("logs")
# 1. transforms该如何使用
tensor_trans=transforms.ToTensor()
tensor_img=tensor_trans(img)

writer.add_image("Tensor_img",tensor_img)

writer.close()
