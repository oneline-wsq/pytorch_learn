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
