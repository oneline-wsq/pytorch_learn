from PIL import Image
from torchvision import transforms

image_path="数据集/dataset/train/ants_image/0013035.jpg"
img=Image.open(image_path)
print(img) # <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=768x512 at 0x1FA82EDB640>

tensor_trans=transforms.ToTensor() # 这里是创建了一个类的实例对象
tensor_img=tensor_trans(img) # 直接以“对象名（）”的形式进行调用

