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




