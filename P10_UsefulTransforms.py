from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer=SummaryWriter("logs")
img=Image.open("images/testface.jpg")
print(img)

# ToTensor
trans_totensor=transforms.ToTensor()
img_tensor=trans_totensor(img)
writer.add_image("ToTensor",img_tensor)

# Normalize
print(img_tensor[0][0][0])
trans_norm=transforms.Normalize([0.5,2,0.5],[10,0.5,0.5]) # 这里的公式是(img-mean)/std=2*img-1; 相当于从[0,1]变为[-1,1]
img_norm=trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize",img_norm,2)

# Resize
print(img.size)
trans_resize=transforms.Resize([512,512])
img_resize=trans_resize(img_tensor)
print(img_resize.shape)
writer.add_image("Resize",img_resize)

# Compose
trans_resize2=transforms.Resize(512)
trans_compose=transforms.Compose([trans_resize2,trans_totensor])
img_resize=trans_compose(img)
print(img_resize.shape)

# RandomCrop 随机裁剪
trans_random=transforms.RandomCrop((512,1000))
trans_compose_2=transforms.Compose([trans_random,trans_totensor])
for i in range(10):
    img_crop=trans_compose_2(img)
    writer.add_image("RandomCrop",img_crop,i)
    print(img_crop.shape)

writer.close()
