from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter("logs") # 创建一个实例，“logs为文件地址”
# writer.add_image()

# 绘制 y=x
for i in range(100):
    writer.add_scalar("y=2x",2*i,i) # 两个i分别代表y轴和x轴

writer.close()
