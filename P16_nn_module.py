import torch
from torch import nn


class Tudui(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self,input):
        output=input+1
        return output

tudui=Tudui()
x=torch.tensor(1.0)
output=tudui(x) # 这里可以直接在括号里是因为集成的nn.Module中forward方法是__call__()方法的实现，可调用对象会调用__call__()方法
print(output)
