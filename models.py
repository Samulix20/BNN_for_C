import torch
import torch.nn as nn

from bnnc.torch import ResidualBlock

# Model definitions

class BasicBlock(nn.Module):
    
    def __init__(self, layer: nn.Module, activation: str, flatten: bool = False, pool: nn.Module = None):
        super().__init__()
        self.layer = layer
    
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "softmax":
            self.activation = nn.Softmax(dim=1)

        self.pool = pool

        self.flatten = flatten

    def forward(self, x):

        if self.flatten:
            x = x.permute((0,2,3,1)).flatten(1)

        x = self.layer(x)
        x = self.activation(x)

        if self.pool is not None:
            x = self.pool(x)

        return x


class B2N2(nn.Module):

    def __init__(self):
        super().__init__()
        self.l1 = BasicBlock(nn.Conv2d(1, 32, 3, 1, "same"), "relu")
        self.l2 = BasicBlock(nn.Conv2d(32, 32, 3, 1, "same"), "relu", pool=nn.MaxPool2d(2))
        self.l3 = BasicBlock(nn.Conv2d(32, 64, 3, 1, "same"), "relu")
        self.l4 = BasicBlock(nn.Conv2d(64, 64, 3, 1, "same"), "relu", pool=nn.MaxPool2d(2))
        self.l5 = BasicBlock(nn.Conv2d(64, 128, 3, 1, "same"), "relu")
        self.l6 = BasicBlock(nn.Conv2d(128, 128, 3, 1, "same"), "relu", pool=nn.MaxPool2d(2))
        self.l7 = BasicBlock(nn.Linear(2048, 10), "softmax", flatten=True)
    
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        return x


class LENET(nn.Module):

    def __init__(self):
        super().__init__()
        self.l1 = BasicBlock(nn.Conv2d(1, 32, 3, 1), "relu")
        self.l2 = BasicBlock(nn.Conv2d(32, 64, 3, 1), "relu", pool=nn.MaxPool2d(2))
        self.l3 = BasicBlock(nn.Linear(12544, 128), "relu", flatten=True)
        self.l4 = BasicBlock(nn.Linear(128, 10), "softmax")

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        return x


class AddResidual(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x + y


class RESNET_TINY(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv0 = nn.Conv2d(3, 16, 3, 1, "same")
        self.bn0 = nn.BatchNorm2d(16)
        self.relu0 = nn.ReLU()

        self.stack1 = ResidualBlock(16, 16, 1)
        self.stack2 = ResidualBlock(16, 32, 2)
        self.stack3 = ResidualBlock(32, 64, 2)

        self.avg1 = nn.AvgPool2d(8)
        self.flat1 = nn.Flatten()
        self.fc1 = nn.Linear(64, 10)
        self.sfm1 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.stack1(x)
        x = self.stack2(x)
        x = self.stack3(x)
        x = self.avg1(x)
        x = self.flat1(x)
        x = self.fc1(x)
        x = self.sfm1(x)
        return x


