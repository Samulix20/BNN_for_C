import torch
import torch.nn as nn

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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        if stride == 1:
            c1 = nn.Conv2d(in_channels, out_channels, 3, 1, "same")
        else:
            c1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)

        self.conv1 = c1
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, "same")
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.diff = in_channels != out_channels
        if (self.diff):
            self.extra_conv = c1

        self.relu2 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.bn2(y)

        if(self.diff):
            x = self.extra_conv(x)

        y = x + y
        y = self.relu2(y)
        return y


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


