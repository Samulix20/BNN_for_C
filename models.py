import torch
import torch.nn as nn

from bnnc.torch import ResidualBlock, BasicBlock, CompresionLayer, CompresionModel

# Model definitions

class HYPER(nn.Module):
    
    def __init__(self, num_in, num_out):
        super().__init__()
        self.l = torch.nn.Sequential(
            nn.Linear(num_in, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_out),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.l(x)
        return x

class NEW_HYPER(nn.Module):

    def __init__(self, old_module, nmc):
        super().__init__()

        self.cm = CompresionModel([
            CompresionLayer(BasicBlock(old_module.get_submodule("l.0"), "relu"), nmc),
            CompresionLayer(BasicBlock(old_module.get_submodule("l.2"), "relu"), nmc),
            CompresionLayer(BasicBlock(old_module.get_submodule("l.4"), "softmax"), nmc)
        ])

    def forward(self, x):
        return self.cm.forward(x)


class B2N2(nn.Module):

    def __init__(self):
        super().__init__()
        self.l = torch.nn.Sequential(
            BasicBlock(nn.Conv2d(1, 32, 3, 1, "same"), "relu"),
            BasicBlock(nn.Conv2d(32, 32, 3, 1, "same"), "relu", pool=nn.MaxPool2d(2)),
            BasicBlock(nn.Conv2d(32, 64, 3, 1, "same"), "relu"),
            BasicBlock(nn.Conv2d(64, 64, 3, 1, "same"), "relu", pool=nn.MaxPool2d(2)),
            BasicBlock(nn.Conv2d(64, 128, 3, 1, "same"), "relu"),
            BasicBlock(nn.Conv2d(128, 128, 3, 1, "same"), "relu", pool=nn.MaxPool2d(2)),
            BasicBlock(nn.Linear(2048, 10), "softmax", flatten=True)
        )
    
    def forward(self, x):
        x = self.l(x)
        return x


class LENET(nn.Module):

    def __init__(self):
        super().__init__()
        self.l = torch.nn.Sequential(
            BasicBlock(nn.Conv2d(1, 32, 3, 1), "relu"),
            BasicBlock(nn.Conv2d(32, 64, 3, 1), "relu", pool=nn.MaxPool2d(2)),
            BasicBlock(nn.Linear(12544, 128), "relu", flatten=True),
            BasicBlock(nn.Linear(128, 10), "softmax")
        )

    def forward(self, x):
        x = self.l(x)
        return x


class NEW_LENET(nn.Module):

    def __init__(self, old_module, nmc):
        super().__init__()
        self.cm = CompresionModel([
            CompresionLayer(old_module.get_submodule("l.0"), nmc),
            CompresionLayer(old_module.get_submodule("l.1"), nmc),
            CompresionLayer(old_module.get_submodule("l.2"), nmc),
            CompresionLayer(old_module.get_submodule("l.3"), nmc)
        ])
    
    def forward(self, x):
        return self.cm.forward(x)

class AddResidual(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x + y


class RESNET_TINY(nn.Module):

    def __init__(self):
        super().__init__()

        self.l1 = torch.nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            ResidualBlock(16, 16, 1),
            ResidualBlock(16, 32, 2),
            ResidualBlock(32, 64, 2),
            nn.AvgPool2d(8)
        )

        self.l2 = torch.nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.l1(x)
        x = x.permute((0,2,3,1))
        x = self.l2(x)
        return x
