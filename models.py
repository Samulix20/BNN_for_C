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


class AutoencoderBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.l = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.f = nn.ReLU()

    def forward(self, x):
        x = self.l(x)
        x = self.bn(x)
        x = self.f(x)
        return x

class AUTOENCODER_TINY(nn.Module):

    def __init__(self, num_features):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            AutoencoderBlock(num_features, 128),
            AutoencoderBlock(128, 128),
            AutoencoderBlock(128, 128),
            AutoencoderBlock(128, 128),
            AutoencoderBlock(128, 8)
        )

        self.decoder = torch.nn.Sequential(
            AutoencoderBlock(8, 128),
            AutoencoderBlock(128, 128),
            AutoencoderBlock(128, 128),
            AutoencoderBlock(128, 128),
            nn.Linear(128, num_features),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x