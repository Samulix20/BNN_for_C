import torch
import torch.nn as nn
import torch.nn.functional as F

from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss
from bayesian_torch.layers.flipout_layers import conv_flipout as bnn_conv, linear_flipout as bnn_linear

from .model_info import *

# Residual block, required for creating the residual buffers

class AddResidual(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y):
        x = x + y
        return x

class ResidualConv(nn.Module):
    def __init__(self, l):
        super().__init__()
        self.l = l

    def forward(self, x):
        x = self.l(x)
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
            self.extra_conv = ResidualConv(c1)

        self.res = AddResidual()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.bn2(y)

        if(self.diff):
            x = self.extra_conv(x)

        y = self.res(x, y)
        y = self.relu2(y)
        return y

def info_from_model(model: nn.Module, name: str) -> ModelInfo:
    """
    Model linear layers must be trained with a flattened input tensor with shape
    [width x height x channels]. Care by default PyTorch layers use shape
    [channels x width x height]
    """

    model_info = ModelInfo(name)
    for n, t in model.named_modules():
        if n == "":
            continue
        l = LayerInfo()
        l.name = n.replace(".", "_")
        if isinstance(t, nn.ReLU):
            l.type = "ReLU"
            l.is_activation = True
        elif isinstance(t, nn.Softmax):
            l.type = "Softmax"
            l.is_activation = True
        elif isinstance(t, nn.MaxPool2d):
            l.type = "MaxPool2D"
            l.kernel_size = t.kernel_size
        elif isinstance(t, nn.AvgPool2d):
            l.type = "AvgPool2D"
            l.kernel_size = t.kernel_size
        elif isinstance(t, bnn_conv.Conv2dFlipout):
            l.type = "Conv2D"
            # Padding must be "same" or "valid" == (0,0)
            if t.padding == (0,0) or t.padding == "valid":
                l.padding = "valid"
            else:
                l.padding = t.padding
            l.kernel_size = t.kernel_size
            l.stride = t.stride
            l.in_channels = t.in_channels
            l.out_channels = t.out_channels
            # Buffers
            l.mu_buffer = t.mu_kernel.permute((0,2,3,1)).detach().numpy()
            l.sigma_buffer = F.softplus(t.rho_kernel).permute((0,2,3,1)).detach().numpy()
            l.mu_bias = t.mu_bias.detach().numpy()
            l.sigma_bias = F.softplus(t.rho_bias).detach().numpy()
        elif isinstance(t, bnn_linear.LinearFlipout):
            l.type = "Linear"
            l.in_features = t.in_features
            l.out_features = t.out_features
            # Buffers
            l.mu_buffer = t.mu_weight.detach().numpy()
            l.sigma_buffer = F.softplus(t.rho_weight).detach().numpy()
            l.mu_bias = t.mu_bias.detach().numpy()
            l.sigma_bias = F.softplus(t.rho_bias).detach().numpy()
        elif isinstance(t, nn.BatchNorm2d):
            l.type = "BatchNorm2D"
        elif isinstance(t, ResidualBlock):
            l.type = "ResidualBlock"
        elif isinstance(t, AddResidual):
            l.type = "ResidualAdd"
            l.input_from_residual = True
        elif isinstance(t, ResidualConv):
            l.type = "ResidualConv"
        elif isinstance(t, nn.Flatten):
            continue
        model_info.layers.append(l)
    model_info.prune()
    model_info.fold_layers()
    model_info.layers[0].is_input = True
    return model_info
