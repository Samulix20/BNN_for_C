import torch
import torch.nn as nn
import torch.nn.functional as F

from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss
from bayesian_torch.layers.flipout_layers import conv_flipout as bnn_conv, linear_flipout as bnn_linear

from .model_info import *

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


# V0. Compresion layers
# They calculate the distribution of the outputs and then resample them

class CompresionModel(nn.Module):

    def __init__(self, layer_list):
        super().__init__()
        self.layers = nn.ModuleList(layer_list)
    
    def forward(self, x):
        y_mu = x
        y_sigma = torch.zeros_like(x)
        for l in self.layers:
            y_mu, y_sigma, y = l.forward(y_mu, y_sigma)
        return y

class CompresionLayer(nn.Module):
    def __init__(self, b: BasicBlock, nmc):
        super().__init__()
        self.block = b
        self.nmc = nmc

    def forward(self, x_mu, x_sigma):
        
        x_b = x_sigma * sqrt(12)
        x_a = x_mu - x_b / 2
        x_u = torch.zeros_like(x_b)

        output_list = []
        for _ in range(self.nmc):
            
            #x = torch.normal(x_mu, x_sigma)
            x = x_u.uniform_() * x_b + x_a
            
            x = self.block.forward(x)
            output_list.append(x)
        
        y = torch.stack(output_list)
        y_mu = torch.mean(y, 0)
        y_sigma = torch.std(y, 0)
        
        return y_mu, y_sigma, y


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
        l = LayerInfo(n.replace(".", "_"))

        if isinstance(t, nn.MaxPool2d):
            l = Pool2DInfo(l, "max", t.kernel_size)

        elif isinstance(t, nn.AvgPool2d):
            l = Pool2DInfo(l, "avg", t.kernel_size)

        elif isinstance(t, bnn_conv.Conv2dFlipout):
            l = Conv2DInfo(l)
            
            l.padding = t.padding
            l.kernel_size = np.array(t.kernel_size)
            l.stride = np.array(t.stride)
            l.in_channels = t.in_channels
            l.out_channels = t.out_channels
            # Buffers
            l.mu_buffer = t.mu_kernel.permute((0,2,3,1)).detach().numpy()
            l.sigma_buffer = F.softplus(t.rho_kernel).permute((0,2,3,1)).detach().numpy()
            l.mu_bias = t.mu_bias.detach().numpy()
            l.sigma_bias = F.softplus(t.rho_bias).detach().numpy()
        
        elif isinstance(t, bnn_linear.LinearFlipout):
            l = LinearInfo(l)
            
            l.in_features = t.in_features
            l.out_features = t.out_features
            # Buffers
            l.mu_buffer = t.mu_weight.detach().numpy()
            l.sigma_buffer = F.softplus(t.rho_weight).detach().numpy()
            l.mu_bias = t.mu_bias.detach().numpy()
            l.sigma_bias = F.softplus(t.rho_bias).detach().numpy()

        elif isinstance(t, nn.ReLU):
            l = FoldableInfo(l, "ReLU")
        elif isinstance(t, nn.Softmax):
            l = FoldableInfo(l, "Softmax")

        elif isinstance(t, nn.BatchNorm2d):
            l = BatchNorm2DInfo(l)
            l.bn_gamma = t.weight.detach().numpy()
            l.bn_beta = t.bias.detach().numpy()
            l.bn_mean = t.running_mean.detach().numpy()
            l.bn_var = t.running_var.detach().numpy()
            l.bn_eps = t.eps

        elif isinstance(t, ResidualBlock):
            l = FoldableInfo(l, "ResidualBlock")
        elif isinstance(t, ResidualConv):
            l = FoldableInfo(l, "ResidualConv")
        elif isinstance(t, AddResidual):
            l = ResidualAddInfo(l)

        else:
            # Not implemented layers
            continue
        
        model_info.layers.append(l)

    model_info.prune()
    model_info.fold_layers()
    model_info.layers[0].is_input = True
    return model_info
