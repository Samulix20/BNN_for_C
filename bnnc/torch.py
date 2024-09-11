import torch
import torch.nn as nn
import torch.nn.functional as F

from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss
from bayesian_torch.layers.flipout_layers import conv_flipout as bnn_conv, linear_flipout as bnn_linear

from .model_info import *


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
        l.name = n
        if isinstance(t, nn.ReLU):
            l.type = "ReLU"
            l.is_activation = True
        elif isinstance(t, nn.LogSoftmax):
            # LogSoftmax can be turned into Softmax for inference
            l.type = "Softmax"
            l.is_activation = True
        elif isinstance(t, nn.MaxPool2d):
            l.type = "MaxPool2D"
            l.kernel_size = t.kernel_size
        elif isinstance(t, bnn_conv.Conv2dFlipout):
            l.type = "Conv2D"
            # Padding must be "same" or "valid" == (0,0)
            if t.padding == (0,0):
                l.padding = "valid"
            elif t.padding != "same" and t.padding != "valid":
                panic()
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
        model_info.layers.append(l)
    model_info.prune()
    model_info.fuse_activations()
    model_info.layers[0].is_input = True
    return model_info
