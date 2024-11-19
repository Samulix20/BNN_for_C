import os

import numpy as np

import torch
import torch.nn.functional as F

import torchvision
from torchvision import transforms

from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn

import models

class Conf:
    # Try Use GPU
    use_cuda = True

    # BNN conversion hyperparmeters
    bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Flipout",  # Flipout or Reparameterization
        "moped_enable": True,  # True to initialize mu/sigma from the pretrained dnn weights
        "moped_delta": 0.5,
    }

    hyper_model_list = ["BO", "IP", "KSC", "PU", "SV"]

    model_list = hyper_model_list + ["LENET", "B2N2", "RESNET"]

    hyper_models_params = {
        "BO": {"num_in": 145, "num_out": 14},
        "IP": {"num_in": 200, "num_out": 16},
        "KSC": {"num_in": 176, "num_out": 13},
        "PU": {"num_in": 103, "num_out": 9},
        "SV": {"num_in": 204, "num_out": 16},
    }

def init_folders():
    os.system(f"""
        mkdir -p Data
        mkdir -p Code
        mkdir -p Figures
        mkdir -p Model
        mkdir -p Predictions
    """)

def baseline_path(model:str):
    return f"Predictions/{model}-baseline.npz"

def prediction_path(modelname:str, generation_method:str, fixed_bits:int):
    return f"Predictions/{modelname}-{generation_method}-{fixed_bits}.npz"

def get_device():
    use_cuda = torch.cuda.is_available() and Conf.use_cuda
    if use_cuda:
        return torch.device("cuda")
    else:
        return torch.device("cpu")

class HyperDataset():
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).float(), self.targets[idx]


def get_data(model:str):

    if model in Conf.hyper_model_list:
        dts = np.load(f"Data/{model}.npz")
        train_data = HyperDataset(dts["train_data"], dts["train_targets"])
        test_data = HyperDataset(dts["test_data"], dts["test_targets"])
        return train_data, test_data

    if not model == "RESNET":
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.ToTensor()

    train_data = torchvision.datasets.CIFAR10('Data', train=True, download=True, transform=transform)
    test_data = torchvision.datasets.CIFAR10('Data', train=False, transform=transform)

    return train_data, test_data

def get_model(model:str, state:str):

    def nll_loss_sum(x, target):
        return F.nll_loss(torch.log(x), target, reduction="sum")
    lf = nll_loss_sum

    if model == "RESNET":
        m = models.RESNET_TINY()
    elif model == "B2N2":
        m = models.B2N2()
    elif model == "LENET":
        m = models.LENET()
    elif model in Conf.hyper_model_list:
        m = models.HYPER(**Conf.hyper_models_params[model])

    if state == "untrained":
        pass

    elif state == "trained":
        m.load_state_dict(torch.load(f"Model/best_{model}", weights_only=True))
        dnn_to_bnn(m, Conf.bnn_prior_parameters)

    elif state == "bnn":
        dnn_to_bnn(m, Conf.bnn_prior_parameters)
        m.load_state_dict(torch.load(f"Model/best_bnn_{model}", weights_only=True))

    return m, lf
