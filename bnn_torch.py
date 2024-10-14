import torch
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary

from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss
from bayesian_torch.layers.flipout_layers import conv_flipout as bnn_conv, linear_flipout as bnn_linear

import torchvision
from torchvision import transforms

import numpy as np
import numpy.typing as npt

import bnnc
import bnnc.torch

import time

class AppArgs:
    def __init__(self):
        self.train_batch_size = 32
        self.num_mc_train = 1
        self.num_mc_validation = 100
        self.num_epochs = 500
        self.learning_rate = 0.001
        self.dataset = "MNIST"
        self.model_name = "LENET"
        self.num_workers = 20
        self.max_img_per_worker = 10

    def model_path(self):
        return f"Model/bnn_{self.model_name}_{self.dataset}_{self.num_epochs}"

MODEL_NAMES = {
    "LENET": LENET,
    "B2N2": B2N2
}

def train_clasic(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(torch.log(output), target)
        loss.backward()
        optimizer.step()

def test_clasic(args, model, device, test_loader, prefix):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.exp().argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_loss += F.nll_loss(output, target)

    print(f'{prefix}: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)')


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for data, target in train_loader:
        
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        output_ = []
        kl_ = []
        for mc_run in range(args.num_mc):
            output = model(data)
            kl = get_kl_loss(model)
            output_.append(output)
            kl_.append(kl)
        output = torch.mean(torch.stack(output_), dim=0)
        kl = torch.mean(torch.stack(kl_), dim=0)

        loss = F.nll_loss(output, target, reduction='sum') + (kl / args.train_batch_size)
        loss.backward()
        optimizer.step()

def test(args, model, device, test_loader, prefix):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output_ = []
            kl_ = []
            for mc_run in range(args.num_mc):
                output = model(data)
                # log-softmax -> softmax
                output = torch.exp(output)
                kl = get_kl_loss(model)
                output_.append(output)
                kl_.append(kl)
            raw_preds = torch.stack(output_)
            output = torch.mean(raw_preds, dim=0)
            kl = torch.mean(torch.stack(kl_), dim=0)

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'{prefix}: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)')
    return raw_preds

def cuda_dev():
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

def create_model(args: AppArgs):
    const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Flipout",  # Flipout or Reparameterization
        "moped_enable": False,  # True to initialize mu/sigma from the pretrained dnn weights
        "moped_delta": 0.5,
    }
    model = MODEL_NAMES[args.model_name](args)
    dnn_to_bnn(model, const_bnn_prior_parameters)
    return model

def load_model(args):
    model = create_model(args)
    model.load_state_dict(torch.load(args.model_path(), weights_only=True))
    return model


def get_data_color(args: AppArgs):
    transform = transforms.ToTensor()
    train_data = torchvision.datasets.CIFAR10('Data', train=True, download=True, transform=transform)
    test_data = torchvision.datasets.CIFAR10('Data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.train_batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data))
    return (train_data, train_loader, test_data, test_loader)

def get_data(args: AppArgs):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()  
    ])

    train_data = vars(torchvision.datasets)[args.dataset]('Data', train=True, download=True, transform=transform)
    test_data = vars(torchvision.datasets)[args.dataset]('Data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.train_batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data))

    return (train_data, train_loader, test_data, test_loader)

def main_train():
    args = AppArgs()
    args.num_mc = args.num_mc_train

    device = cuda_dev()
    model = create_model(args)
    model.to(device)

    train_data, train_loader, test_data , test_loader = get_data(args)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.num_epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader, f"Epoch {epoch}")

    torch.save(model.state_dict(), args.model_path())

def main_validate():
    args = AppArgs()
    args.num_mc = args.num_mc_validation

    device = cuda_dev()
    model = load_model(args)
    model.to(device)

    train_data, train_loader, test_data , test_loader = get_data(args)

    preds = test(args, model, device, test_loader, "Validation").cpu().numpy()
    targets = np.array(test_data.targets)
    metrics, averages = bnnc.uncertainty.analyze_predictions(preds, targets)
    #all_plots((metrics, averages, preds), targets, "Figures")

    bnnc.plot.compare_predictions_plots((metrics, averages, preds), (metrics, averages, preds), targets, "Figures")
    bnnc.plot.free_plot_memory()


def panic(msg="ERROR"):
    print(msg)
    exit(1)

def parse_predictions(c_prediction_path):
    df = read_csv(c_prediction_path)
    bgm = []
    mc_samples = df["mcpass"].max() + 1
    for i in range(mc_samples):
        bgm.append(df[(df["mcpass"] == i)].filter(regex="class").values)
    return np.array(bgm)

import os
import subprocess
from pandas import read_csv
from multiprocessing import Pool
from io import StringIO

def parallel_c(x: tuple[int, npt.NDArray, bnnc.model_info.ModelInfo]):
    i, data, model_info = x
    worker_folder = f"Code/worker_{i}"

    os.system(f"""
        rm -rf {worker_folder}
        mkdir -p {worker_folder}
        cp -r {bnnc.model_info.c_sources_abspath}/bnn {worker_folder}
        cp {bnnc.model_info.c_sources_abspath}/Makefile {worker_folder}
        cp {bnnc.model_info.c_sources_abspath}/test_main.c {worker_folder}/main.c
        cp Code/bnn_config.h {worker_folder}
        cp Code/bnn_model.h {worker_folder}
        cp Code/bnn_model_weights.h {worker_folder}
    """)

    d = model_info.create_c_data(data)
    with open(f"{worker_folder}/test_data.h", "w") as f:
        f.write(d)

    os.system(f"""
        cd {worker_folder}
        make main > run.log
    """)

    return parse_predictions(f"{worker_folder}/run.log")

def main_info():
    args = AppArgs()
    model = load_model(args)
    train_data, train_loader, test_data, test_loader = get_data(args)

    for data, targets in test_loader:
        pass
    data = data.permute((0,2,3,1))
    input_shape = np.array(data[0].shape)
    flat_data = data.numpy().reshape((data.shape[0], -1))

    model_info = bnnc.torch.info_from_model(model, "bnn_model")
    model_info.calculate_buffers(input_shape)

    model_info.uniform_weight_transform()
    #model_info.bernoulli_weight_transform()

    l, h, w = model_info.create_c_code()
    with open("Code/bnn_config.h", "w") as f:
        f.write(l)
    with open("Code/bnn_model.h", "w") as f:
        f.write(h)
    with open("Code/bnn_model_weights.h", "w") as f:
        f.write(w)

    num_targets = args.num_workers * args.max_img_per_worker
    split_data = np.split(flat_data[:num_targets], args.num_workers)

    model_info.print_cinfo()

    with Pool(args.num_workers) as p:
        work = []
        for i, data in enumerate(split_data):
            work.append((i+1, data, model_info))

        time_start = time.time()
        print(f"{time.strftime("%H:%M:%S", time.localtime(time_start))} -- Starting C predictions")
        preds = np.concatenate(p.map(parallel_c, work), 1)
        preds[preds < 0] = 0
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start))
        print(f"{num_targets} img C preditions done in {elapsed_time} using {args.num_workers} threads")
        np.savez("Code/predictions", preds)

def main_pred_test():
    args = AppArgs()
    args.num_mc = args.num_mc_validation

    device = cuda_dev()

    model = load_model(args)
    model.to(device)

    num_targets = args.num_workers * args.max_img_per_worker

    train_data, train_loader, test_data, test_loader = get_data(args)
    for data, targets in test_loader:
        pass
    targets = targets[:num_targets]

    preds = test(args, model, device, test_loader, "Validation").cpu().numpy()[:,:num_targets,:]
    pydata = (*bnnc.uncertainty.analyze_predictions(preds, targets), preds)
    pyacc = bnnc.uncertainty.accuracy(pydata[0])

    preds = np.load("Code/predictions.npz")["arr_0"]
    cdata = (*bnnc.uncertainty.analyze_predictions(preds, targets), preds)
    cacc = bnnc.uncertainty.accuracy(cdata[0])

    match_ratio = bnnc.uncertainty.match_ratio(cdata[0], pydata[0])

    print(f"PY ACC {pyacc} -- C ACC {cacc} -- MATCH {match_ratio}")

    bnnc.plot.compare_predictions_plots(pydata, cdata, targets, "Figures")


    #s_input = np.array([32,32])
    #kernel = np.array([3,3])
    #stride = np.array([2,2])
    #output = np.ceil(s_input/stride)
    #p_min = np.clip(((output - 1) * stride - s_input + kernel).astype(int), 0, None)
    #print(output, p_min)

    #print(np.floor((s_input+2*p-kernel)/stride + 1))


def main_resnet():
    args = AppArgs()
    args.num_mc = args.num_mc_train

    device = cuda_dev()
    model = RESNET_TINY(args)
    model.to(device)

    train_data, train_loader, test_data , test_loader = get_data_color(args)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    for epoch in range(1, 100 + 1):
        train_clasic(args, model, device, train_loader, optimizer, epoch)
        test_clasic(args, model, device, test_loader, f"Epoch {epoch}")


    model.to("cpu")
    const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Flipout",  # Flipout or Reparameterization
        "moped_enable": True,  # True to initialize mu/sigma from the pretrained dnn weights
        "moped_delta": 0.5,
    }
    dnn_to_bnn(model, const_bnn_prior_parameters)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    for epoch in range(1, 100 + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader, f"Epoch {epoch}")

    torch.save(model.state_dict(), args.model_path())

if __name__ == "__main__":
    #main_train()
    #main_validate()
    #main_info()
    #main_pred_test()
    main_resnet()