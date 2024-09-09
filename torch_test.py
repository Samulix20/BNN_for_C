import torch
import torch.nn as nn
import torch.nn.functional as F

from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss
from bayesian_torch.layers.flipout_layers import conv_flipout as bnn_conv, linear_flipout as bnn_linear

import torchvision
from torchvision import transforms

import numpy as np

import bnnc
import bnnc.torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.relu2 = nn.ReLU()
        self.mp1 = nn.MaxPool2d(2)
        self.flat1 = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(16384, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        self.logsfm1 = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x

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

def create_model():
    const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Flipout",  # Flipout or Reparameterization
        "moped_enable": False,  # True to initialize mu/sigma from the pretrained dnn weights
        "moped_delta": 0.5,
    }
    model = Net()
    dnn_to_bnn(model, const_bnn_prior_parameters)
    return model

def load_model(args):
    model = create_model()
    model.load_state_dict(torch.load(args.model_path, weights_only=True))
    return model

class AppArgs:
    train_batch_size = 100
    num_mc_train = 1
    num_mc_validation = 100
    num_epochs = 3
    model_path = "Model/bnn_cifar10"
    learning_rate = 0.001

def main_train():
    args = AppArgs()
    args.num_mc = args.num_mc_train

    device = cuda_dev()
    model = create_model()
    model.to(device)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()  
    ])

    train_data = torchvision.datasets.CIFAR10('Data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.train_batch_size)

    test_data = torchvision.datasets.CIFAR10('Data', train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.num_epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader, f"Epoch {epoch}")

    torch.save(model.state_dict(), args.model_path)

def main_validate():
    args = AppArgs()
    args.num_mc = args.num_mc_validation

    device = cuda_dev()
    model = load_model(args)
    model.to(device)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()  
    ])

    test_data = torchvision.datasets.CIFAR10('Data', train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data))

    preds = test(args, model, device, test_loader, "Validation").cpu().numpy()
    targets = np.array(test_data.targets)
    metrics, averages = bnnc.uncertainty.analyze_predictions(preds, targets)
    #all_plots((metrics, averages, preds), targets, "Figures")

    bnnc.plot.compare_predictions_plots((metrics, averages, preds), (metrics, averages, preds), targets, "Figures")
    bnnc.plot.free_plot_memory()


def panic(msg="ERROR"):
    print(msg)
    exit(1)

def main_info():
    args = AppArgs()
    model = load_model(args)

    model_info = bnnc.torch.info_from_model(model, "bnn_model")
    model_info.calculate_buffers(np.array([32, 32, 1]))

    h, w = bnnc.code_gen.model_to_c(model_info)
    print(h)


if __name__ == "__main__":
    #main_train()
    #main_validate()
    main_info()
