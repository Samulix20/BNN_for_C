from models import *
from train import *

import torchvision
from torchvision import transforms

from torchinfo import summary

class TrainParams:
    lr = 0.001
    num_epochs = 200
    batch_size = 128
    bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Flipout",  # Flipout or Reparameterization
        "moped_enable": True,  # True to initialize mu/sigma from the pretrained dnn weights
        "moped_delta": 0.5,
    }


def get_optimizer(model: nn.Module):
        return torch.optim.Adam(model.parameters(), lr=TrainParams.lr)

def get_device(use_cuda: bool = True):
    use_cuda = torch.cuda.is_available() and use_cuda
    if use_cuda:
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def get_data():
    transform = transforms.ToTensor()
    train_data = torchvision.datasets.CIFAR10('Data', train=True, download=True, transform=transform)
    test_data = torchvision.datasets.CIFAR10('Data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=TrainParams.batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data))
    return (train_data, train_loader, test_data, test_loader)

def main_resnet():
    device = get_device()

    model = RESNET_TINY()
    model.to(device)

    optimizer = get_optimizer(model)

    train_data, train_loader, test_data , test_loader = get_data()
    logger = ModelTrainLogger("Model", "ResnetTiny")

    for epoch in range(1, TrainParams.num_epochs + 1):
        train(model, train_loader, device, optimizer, logger)
        test(model, test_loader, device, logger)
        if logger.is_overfitting():
            break

    logger.save_model(model)
    logger.info()

    print("Bayesian Training")

    model.to("cpu")
    model.load_state_dict(torch.load('Model/best_ResnetTiny', weights_only=False))
    dnn_to_bnn(model, TrainParams.bnn_prior_parameters)
    model.to(device)

    optimizer = get_optimizer(model)
    train_data, train_loader, test_data , test_loader = get_data()
    logger = ModelTrainLogger("Model", "BnnResnetTiny")

    for epoch in range(1, TrainParams.num_epochs + 1):
        bayesian_train(model, train_loader, device, optimizer, 1, logger)
        bayesian_test(model, test_loader, device, 100, logger)
        if logger.is_overfitting():
            break
    
    logger.save_model(model)
    logger.info()


def main_ld():
    device = get_device()
    model = RESNET_TINY()
    dnn_to_bnn(model, TrainParams.bnn_prior_parameters)
    model.load_state_dict(torch.load('Model/best_BnnResnetTiny', weights_only=True))
    model.to(device)

    train_data, train_loader, test_data , test_loader = get_data()

    bayesian_test(model, test_loader, device, 100)


if __name__ == "__main__":
    main_resnet()
    main_ld()
