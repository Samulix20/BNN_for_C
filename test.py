from models import *
from train import *
from c_run import *

import torchvision
from torchvision import transforms

import numpy as np

import bnnc
import bnnc.torch


class TrainParams:
    lr = 0.001
    weight_decay = 0.0001
    num_epochs = 500
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
        return torch.optim.Adam(model.parameters(), lr=TrainParams.lr, weight_decay=TrainParams.weight_decay)

def get_device(use_cuda: bool = True):
    use_cuda = torch.cuda.is_available() and use_cuda
    if use_cuda:
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def get_data(gray: bool = False):
    if gray:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()  
        ])
    else:
        transform = transforms.ToTensor()

    train_data = torchvision.datasets.CIFAR10('Data', train=True, download=True, transform=transform)
    test_data = torchvision.datasets.CIFAR10('Data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=TrainParams.batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data))
    return (train_data, train_loader, test_data, test_loader)

def main_train():
    device = get_device()
    model = RESNET_TINY()
    dnn_to_bnn(model, TrainParams.bnn_prior_parameters)
    model.to(device)

    print(model)

    optimizer = get_optimizer(model)
    train_data, train_loader, test_data , test_loader = get_data()
    logger = ModelTrainLogger("Model", "BnnResnetTiny")

    for epoch in range(1, TrainParams.num_epochs + 1):
        bayesian_train(model, train_loader, device, optimizer, 1, logger)
        bayesian_test(model, test_loader, device, 100, logger)
        if logger.is_overfitting():
            break
        logger.next_epoch()
    
    logger.save_model(model)
    logger.info()

def main_ld():
    device = get_device()
    model = RESNET_TINY()
    dnn_to_bnn(model, TrainParams.bnn_prior_parameters)
    #model.load_state_dict(torch.load('Model/best_BnnResnetTiny', weights_only=True))

    train_data, train_loader, test_data , test_loader = get_data()

    print(model)

    model_info = bnnc.torch.info_from_model(model, "bnn_model")
    for data, targets in test_loader:
        pass
    targets = targets[:10]
    data = data.permute((0,2,3,1))
    input_shape = np.array(data[0].shape)
    flat_data = data.numpy().reshape((data.shape[0], -1))

    model_info.calculate_buffers(input_shape)
    model_info.print_buffer_info()

    l, h, w = model_info.create_c_code()
    print(h)

    #bayesian_test(model, test_loader, device, 100)

def main_c():
    num_workers = 20
    max_img_per_worker = 10
    num_targets = num_workers * max_img_per_worker
    
    device = get_device()
    model = B2N2()
    dnn_to_bnn(model, TrainParams.bnn_prior_parameters)
    model.load_state_dict(torch.load('Model/best_B2N2', weights_only=True))

    train_data, train_loader, test_data, test_loader = get_data(gray=True)

    for data, targets in test_loader:
        pass
    targets = targets[:num_targets]
    data = data.permute((0,2,3,1))
    input_shape = np.array(data[0].shape)
    flat_data = data.numpy().reshape((data.shape[0], -1))

    model_info = bnnc.torch.info_from_model(model, "bnn_model")
    model_info.calculate_buffers(input_shape)
    model_info.uniform_weight_transform()
    run_c_model(model_info, flat_data, num_workers, max_img_per_worker)

    model.to(device)
    preds = bayesian_test(model, test_loader, device, 100).cpu().numpy()[:,:num_targets,:]
    pydata = (*bnnc.uncertainty.analyze_predictions(preds, targets), preds)
    pyacc = bnnc.uncertainty.accuracy(pydata[0])

    preds = np.load("Code/predictions.npz")["arr_0"]
    cdata = (*bnnc.uncertainty.analyze_predictions(preds, targets), preds)
    cacc = bnnc.uncertainty.accuracy(cdata[0])

    match_ratio = bnnc.uncertainty.match_ratio(cdata[0], pydata[0])

    print(f"PY ACC {pyacc} -- C ACC {cacc} -- MATCH {match_ratio}")

    bnnc.plot.compare_predictions_plots(pydata, cdata, targets, "Figures")

if __name__ == "__main__":
    main_ld()
    #main_c()
    #main_train()
