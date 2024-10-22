from models import *
from train import *
from c_run import *

import torchvision
from torchvision import transforms

import numpy as np

import bnnc
import bnnc.torch

class TestConfig:
    use_cuda = True
    model = "B2N2"
    lr = 0.001
    weight_decay = 0.0001
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
    num_workers = 20
    max_img_per_worker = 5000

    def figure_dir():
        d = f"Figures/{TestConfig.model}"
        os.system(f"mkdir -p {d}")
        return d

    def get_model(bnn=False):
        if TestConfig.model == "RESNET":
            m = RESNET_TINY()
        elif TestConfig.model == "B2N2":
            m = B2N2()
        elif TestConfig.model == "LENET":
            m = LENET()

        if bnn:
            dnn_to_bnn(m, TestConfig.bnn_prior_parameters)

        return m

    def get_device():
        use_cuda = torch.cuda.is_available() and TestConfig.use_cuda
        if use_cuda:
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def get_optimizer(model: nn.Module):
        return torch.optim.Adam(model.parameters(), lr=TestConfig.lr, weight_decay=TestConfig.weight_decay)

    def get_data():
        gray = not TestConfig.model == "RESNET"
        
        if gray:
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor()  
            ])
        else:
            transform = transforms.ToTensor()

        train_data = torchvision.datasets.CIFAR10('Data', train=True, download=True, transform=transform)
        test_data = torchvision.datasets.CIFAR10('Data', train=False, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=TestConfig.batch_size)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data))
        return (train_loader, test_loader)
    
    def get_logger(bnn=False):
        t = TestConfig.model
        if bnn:
            t = "bnn_" + t
        return ModelTrainLogger("Model", t)

    def get_all(bnn=False, moped=False, load=False):
        if moped:
            m = TestConfig.get_model()
            l = TestConfig.get_logger()
            l.load_best_model(m)
            dnn_to_bnn(m, TestConfig.bnn_prior_parameters)
            l = TestConfig.get_logger(bnn=True)
        else:
            m = TestConfig.get_model(bnn=bnn)
            l = TestConfig.get_logger(bnn=bnn)
        
        if load:
            l.load_best_model(m)

        return (TestConfig.get_device(), m, TestConfig.get_optimizer(m), l, TestConfig.get_data())


def main_train():
    device, model, optimizer, logger, (train_loader, test_loader) = TestConfig.get_all()

    model.to(device)
    for epoch in range(1, TestConfig.num_epochs + 1):
        train(model, train_loader, device, optimizer, logger)
        test(model, train_loader, device, logger)
        if logger.is_overfitting():
            break
        logger.next_epoch()

    device, model, optimizer, logger, (train_loader, test_loader) = TestConfig.get_all(moped=True)

    model.to(device)
    for epoch in range(1, TestConfig.num_epochs + 1):
        bayesian_train(model, train_loader, device, optimizer, 1, logger)
        bayesian_test(model, test_loader, device, 100, logger)
        if logger.is_overfitting():
            break
        logger.next_epoch()
    
    logger.save_model(model)
    logger.info()

def main_ld():
    device, model, optimizer, logger, (train_loader, test_loader) = TestConfig.get_all(bnn=True, load=True)
    model.to(device)
    bayesian_test(model, test_loader, device, 100)

    model.to("cpu")

    for data, targets in test_loader:
        pass
    data = data.permute((0,2,3,1))
    input_shape = np.array(data[0].shape)
    flat_data = data.numpy().reshape((data.shape[0], -1))

    model_info = bnnc.torch.info_from_model(model, "bnn_model")
    model_info.calculate_buffers(input_shape)
    model_info.print_buffer_info()

def main_c():
    num_workers = TestConfig.num_workers
    max_img_per_worker = TestConfig.max_img_per_worker
    num_targets = num_workers * max_img_per_worker

    device, model, optimizer, logger, (train_loader, test_loader) = TestConfig.get_all(bnn=True, load=True)

    for data, targets in test_loader:
        pass
    targets = targets[:num_targets]
    data = data.permute((0,2,3,1))
    input_shape = np.array(data[0].shape)
    flat_data = data.numpy().reshape((data.shape[0], -1))

    model.to(device)
    preds = bayesian_test(model, test_loader, device, 100).cpu().numpy()[:,:num_targets,:]
    pydata = (*bnnc.uncertainty.analyze_predictions(preds, targets), preds)
    pyacc = bnnc.uncertainty.accuracy(pydata[0])

    model.to("cpu")
    model_info = bnnc.torch.info_from_model(model, "bnn_model")
    model_info.calculate_buffers(input_shape)
    model_info.uniform_weight_transform()
    run_c_model(model_info, flat_data, num_workers, max_img_per_worker)

    preds = np.load("Code/predictions.npz")["arr_0"]
    cdata = (*bnnc.uncertainty.analyze_predictions(preds, targets), preds)
    cacc = bnnc.uncertainty.accuracy(cdata[0])

    match_ratio = bnnc.uncertainty.match_ratio(cdata[0], pydata[0])

    print(f"PY ACC {pyacc} -- C ACC {cacc} -- MATCH {match_ratio}")


    bnnc.plot.compare_predictions_plots(pydata, cdata, targets, TestConfig.figure_dir())

if __name__ == "__main__":
    TestConfig.model = "LENET"
    main_ld()
    main_c()
    TestConfig.model = "B2N2"
    main_ld()
    main_c()
    TestConfig.model = "RESNET"
    main_ld()
    main_c()
