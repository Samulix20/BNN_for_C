import torchvision
from torchvision import transforms

import numpy as np

import bnnc
import bnnc.torch

from models import *
from train import *
from c_run import *

class TestConfig:
    model = "RESNET"

    # Try Use GPU
    use_cuda = True

    # Training parameters
    lr = 0.001
    weight_decay = 0.0001
    num_epochs = 200
    batch_size = 128

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

    # C config
    num_workers = 20
    max_img_per_worker = 500
    fixed_bits = 10
    generation_method = "Uniform"

    def test_id():
        return f"{TestConfig.model}-{TestConfig.fixed_bits}-{TestConfig.generation_method}"

    def figure_dir():
        d = f"Figures/{TestConfig.test_id()}"
        os.system(f"mkdir -p {d}")
        return d

    def prediction_dir():
        d = f"Predictions/{TestConfig.test_id()}"
        os.system(f"mkdir -p {d}")
        return d

    def get_model(bnn=False):

        def nll_loss_sum(x, target):
            return F.nll_loss(torch.log(x), target, reduction="sum")

        def mse_loss_sum(x, target):
            return F.mse_loss(x, target, reduction="sum")

        lf = nll_loss_sum

        if TestConfig.model == "RESNET":
            m = RESNET_TINY()
        elif TestConfig.model == "B2N2":
            m = B2N2()
        elif TestConfig.model == "LENET":
            m = LENET()

        if bnn:
            dnn_to_bnn(m, TestConfig.bnn_prior_parameters)

        return m, lf

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
            m, lf = TestConfig.get_model()
            l = TestConfig.get_logger()
            l.load_best_model(m)
            dnn_to_bnn(m, TestConfig.bnn_prior_parameters)
            l = TestConfig.get_logger(bnn=True)
        else:
            m, lf = TestConfig.get_model(bnn=bnn)
            l = TestConfig.get_logger(bnn=bnn)
        
        if load:
            l.load_best_model(m)

        return (TestConfig.get_device(), m, TestConfig.get_optimizer(m), lf, l, TestConfig.get_data())

    def apply_weight_transform(model_info: bnnc.model_info.ModelInfo):
        if TestConfig.generation_method == "Uniform":
            model_info.uniform_weight_transform()
        elif TestConfig.generation_method == "Bernoulli":
            model_info.bernoulli_weight_transform()
        elif TestConfig.generation_method == "Normal":
            return

    def train_model():
        print(f"Model: {TestConfig.model}")
        device, model, optimizer, loss, logger, (train_loader, test_loader) = TestConfig.get_all()

        model.to(device)
        for epoch in range(1, TestConfig.num_epochs + 1):
            train(model, train_loader, device, optimizer, loss, logger)
            test(model, train_loader, device, loss, logger)
            if logger.is_overfitting():
                break
            logger.next_epoch()

        device, model, optimizer, loss, logger, (train_loader, test_loader) = TestConfig.get_all(moped=True)

        model.to(device)
        for epoch in range(1, TestConfig.num_epochs + 1):
            bayesian_train(model, train_loader, device, optimizer, loss, 1, logger)
            bayesian_test(model, test_loader, device, loss, 100, logger)
            if logger.is_overfitting():
                break
            logger.next_epoch()

        logger.save_model(model)
        logger.info()

    def test_model(run_c = False):
        print(f"Model: {TestConfig.model}")

        num_workers = TestConfig.num_workers
        max_img_per_worker = TestConfig.max_img_per_worker
        num_targets = num_workers * max_img_per_worker

        if not run_c:
            num_targets = -1

        device, model, optimizer, loss, logger, (train_loader, test_loader) = TestConfig.get_all(bnn=True, load=True)

        for data, targets in test_loader:
            pass
        targets = targets[:num_targets]
        data = data.permute((0,2,3,1))
        input_shape = np.array(data[0].shape)
        flat_data = data.numpy().reshape((data.shape[0], -1))

        model.to(device)
        preds = bayesian_test(model, test_loader, device, loss, 100).cpu().numpy()[:,:num_targets,:]
        pydata = (*bnnc.uncertainty.analyze_predictions(preds, targets), preds)
        pyacc = bnnc.uncertainty.accuracy(pydata[0])
        pyece, pyuce = bnnc.uncertainty.calibration_errors(pydata[0])

        if not run_c:
            print(f"ACC {pyacc:.5} ECE {pyece:.5} UCE {pyuce:.5}")
            return

        model.to("cpu")
        model_info = bnnc.torch.info_from_model(model, "bnn_model")
        model_info.calculate_buffers(input_shape)
        model_info.print_buffer_info()
        TestConfig.apply_weight_transform(model_info)
        model_info.fixed_bits = TestConfig.fixed_bits
        run_c_model(model_info, flat_data, num_workers, max_img_per_worker)

        preds = np.load("Code/predictions.npz")["arr_0"]
        os.system(f"mv Code/predictions.npz {TestConfig.prediction_dir()}")

        cdata = (*bnnc.uncertainty.analyze_predictions(preds, targets), preds)
        cacc = bnnc.uncertainty.accuracy(cdata[0])
        cece, cuce = bnnc.uncertainty.calibration_errors(cdata[0])

        match_ratio, c_conf, py_conf, c_u, py_u = bnnc.uncertainty.match_ratio(cdata[0], pydata[0])

        print(f"ACC {pyacc:.5} ECE {pyece:.5} UCE {pyuce:.5} -- Python")
        print(f"ACC {cacc:.5} ECE {cece:.5} UCE {cuce:.5} -- C")
        print(f"Matching preditions {match_ratio:.5}")
        print(f"Mismatch Average Confidence C {c_conf:.5} Python {py_conf:.5}")
        print(f"Mismatch Average Uncertainty C {c_u:.5} Python {py_u:.5}")

        bnnc.plot.compare_predictions_plots(pydata, cdata, targets, TestConfig.figure_dir())


def train_all():
    for model in ["LENET", "B2N2", "RESNET"]:
        TestConfig.model = model
        #TestConfig.train_model()
        TestConfig.test_model()

if __name__ == "__main__":
    train_all()
    pass