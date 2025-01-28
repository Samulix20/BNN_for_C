import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from bayesian_torch.models.dnn_to_bnn import get_kl_loss

from math import isnan

class ModelTrainLogger:
    def __init__(self, folder: str, name: str):

        self.best_train = None
        self.best_train_epoch = None

        self.best_test = None
        self.best_test_epoch = None

        self.epoch = 1
        self.test_worse = 0

        self.folder = folder
        self.name = name

        self.nan_found = False

    def next_epoch(self):
        self.epoch += 1

    def save_model(self, model: nn.Module):
        torch.save(model.state_dict(), f"{self.folder}/{self.name}_{self.epoch - 1}")

    def save_best_model(self, model: nn.Module):
        torch.save(model.state_dict(), f"{self.folder}/best_{self.name}")

    def load_best_model(self, model: nn.Module):
        model.load_state_dict(torch.load(f"{self.folder}/best_{self.name}", weights_only=True))

    def log_train(self, loss, acc):
        if isnan(loss):
            self.nan_found = True

        if self.best_train is None or loss < self.best_train:
            self.best_train = loss
            self.best_train_epoch = self.epoch
        print(f'+ Epoch {self.epoch} Train: Average loss: {loss:.4f}, Accuracy: {100. * acc:.2f}%')

    def log_test(self, model: nn.Module, loss, acc):
        if self.best_test is None or loss < self.best_test:
            self.best_test = loss
            self.best_test_epoch = self.epoch
            self.test_worse = 0
            self.save_best_model(model)
        else:
            self.test_worse += 1

        print(f'- Epoch {self.epoch} Test: Average loss: {loss:.4f}, Accuracy: {100. * acc:.2f}%')

    def info(self):
        print(f"Train Epoch {self.best_train_epoch} loss: {self.best_train:4f}")
        print(f"Test Epoch {self.best_test_epoch} loss: {self.best_test:4f}")

    def is_overfitting(self, threshold: int = 5):
        return self.test_worse > threshold or self.nan_found

    def log_msg(loss, acc):
        print(f'Test: Average loss: {loss:.4f}, Accuracy: {100. * acc:.2f}%')


def get_batch_correct(model_output, target):
    pred = model_output.argmax(dim=1, keepdim=True)
    return pred.eq(target.view_as(pred)).sum().item()

# Training utils

def train(model: nn.Module, loader, device, optimizer, loss_func, logger: ModelTrainLogger = None):
    model.train()
    train_loss = 0
    train_correct = 0

    for data, target in loader:
        optimizer.zero_grad()
        
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = loss_func(output, target)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += get_batch_correct(output, target)

    train_loss /= len(loader.dataset)
    train_acc = train_correct / len(loader.dataset)
    logger.log_train(train_loss, train_acc)

def bayesian_train(model: nn.Module, loader, device, optimizer, loss_func, num_mc, logger: ModelTrainLogger = None):
    model.train()
    train_loss = 0
    train_correct = 0

    for data, target, in loader:
        optimizer.zero_grad()

        data, target = data.to(device), target.to(device)
        batch_size = data.shape[0]
        
        output_list = []
        kl_list = []
        for mc_run in range(num_mc):
            output_list.append(model(data))
            kl_list.append(get_kl_loss(model))
        output = torch.mean(torch.stack(output_list), dim=0)
        kl = torch.mean(torch.stack(kl_list), dim=0)
        loss = loss_func(output, target) + (kl / batch_size)
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += get_batch_correct(output, target)

    train_loss /= len(loader.dataset)
    train_acc = train_correct / len(loader.dataset)
    logger.log_train(train_loss, train_acc)


def test(model: nn.Module, loader, device, loss_func, logger: ModelTrainLogger = None):
    model.eval()
    test_loss = 0
    test_correct = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_func(output, target)

            test_correct += get_batch_correct(output, target)
            test_loss += loss.item()
    
    test_loss /= len(loader.dataset)
    test_acc = test_correct / len(loader.dataset)

    if logger is not None:
        logger.log_test(model, test_loss, test_acc)
    else:
        ModelTrainLogger.log_msg(test_loss, test_acc)
    
    return 

def bayesian_test(model: nn.Module, loader, device, loss_func, num_mc, logger: ModelTrainLogger = None):
    model.eval()
    test_loss = 0
    test_correct = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            batch_size = data.shape[0]

            output_list = []
            kl_list = []
            for mc_run in range(num_mc):
                output_list.append(model(data))
                kl_list.append(get_kl_loss(model))
            raw_output = torch.stack(output_list)
            output = torch.mean(raw_output, dim=0)
            kl = torch.mean(torch.stack(kl_list), dim=0)
            loss = loss_func(output, target) + (kl / batch_size)

            test_correct += get_batch_correct(output, target)
            test_loss += loss.item()
    
    test_loss /= len(loader.dataset)
    test_acc = test_correct / len(loader.dataset)

    if logger is not None:
        logger.log_test(model, test_loss, test_acc)
    else:
        ModelTrainLogger.log_msg(test_loss, test_acc)

    return raw_output


import testconf

class TrainParams:
    # Training parameters
    lr = 0.001
    weight_decay = 0.0001
    num_epochs = 1
    batch_size = 128


def train_model(modelname:str):
    print(f"Model: {modelname}")

    device = testconf.get_device()
    train_data, test_data = testconf.get_data(modelname)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=TrainParams.batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data))

    model, loss = testconf.get_model(modelname, "untrained")
    optimizer = torch.optim.Adam(model.parameters(), lr=TrainParams.lr, weight_decay=TrainParams.weight_decay)
    logger = ModelTrainLogger("Model", modelname)

    # MOPED training framework

    model.to(device)
    for epoch in range(1, TrainParams.num_epochs + 1):
        train(model, train_loader, device, optimizer, loss, logger)
        test(model, train_loader, device, loss, logger)
        if logger.is_overfitting():
            break
        logger.next_epoch()

    model, loss = testconf.get_model(modelname, "trained")
    optimizer = torch.optim.Adam(model.parameters(), lr=TrainParams.lr, weight_decay=TrainParams.weight_decay)
    logger = ModelTrainLogger("Model", f"bnn_{modelname}")

    model.to(device)
    for epoch in range(1, TrainParams.num_epochs + 1):
        bayesian_train(model, train_loader, device, optimizer, loss, 1, logger)
        bayesian_test(model, test_loader, device, loss, 100, logger)
        if logger.is_overfitting():
            break
        logger.next_epoch()

    logger.info()

def test_model(modelname:str):
    print(f"Model: {modelname}")

    device = testconf.get_device()
    _, test_data = testconf.get_data(modelname)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data))

    model, loss = testconf.get_model(modelname, "bnn")
    model.to(device)
    preds = bayesian_test(model, test_loader, device, loss, 100).cpu().numpy()
    np.savez(testconf.baseline_path(modelname), preds)

def __fu():
    testconf.init_folders()
    
    for model in testconf.Conf.model_list:
        #train_model(model)
        #test_model(model)
        pass

    for model in testconf.Conf.hyper_model_list:
        test_model(model)

import bnnc

def new_bayesian_test(model: nn.Module, loader, device):
    model.to(device)
    model.eval()
    test_loss = 0
    test_correct = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            raw_output = model(data)
            output = torch.mean(raw_output, dim=0)
            test_correct += get_batch_correct(output, target)
    
    test_acc = test_correct / len(loader.dataset)
    ModelTrainLogger.log_msg(0, test_acc)

    return raw_output


if __name__ == "__main__":

    for modelname in testconf.Conf.hyper_model_list:

        print(modelname)

        device = testconf.get_device()
        _, test_data = testconf.get_data(modelname)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data))

        old_model, loss = testconf.get_model(modelname, "bnn")
        old_model.to(device)
        print("OLD")
        oro = bayesian_test(old_model, test_loader, device, loss, 100).cpu().numpy()

        for data, targets in test_loader:
            pass
        targets = targets.numpy()

        aop = bnnc.metrics.analyze_predictions(oro, targets)
        print(aop["ece"], aop["uce"], aop["re"])

        model = testconf.models.NEW_HYPER(old_model, 100)
        print("NEW")
        nro = new_bayesian_test(model, test_loader, device).cpu().numpy()
        anp = bnnc.metrics.analyze_predictions(nro, targets)
        print(anp["ece"], anp["uce"], anp["re"])
    
    modelname = "LENET"

    print(modelname)

    device = testconf.get_device()
    _, test_data = testconf.get_data(modelname)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data))

    old_model, loss = testconf.get_model(modelname, "bnn")
    old_model.to(device)
    print("OLD")
    oro = bayesian_test(old_model, test_loader, device, loss, 100).cpu().numpy()

    for data, targets in test_loader:
        pass
    targets = targets.numpy()

    aop = bnnc.metrics.analyze_predictions(oro, targets)
    print(aop["ece"], aop["uce"], aop["re"])

    model = testconf.models.NEW_LENET(old_model, 100)
    print("NEW")
    _, test_data = testconf.get_data(modelname)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=500)
    nro = new_bayesian_test(model, test_loader, device).cpu().numpy()
    anp = bnnc.metrics.analyze_predictions(nro, targets)
    print(anp["ece"], anp["uce"], anp["re"])
