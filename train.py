import torch
import torch.nn as nn
import torch.nn.functional as F

from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss

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
