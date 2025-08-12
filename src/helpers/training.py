import os

import yaml
import numpy as np
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


#======Data Loaders=========================================================
NUM_WORKERS = 4

def get_loader(
        dataset: Dataset,
        shuffle: bool=False,
        batch_size: int=1, 
        num_workers: int=NUM_WORKERS, 
        pin_memory: bool=True,
    ):

    loader = DataLoader(
        dataset=dataset, 
        shuffle=shuffle, 
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    n_iters = int(np.ceil(len(dataset) / batch_size))

    return loader, n_iters



#======Optimizer & Scheduler=========================================================
OPTIMIZERS = {
    'SGD':      optim.SGD,
    'ADAM':     optim.Adam,
    'ADAMW':    optim.AdamW,
}

SCHEDULERS = {
    'STEPLR':   optim.lr_scheduler.StepLR,
}


def get_optimizer(params, optimizer, base_lr, base_batch_size, batch_size=None, **kwargs):
    batch_size = batch_size or base_batch_size
    lr = base_lr*batch_size/base_batch_size
    assert optimizer.upper() in OPTIMIZERS, "optimizer is not correct"
    return OPTIMIZERS[optimizer.upper()](params, lr=lr, **kwargs)


def get_scheduler(optimizer, scheduler, **kwargs):
    assert scheduler.upper() in SCHEDULERS, "scheduler is not correct"
    return SCHEDULERS[scheduler.upper()](optimizer=optimizer, **kwargs)


def get_optimizer_scheduler(params, cfg_optimizer, cfg_scheduler):
    optimizer = get_optimizer(params, **cfg_optimizer)
    scheduler = get_scheduler(optimizer, **cfg_scheduler)
    return optimizer, scheduler


#======Accuracy=========================================================
def calculate_accuracy(logits, labels):
    """
    Calculate accuracy from logits and labels.
    
    Args:
        logits (torch.Tensor): The model's output logits of shape [b, c].
        labels (torch.Tensor): The ground-truth labels of shape [b].
        
    Returns:
        float: The accuracy as a percentage.
    """
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == labels).sum().item()
    accuracy = (correct / labels.size(0)) * 100
    return accuracy


#======Count paramters=========================================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


#======Read and save config file=========================================================
def read_config(path):
    with open(str(path)+".yaml", 'r') as f:
        return yaml.load(f, yaml.FullLoader)


def save_config(cfg, path):
    with open(str(path)+".yaml", 'w') as f:
        return yaml.dump(cfg, f)


#======Get list of models=========================================================
def get_models_list(
        dir: str, 
        prefix: str,
    ):
    models = [epoch for epoch in os.listdir(dir) if epoch.startswith(prefix)]
    models_int = sorted([int(epoch[len(prefix):-3]) for epoch in models])
    return [prefix+str(epoch)+'.pt' for epoch in models_int]