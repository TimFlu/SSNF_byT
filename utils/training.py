import numpy as np
import pickle as pkl
import time
import os
import logging

logger = logging.getLogger(__name__)
from pathlib import Path

script_dir = Path(__file__).parent.absolute()
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from utils.datasets import ddp_setup












class EarlyStopping:
    def __init__(self, patience=5, min_delta=1):
        self.patience = patience
        self.min_delta = min_delta
        self.couonter = 0
        self.best_loss = 10e10
        self.early_stop = False

    def __call__(self, val_loss):
        relative_loss = (self.best_loss - val_loss) / self.best_loss * 100
        if relative_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif relative_loss < self.min_delta:
            self.counter += 1
            logger.info(
                f"Earky stopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                logger.info("Early stopping")
                self.early_stop = True


def train_base(device, cfg, world_size = None, device_ids=None):
    # device is device when not distributed and rank when distributed
    if world_size is not None:
        ddp_setup(device, world_size)
    
    device_id = device_ids[device] if device_ids is not None else device

    # create (and load) the model