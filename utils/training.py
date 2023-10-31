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
from utils.custom_models import (
    create_mixture_flow_model,
    save_model,
    load_model,
    load_fff_mixture_model
)
from utils.models import get_zuko_nsf





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
    input_dim = len(cfg.target_variables)
    context_dim = len(cfg.context_variables)
    if cfg.model.name == "zuko_nsf":
        model = get_zuko_nsf(
            input_dim=input_dim,
            context_dim=context_dim,
            ntransforms=cfg.model.ntransforms,
            nbins=cfg.model.nbins,
            nnodes=cfg.model.nnodes,
            nlayers=cfg.model.nlayers,
        )
    
    if cfg.checkpoint is not None:
        # assume that the checkpoint is a path to a directory
        model, _, _, start_epoch, th, _ = load_model(
            model, model_dir=cfg.checkpoint, filename="checkpoint-latest.pt"
        )
        model = model.to(device)
        best_train_loss = np.min(th)
        logger.info("Loaded model from checkpoint: {}".format(cfg.checkpoint))
        logger.info("Resuming from epoch {}".format(start_epoch))
        logger.info("Best train loss found to be: {}".format(best_train_loss))
    else:
        start_epoch = 1
        best_train_loss = 10000000

    model = model.to(device)      

    early_stopping = EarlyStopping(
        patience=cfg.stopper.patience, min_delta=cfg.stopper.min_delta
    )  

    if world_size is not None:
        ddp_model = DDP(
            model,
            device_ids=[device],
            output_device=device,
        )
        model = ddp_model.module
    else:
        ddp_model = model
        logger.info(
        "Number of parameters: {}".format(sum(p.numel() for p in model.parameters()))
        )

    # make datasets
    sample = cfg.sample
    calo = cfg.calo

    if sample == "data":
        if calo == "eb":
            train_file = f"{script_dir}/../preprocess/data_eb_train.parquet"
            test_file = f"{script_dir}/../preprocess/data_eb_test.parquet"
        elif calo == "ee":
            train_file = f"{script_dir}/../preprocess/data_ee_train.parquet"
            test_file = f"{script_dir}/../preprocess/data_ee_test.parquet"
    elif sample == "mc":
        if calo == "eb":
            train_file = f"{script_dir}/../preprocess/mc_eb_train.parquet"
            test_file = f"{script_dir}/../preprocess/mc_eb_test.parquet"
        elif calo == "ee":
            train_file = f"{script_dir}/../preprocess/mc_ee_train.parquet"
            test_file = f"{script_dir}/../preprocess/mc_ee_test.parquet"

    with open(f"{script_dir}/../preprocess/piplines_{calo}.pkl", "rb") as file:
        pipelines = pkl.load(file)
        pipelines = pipelines[cfg.pipelines]
