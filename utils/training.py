import numpy as np
import pickle as pkl
import time
import os
import logging

logger = logging.getLogger(__name__)
from pathlib import Path

script_dir = Path(__file__).parent.absolute()

from utils.log import setup_comet_logger

import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from utils.datasets import ddp_setup, ParquetDataset
from utils.custom_models import (
    create_mixture_flow_model,
    save_model,
    load_model,
    load_fff_mixture_model
)
from utils.models import get_zuko_nsf
from utils.plots import sample_and_plot_base





class EarlyStopping:
    """
    Stops the training if the relative_loss is bigger than a threshold for a
    certain amount of times
    """
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
    print("training.py: train_base called")
    # device is device when not distributed and rank when distributed
    print("training.py: world size = ", world_size)
    if world_size is not None:
        ddp_setup(device, world_size)
    
    # usually device_id seems to be cuda
    device_id = device_ids[device] if device_ids is not None else device

    # create (and load) the model
    input_dim = len(cfg.target_variables)
    context_dim = len(cfg.context_variables)
    if cfg.model.name == "zuko_nsf":
        # Creates a zuko NSF flow as model
        model = get_zuko_nsf(
            input_dim=input_dim,
            context_dim=context_dim,
            ntransforms=cfg.model.ntransforms,
            nbins=cfg.model.nbins,
            nnodes=cfg.model.nnodes,
            nlayers=cfg.model.nlayers,
        )
    print("training.py: cfg.checkpoint = ", cfg.checkpoint)
    # in zuko0 cfg, cfg.checkpoint = None
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

    # Initialize the early_stopping
    early_stopping = EarlyStopping(
        patience=cfg.stopper.patience, min_delta=cfg.stopper.min_delta
    )  

    print("training.py: world size = ", world_size)
    # in zuko0 config world_size evaluates to None
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

    with open(f"{script_dir}/../preprocess/pipelines_{calo}.pkl", "rb") as file:
        pipelines = pkl.load(file)
        pipelines = pipelines[cfg.pipelines]
    
    # TODO: Check function ParquetDataset
    train_dataset = ParquetDataset(
        train_file,
        cfg.context_variables,
        cfg.target_variables,
        device=device,
        pipelines=pipelines,
        rows=cfg.train.size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False if world_size is not None else True,
        sampler=DistributedSampler(train_dataset) if world_size is not None else None,
        # num_workers=2,
        # pin_memory=True,
    )
    test_dataset = ParquetDataset(
        test_file,
        cfg.context_variables,
        cfg.target_variables,
        device=device,
        pipelines=train_dataset.pipelines,
        rows=cfg.test.size,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.test.batch_size,
        shuffle=False,
        # num_workers=2,
        # pin_memory=True,
    )

    # train the model
    # SummaryWriter from TensorBoard
    writer = SummaryWriter(log_dir="runs")
    comet_name = os.getcwd().split("/")[-1]
    comet_logger = setup_comet_logger(comet_name, cfg.model)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.optimizer.learning_rate,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        weight_decay=cfg.optimizer.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs)

    train_history, test_history = [], []
    for epoch in range(start_epoch, cfg.epochs + 1):
        if world_size is not None:
            b_sz = len(next(iter(train_loader))[0])
            logger.info(
                f"[GPU{device_id}] | Rank {device} | Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(train_loader)}"
            )
            train_loader.sampler.set_epoch(epoch)
        logger.info(f"Epoch {epoch}/{cfg.epochs}:")

        train_losses, test_losses = [], []
        #train
        # time.time() from torch
        start = time.time()
        logger.info("Training...")
        for context, target, weights, _ in train_loader:
            # context, target = context.to(device), target.to(device)
            model.train()
            optimizer.zero_grad()

            if "zuko" in cfg.model.name:
                loss = -ddp_model(context).log_prob(target)
                loss = weights * loss
            
            loss = loss.mean()
            train_losses.append(loss.item())

            loss.backward()
            optimizer.step()
            scheduler.step()
        
        epoch_train_loss = np.mean(train_losses)
        train_history.append(epoch_train_loss)

        # test
        logger.info("Testing...")
        for context, target, weights, _ in test_loader:
            with torch.no_grad():
                model.eval()
                if "zuko" in cfg.model.name:
                    loss = -ddp_model(context).log_prob(target)
                    loss = weights * loss
                loss = loss.mean()
                test_losses.append(loss.item())
        epoch_test_loss = np.mean(test_losses)
        test_history.append(epoch_test_loss)
        if device == 0 or world_size is None:
            writer.add_scalars(
                "Losses", {"train": epoch_train_loss, "val": epoch_test_loss}, epoch
            )
            comet_logger.log_metrics(
                {"train": epoch_train_loss, "val": epoch_test_loss}, step=epoch
            )
        
        # sample and validation
        # TODO: check sample_and_plot function
        if epoch % cfg.sample_every == 0 or epoch == 1:
            logger.info("Sampling and plotting...")
            sample_and_plot_base(
                test_loader=test_loader,
                model=model,
                model_name=cfg.model.name,
                epoch=epoch,
                writer=writer,
                comet_logger=comet_logger,
                context_variables=cfg.context_variables,
                target_variables=cfg.target_variables,
                device=device,
                pipeline=cfg.pipelines,
                calo=calo,
            )

        duration = time.time() - start
        logger.info(
            f"Epoch {epoch} | GPU{device_id} | Rank {device} - train loss: {epoch_train_loss:.4f} - val loss: {epoch_test_loss:.4f} - time: {duration:.2f}s"
        )
        if device == 0 or world_size is None:
            save_model(
                epoch,
                ddp_model,
                scheduler,
                train_history,
                test_history,
                name="checkpoint-latest.pt",
                model_dir=".",
                optimizer=optimizer,
                is_ddp=world_size is not None,
            )

        if epoch_train_loss < best_train_loss:
            logger.info("New best train loss, saving model...")
            best_train_loss = epoch_train_loss
            if device == 0 or world_size is None:
                save_model(
                    epoch,
                    ddp_model,
                    scheduler,
                    train_history,
                    test_history,
                    name="best_train_loss.pt",
                    model_dir=".",
                    optimizer=optimizer,
                    is_ddp=world_size is not None,
                )

        early_stopping(epoch_train_loss)
        if early_stopping.early_stop:
            break

    writer.close()
    

