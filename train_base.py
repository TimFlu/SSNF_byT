import hydra
from hydra.utils import get_original_cwd
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import logging
logger = logging.getLogger(__name__)


from utils.training import train_base


@hydra.main(version_base=None, config_path="config_base", config_name="cfg_test")
def main(cfg):

    # In hydra config we enably the feature that changes de cwd to the experiment dir
    initial_dir = get_original_cwd()
    logger.debug("Initial dir: ", initial_dir)
    logger.debug("Current dir: ", os.getcwd())

    # save the config
    cfg_name = HydraConfig.get().job.name    
    with open(f"{os.getcwd()}/{cfg_name}.yaml", "w") as file:
        OmegaConf.save(config=cfg, f=file)
    
    env_var = os.environ.get("CUDA_VISIBLE_DEVICES")

    if env_var:
        actual_devices = env_var.split(",")
        actual_devices = [int(d) for d in actual_devices]
    else:
        actual_devices = list(range(torch.cuda.device_count()))

    logger.debug("Actual devices: ", actual_devices)
    
    logger.info("Training with cfg: \n".format(OmegaConf.to_yaml(cfg)))
    print("train_base: cfg.distributed = ", cfg.distributed)
    if cfg.distributed:
        world_size = torch.cuda.device_count()
        # make dictionary with k: rank, v: actual device
        dev_dct = {i: actual_devices[i] for i in range(world_size)}
        logger.info(f"Devices dict: {dev_dct}")
        # Used to launch multiple processes, each with its own Python interpreter.
        mp.spawn(
            train_base,
            args=(cfg, world_size, dev_dct),
            nprocs=world_size,
            join=True
        )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("train_base.py: device = ", device)
        train_base(device, cfg)

    
if __name__ == "__main__":
    main()