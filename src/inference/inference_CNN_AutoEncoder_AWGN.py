"""
This file is to train the model with the CNN AutoEncoder architecture

"""

# 1. Import Libraries

import sys

sys.path.append("src/")
import os
from inference.inference_utils import inference_loop
from models.CNN_AutoEncoder import CNN_AutoEncoder
from features.build_pytorch_dataset import InfobitDataset
from torch.utils.data import DataLoader
import wandb
from datetime import datetime
import torch.optim as optim
import torch.nn as nn
import torch
import random
import numpy as np


def make(config):

    # test dataset
    test_dataset = InfobitDataset(num_samples=1e6, k=config.k)

    # create test dataloader
    test_dataloader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False
    )

    # Initialize the model
    model = CNN_AutoEncoder(
        M1=config.M1,
        M2=config.M2,
        N_prime=config.N_prime,
        k=config.k,
        L=config.L,
        n=config.n,
        k_mod=config.k_mod,
    )

    # load the well trained model
    model.load_state_dict(torch.load(config.model_file)["model_state_dict"])

    return model, test_dataloader


def model_pipeline(hyperparameters):

    # tell wandb to get started
    with wandb.init(
        project="Joint-detection-and-decoding-in-short-packet-communications",
        config=hyperparameters,
    ):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # make the model, data, and optimization problem
        model, test_dataloader = make(config)
        print(model)

        # and use them to train the model
        model = inference_loop(
            model_type=config.model_type,
            model=model,
            test_dataloader=test_dataloader,
            test_snr_min=config.snr_min,
            test_snr_max=config.snr_max,
            k=config.k,
        )

    return model


if __name__ == "__main__":

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    random.seed(hash("setting random seeds") % 2**32 - 1)
    np.random.seed(hash("improves reproducibility") % 2**32 - 1)
    torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
    torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

    # Initialize Weights and Biases
    wandb.login()

    # defining model save location
    save_model_folder = "CNN_AutoEncoder_20240916_16_28_11"

    config = dict(
        model_type="CNN_AutoEncoder",
        description="debugging",
        save_model_folder=save_model_folder,  # the folder that the model be saved
        model_file=f"./models/{save_model_folder}/{save_model_folder + '_epoch100'}.pth",
        batch_size=512,
        learning_rate=1e-3,
        M1=200,
        M2=100,
        N_prime=4,
        k=64,
        L=64,
        n=128,
        k_mod=2,
        snr_min=1,
        snr_max=5,
    )

    # Build, train and analyze the model with the pipeline
    model = model_pipeline(config)
