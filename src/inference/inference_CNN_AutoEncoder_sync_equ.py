"""
This file is to inference the model with the CNN AutoEncoder architecture
(Joint Syncronization, Equalization, and Channel Decoding)

"""

# 1. Import Libraries

import sys

sys.path.append("src/")
import os
from inference.inference_utils_sync_equ import inference_loop
from models.CNN_AutoEncoder_sync_equ import CNN_AutoEncoder
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
        k=config.k,
        N=config.N,
        L=config.L,
        k_mod=config.k_mod,
        F=config.F,
        delay_max=config.delay_max,
        nb=config.nb,
        N_up=config.N_up,
        tp=config.tp,
    )

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
        # print(model)

        # and use them to train the model
        model = inference_loop(
            model_type=config.model_type,
            model=model,
            batch_size=config.batch_size,
            test_dataloader=test_dataloader,
            test_snr_min=config.snr_min,
            test_snr_max=config.snr_max,
            delay_max=config.delay_max,
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
    save_model_folder = "CNN_AutoEncoder_Sync_Equ_20240924_14_41_37"

    config = dict(
        model_type="CNN_AutoEncoder",
        description="inference CNN AutoEncoder with Joint Syncronization, Equalization, and Channel Decoding",
        save_model_folder=save_model_folder,  # the folder that the model be saved
        model_file=f"./models/{save_model_folder}/{save_model_folder + '_epoch5'}.pth",
        batch_size=512,
        learning_rate=1e-3,
        M1=300,
        M2=200,
        k=80,  # number of information bits
        N=288,  # number of coded bits
        L=16,  # L is the greatest common divisor of N and K
        k_prime=4,  # k = K_prime * L
        N_prime=18,  # N = N_prime * L
        n=144,  # number of complex channel uses
        k_mod=2,  # number of bits per symbol
        F=20,  # info feature size, indicating the number of info exchanged between  EQ-CNN and DEC-CNN
        delay_max=10,  # maximum delay units in the channel
        nb=4,  # number of fading blocks
        ts=1,  # sampling interval
        N_up=5,  # upsampling rate
        tp=5,  # period of the pulse
        snr_min=2,
        snr_max=20,
        alpha=0.01,  # factor of the sync loss in the total loss
    )

    # Build, train and analyze the model with the pipeline
    model = model_pipeline(config)
