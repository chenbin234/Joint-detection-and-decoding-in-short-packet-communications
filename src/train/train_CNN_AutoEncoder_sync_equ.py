"""
This file is to train the model with the CNN AutoEncoder architecture
(Joint Syncronization, Equalization, and Channel Decoding)

"""

# 1. Import Libraries

import sys

sys.path.append("src/")
import os
from train.train_utils_sync_equ import training_loop
from models.CNN_AutoEncoder_sync_equ import CNN_AutoEncoder
import wandb
from datetime import datetime
import torch.optim as optim
import torch.nn as nn
import torch
import random
import numpy as np


def make(config):

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

    # Make the loss and optimizer
    loss_fn_decoding = nn.BCELoss()
    loss_fn_sync = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    return model, loss_fn_sync, loss_fn_decoding, optimizer


def model_pipeline(hyperparameters):

    # tell wandb to get started
    with wandb.init(
        project="Joint-detection-and-decoding-in-short-packet-communications",
        config=hyperparameters,
    ):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # make the model, data, and optimization problem
        model, loss_fn_sync, loss_fn_decoding, optimizer = make(config)
        print(model)

        # and use them to train the model
        model = training_loop(
            model_type=config.model_type,
            model=model,
            optimizer=optimizer,
            loss_fn_sync=loss_fn_sync,
            loss_fn_decoding=loss_fn_decoding,
            num_epochs=config.epochs,
            training_steps=config.training_steps,
            batch_size=config.batch_size,
            start_epoch=1,
            print_every=None,
            save_model_name=config.save_model_name,
            save_every=25,
            snr_min=config.snr_min,
            snr_max=config.snr_max,
            k=config.k,
            delay_max=config.delay_max,
            alpha=config.alpha,
            train_num_samples_per_epoch=config.train_num_samples_per_epoch,
            val_num_samples_per_epoch=config.val_num_samples_per_epoch,
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

    time = datetime.now().strftime("%Y%m%d_%H_%M_%S")
    # defining model save location
    save_model_name = "CNN_AutoEncoder_Sync_Equ_" + time
    # create the folder if it doesn't exist
    if not os.path.exists(f"./models/{save_model_name}"):
        os.makedirs(f"./models/{save_model_name}")

    config = dict(
        model_type="CNN_AutoEncoder",
        description="CNN AutoEncoder with Joint Syncronization, Equalization, and Channel Decoding",
        trainable_parameters=0,
        epochs=1,
        training_steps=2,
        batch_size=500,
        learning_rate=1e-3,
        M1=100,
        M2=100,
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
        train_num_samples_per_epoch=1e3,
        val_num_samples_per_epoch=1e3,
        save_model_name=save_model_name,
    )

    # Build, train and analyze the model with the pipeline
    model = model_pipeline(config)
