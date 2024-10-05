"""
This file is to train the model with the CNN AutoEncoder architecture

"""

# 1. Import Libraries

import sys

sys.path.append("src/")
import os
from train.train_utils import training_loop
from models.CNN_AutoEncoder import CNN_AutoEncoder
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
        N_prime=config.N_prime,
        k_prime=config.k_prime,
        k=config.k,
        L=config.L,
        n=config.n,
        k_mod=config.k_mod,
    )

    # Make the loss and optimizer
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    return model, loss_fn, optimizer


def model_pipeline(hyperparameters):

    # tell wandb to get started
    with wandb.init(
        project="Joint-detection-and-decoding-in-short-packet-communications",
        config=hyperparameters,
    ):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # make the model, data, and optimization problem
        model, loss_fn, optimizer = make(config)
        print(model)

        # and use them to train the model
        model = training_loop(
            model_type=config.model_type,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
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
    save_model_name = "CNN_AutoEncoder_" + time
    # create the folder if it doesn't exist
    if not os.path.exists(f"./models/{save_model_name}"):
        os.makedirs(f"./models/{save_model_name}")

    config = dict(
        model_type="CNN_AutoEncoder",
        description="try to reproduce Nourhan Hesham's results",
        trainable_parameters=0,
        epochs=100,
        training_steps=1,
        batch_size=500,
        learning_rate=1e-3,
        M1=200,
        M2=100,
        N_prime=2,
        k_prime=3,
        k=128 * 3,
        L=128,
        n=128,
        k_mod=2,
        snr_min=10,
        snr_max=10,
        save_model_name=save_model_name,
    )

    # Build, train and analyze the model with the pipeline
    model = model_pipeline(config)
