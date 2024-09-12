"""
This file is to train the model with the CNN AutoEncoder architecture

"""

# 1. Import Libraries

import sys

sys.path.append("src/")
import os
from train.train_utils import training_loop
from models.CNN_AutoEncoder import CNN_AutoEncoder
from features.build_pytorch_dataset import InfobitDataset
import wandb
from datetime import datetime
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import random
import numpy as np


def make(config):

    # create the train and val datasets
    train_dataset = InfobitDataset(config.train_dataset_path)
    val_dataset = InfobitDataset(config.val_dataset_path)

    # create the train and val dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False
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

    # Make the loss and optimizer
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    return model, train_dataloader, val_dataloader, loss_fn, optimizer


def model_pipeline(hyperparameters):

    # tell wandb to get started
    with wandb.init(
        project="Joint-detection-and-decoding-in-short-packet-communications",
        config=hyperparameters,
    ):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # make the model, data, and optimization problem
        model, train_dataloader, val_dataloader, loss_fn, optimizer = make(config)
        print(model)

        # and use them to train the model
        model = training_loop(
            model_type=config.model_type,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            num_epochs=config.epochs,
            start_epoch=1,
            print_every=5,
            save_model_name=config.save_model_name,
            save_every=1,
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
    save_model_name = "CNN_AutoEncoder" + time
    # create the folder if it doesn't exist
    if not os.path.exists(f"./models/{save_model_name}"):
        os.makedirs(f"./models/{save_model_name}")

    config = dict(
        model_type="CNN_AutoEncoder",
        trainable_parameters=0,
        train_dataset_path="data/processed/train_dataset_info_bits_2000_1_64.pt",
        val_dataset_path="data/processed/val_dataset_info_bits_1000_1_64.pt",
        epochs=4,
        batch_size=64,
        learning_rate=1e-3,
        M1=200,
        M2=100,
        N_prime=4,
        k=64,
        L=64,
        n=128,
        k_mod=2,
        save_model_name=save_model_name,
    )

    # Build, train and analyze the model with the pipeline
    model = model_pipeline(config)
