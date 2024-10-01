# This file is to plot the BER/BLER v.s. SNR (AWGN) for the CNN AutoEncoder architecture
# The data is scraped from the Xi Zhang's paper using tools like WebPlotDigitizer (url: https://automeris.io/)

import torch
import wandb
import numpy as np


# Initialize Weights and Biases
wandb.login()


# scrape the data from the Xi Zhang's paper
snr_db = np.array(
    [
        1,
        1.5,
        2,
        2.5,
        3,
        3.5,
        4,
    ]
)

ber = np.array([4.818e-2, 3.046e-2, 1.737e-2, 9.021e-3, 4.266e-3, 1.837e-3, 7.206e-4])

# bler = np.array([3.49e-1, 2.202e-1, 1.229e-1, 6.25e-2, 2.892e-2, 9.906e-3, 4.094e-3])

# BLER = 1 - (1 - BER) ** 64
bler_estimate = 1 - (1 - ber) ** 64


# tell wandb to get started
with wandb.init(
    project="Joint-detection-and-decoding-in-short-packet-communications",
):

    # define our custom x axis metric
    wandb.define_metric("custom_step")

    # define which metrics will be plotted against it
    # second plot for BLER
    wandb.define_metric(
        f"BLER v.s. SNR (AWGN)",
        step_metric="custom_step",
    )

    for i in range(len(snr_db)):

        log_dict = {
            f"BLER v.s. SNR (AWGN)": bler_estimate[i],
            "custom_step": snr_db[i],
        }
        wandb.log(log_dict)
