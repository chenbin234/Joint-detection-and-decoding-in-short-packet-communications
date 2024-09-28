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
        4.5,
        5,
    ]
)

ber = np.array([0.04224031, 0.02883016, 0.01872906, 0.01152453, 0.00657063,
       0.00360719, 0.00187   , 0.00089437, 0.00043266])

bler = np.array([0.77155, 0.64605, 0.5077 , 0.3698 , 0.24426, 0.15077, 0.08641,
       0.04475, 0.02295])

# tell wandb to get started
with wandb.init(
    project="Joint-detection-and-decoding-in-short-packet-communications",
):

    # define our custom x axis metric
    wandb.define_metric("custom_step")

    # define which metrics will be plotted against it
    # first plot for BER
    wandb.define_metric(
        f"BER v.s. SNR (AWGN)",
        step_metric="custom_step",
    )

    # second plot for BLER
    wandb.define_metric(
        f"BLER v.s. SNR (AWGN)",
        step_metric="custom_step",
    )

    for i in range(len(snr_db)):

        log_dict = {
            f"BER v.s. SNR (AWGN)": ber[i],
            "custom_step": snr_db[i],
        }
        wandb.log(log_dict)

        log_dict = {
            f"BLER v.s. SNR (AWGN)": bler[i],
            "custom_step": snr_db[i],
        }
        wandb.log(log_dict)
