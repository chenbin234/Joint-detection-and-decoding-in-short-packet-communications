# This file is to plot the BER/BLER v.s. SNR (AWGN) for the LDPC code
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

ber = np.array([1.194e-1, 8.528e-2, 3.958e-2, 1.642e-2, 5.442e-3, 1.427e-3, 2.826e-4])

bler = np.array(
    [
        5.852e-1,
        3.870e-1,
        2.161e-1,
        9.275e-2,
        3.207e-2,
        8.603e-3,
        1.859e-3
    ]
)

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
