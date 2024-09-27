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

ber = np.array([1.1580e-03,4.7531e-04,1.7297e-04, 5.7656e-05,2.0156e-05,7.0312e-06,1.5625e-06, 1.5625e-07 ,0])

bler = np.array([4.6050e-02,2.0160e-02, 7.9200e-03,2.6900e-03, 1.0300e-03, 3.1000e-04, 8.0000e-05, 1.0000e-05, 0])

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
