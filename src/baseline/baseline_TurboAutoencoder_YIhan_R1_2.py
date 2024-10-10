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
        5.02333024006751,
        4.024504781428208,
        3.0207100419001542,
        2.0119460214833462,
        1.0031820010665387,
        0.009325823315989412,
    ]
)

ber = np.array(
    [
        0.00000688722259464526,
        0.0000759530797359186,
        0.001087007420916020,
        0.00949428416072908,
        0.056479616575349306,
        0.1580076077096485,
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

    for i in range(len(snr_db)):

        log_dict = {
            f"BER v.s. SNR (AWGN)": ber[i],
            "custom_step": snr_db[i],
        }
        wandb.log(log_dict)
