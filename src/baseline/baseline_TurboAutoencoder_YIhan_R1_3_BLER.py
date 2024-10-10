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
        -1.4798735054096235,
        -0.9805697457970963,
        -0.4759542440610316,
        0.01803777342795776,
        0.5120297909169471,
        1.0219570347765488,
        1.5159490522655386,
        2.020564554001603,
        2.5145565714905924,
        3.019172073226657,
        3.5237875749627223,
        4.017779592451712,
    ]
)

bler = np.array(
    [
        0.638478279743635,
        0.39622943508792324,
        0.19942598966117894,
        0.08235781829847144,
        0.029925252945024416,
        0.01050049797285331,
        0.0038600820462862243,
        0.0013233091447950162,
        0.00042306220578404614,
        0.00012039562565028215,
        0.00003158214916591491,
        0.000007725930385524197,
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
        f"BLER v.s. SNR (AWGN)",
        step_metric="custom_step",
    )

    for i in range(len(snr_db)):

        log_dict = {
            f"BLER v.s. SNR (AWGN)": bler[i],
            "custom_step": snr_db[i],
        }
        wandb.log(log_dict)
