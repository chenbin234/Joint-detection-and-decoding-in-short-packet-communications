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
        -1.5232004218331068,
        -1.0092750366032865,
        -0.5056281590780629,
        -0.0071205354051375735,
        0.4965263421200863,
        1.0001732196453097,
        1.498680843318235,
        1.9971884669911608,
        2.4956960906640866,
        3.0044822220416076,
        3.4978505918622353,
        4.001497469387458,
    ]
)

ber = np.array(
    [
        0.04523306432641868,
        0.024897033006163773,
        0.009398649287609145,
        0.0029153081882778305,
        0.0007032682486200488,
        0.00019696300054978117,
        0.000059671619045675884,
        0.000018655152460145152,
        0.000005741242329178127,
        0.00000165927103119287,
        3.9713843123360676e-7,
        8.515284474419691e-8,
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
