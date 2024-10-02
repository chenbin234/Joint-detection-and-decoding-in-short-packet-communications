# This file is to plot the BLER v.s. SNR (Block Fading Channel) for the RCUS Bound
# The data is scraped from the Xi Zhang's paper using tools like WebPlotDigitizer (url: https://automeris.io/)

import wandb
import numpy as np


# Initialize Weights and Biases
wandb.login()


# scrape the data from the Xi Zhang's paper
snr_db = np.array(
    [
        2,
        4,
        6,
        8,
        10,
        12,
        14,
        16,
        18,
        20,
    ]
)

bler = np.array(
    [
        0.7127,
        0.4073,
        0.1700,
        0.0539,
        0.0137,
        0.0030,
        5.7912e-04,
        1.0289e-04,
        1.8644e-05,
        2.1618e-06,
    ]
)

# tell wandb to get started
with wandb.init(
    project="Joint-detection-and-decoding-in-short-packet-communications",
):

    # define our custom x axis metric
    wandb.define_metric("custom_step")

    # define which metrics will be plotted against it
    wandb.define_metric(
        f"BER v.s. SNR (Joint Sync and Decoding, Block Fading Channel)",
        step_metric="custom_step",
    )

    for i in range(len(snr_db)):

        log_dict = {
            f"BLER v.s. SNR (Joint Sync and Decoding, Block Fading Channel)": bler[i],
            "custom_step": snr_db[i],
        }
        wandb.log(log_dict)
