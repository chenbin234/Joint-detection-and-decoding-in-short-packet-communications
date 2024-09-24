# This file is to plot the BLER v.s. SNR (Block Fading Channel) for the CNN AutoEncoder architecture
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
        8.401e-1,
        5.985e-1,
        3.401e-1,
        1.485e-1,
        5.737e-2,
        1.784e-2,
        6.514e-3,
        2.026e-3,
        5.791e-4,
        2.301e-4,
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
