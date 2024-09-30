import os
import sionna as sn

# Import TensorFlow and NumPy
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import time
from sionna.utils.plotting import PlotBER, sim_ber

# For the implementation of the Keras models
from tensorflow.keras import Model

import wandb

class CodedSystemAWGN(Model): # Inherits from Keras Model
    def __init__(self, num_bits_per_symbol, n, coderate):
        super().__init__() # Must call the Keras model initializer

        self.num_bits_per_symbol = num_bits_per_symbol
        self.n = n
        self.k = int(n*coderate)
        self.coderate = coderate
        self.constellation = sn.mapping.Constellation("qam", self.num_bits_per_symbol)

        self.mapper = sn.mapping.Mapper(constellation=self.constellation)
        self.demapper = sn.mapping.Demapper("app", constellation=self.constellation)

        self.binary_source = sn.utils.BinarySource()
        self.awgn_channel = sn.channel.AWGN()

        self.encoder = sn.fec.ldpc.LDPC5GEncoder(self.k, self.n)
        self.decoder = sn.fec.ldpc.LDPC5GDecoder(self.encoder, hard_out=True)

    #@tf.function # activate graph execution to speed things up
    def __call__(self, batch_size, ebno_db):
        no = sn.utils.ebnodb2no(ebno_db, num_bits_per_symbol=self.num_bits_per_symbol, coderate=self.coderate)

        bits = self.binary_source([batch_size, self.k])
        codewords = self.encoder(bits)
        x = self.mapper(codewords)
        y = self.awgn_channel([x, no])
        llr = self.demapper([y,no])
        bits_hat = self.decoder(llr)
        return bits, bits_hat
    
CODERATE = 0.5
BATCH_SIZE = 2000
NUM_BITS_PER_SYMBOL = 2

model_coded_awgn = CodedSystemAWGN(num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
                                   n=128,
                                   coderate=CODERATE)

snr_db = np.linspace(1, 5, 5)
ber, bler = sim_ber(model_coded_awgn, ebno_dbs=snr_db, batch_size=BATCH_SIZE, num_target_block_errors=5000, max_mc_iter=100)


# Initialize Weights and Biases
wandb.login()

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