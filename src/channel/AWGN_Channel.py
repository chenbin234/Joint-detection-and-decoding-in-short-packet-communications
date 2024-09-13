# This file is to build a AWGN channel with input (transmitter_signal, SNR) and output (output_signal)
# The channel is defined as:
#     output_signal = transmitter_signal + noise
#     noise ~ N(0, 1/SNR)
# The input signal and output signal are both in the form of torch tensor
# The input SNR is a scalar
# The output signal is the same shape as the input signal

import torch


def AWGN_Channel(transmitter_signal, SNR_db):

    # Convert SNR from dB to linear scale
    SNR = 10 ** (SNR_db / 10)

    # Calculate the noise power
    noise_power = 1 / SNR

    # Generate the noise
    noise = torch.randn(transmitter_signal.shape) * torch.sqrt(torch.tensor(noise_power))

    # Output signal
    output_signal = transmitter_signal + noise

    return output_signal
