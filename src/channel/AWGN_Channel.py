# This file is to build a AWGN channel with input (transmitter_signal, SNR) and output (output_signal)
# The channel is defined as:
#     output_signal = transmitter_signal + noise
#     noise ~ N(0, 1/SNR)
# The input signal and output signal are both in the form of torch tensor
# The input SNR is a scalar
# The output signal is the same shape as the input signal

import torch


def AWGN_Channel(transmitter_signal, SNR_db):
    """
    Function to add AWGN to the input signal.

    Args:
        transmitter_signal (torch.Tensor): The input signal of size (batch_size, 2, n).
        SNR_db (float): The signal-to-noise ratio in dB.

    Returns:
        torch.Tensor: The output signal.
    """
    device = transmitter_signal.device

    # Convert SNR from dB to linear scale
    SNR = 10 ** (SNR_db / 10).to(device)

    # Calculate the signal power
    signal_power = calculate_average_power(transmitter_signal)

    # Calculate the noise power
    noise_power = signal_power / SNR

    # Generate the noise
    # noise = torch.randn(transmitter_signal.shape).to(device) * torch.sqrt(
    #     torch.tensor(noise_power)
    # )
    noise = torch.randn(transmitter_signal.shape).to(device) * torch.sqrt(
        noise_power.clone().detach()
    )

    # Output signal
    output_signal = transmitter_signal + noise

    return output_signal


def calculate_average_power(signal):
    """
    Calculate the average power of a complex signal.

    Parameters:
    signal (torch.Tensor): Input signal of size (batch_size, 2, n)

    Returns:
    torch.Tensor: Average power for each batch
    """
    # Compute power of the real part
    real_power = signal[:, 0, :] ** 2

    # Compute power of the imaginary part
    imag_power = signal[:, 1, :] ** 2

    # Total power is the sum of real and imaginary powers
    total_power = real_power + imag_power

    # Average the power across the signal dimension
    average_power_batch = total_power.mean(dim=1)

    # take the average of the power across the batch dimension
    average_power = average_power_batch.mean()

    return average_power


# Example usage
# batch_size = 10
# n = 100
# signal = torch.randn(batch_size, 2, n)
# average_power = calculate_average_power(signal)
# print(average_power)
