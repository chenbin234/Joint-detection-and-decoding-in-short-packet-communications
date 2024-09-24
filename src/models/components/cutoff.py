# This function is to cutoff the delay from the received signal.
# input size = (batch_size, 2, nb, n // nb * N_up + delay_max)
# output size = (batch_size, 2, n * N_up)
import torch


def cutoff(x_delay, delay_one_hot):
    """
    Cutoff the delay from the received signal.

    Parameters:
    - x_delay (torch.Tensor): The received signal of shape (batch_size, 2, nb, n // nb * N_up + delay_max), each element is a real number.
    - delay_one_hot (torch.Tensor): The one-hot encoded delay of shape (batch_size, delay_max + 1)

    Returns:
    - received_signal_cut (torch.Tensor): The output signal of shape (batch_size, 2, n * N_up)
    """

    # get the batch size, number of bits, and block length
    # block_length = n // nb * N_up + delay_max
    batch_size, _, nb, block_length = x_delay.shape

    # get the delay_max
    delay_max = delay_one_hot.shape[-1] - 1

    # get the delay
    delay = torch.argmax(delay_one_hot, dim=-1)

    # cutoff the delay from the received signal
    received_signal_cut = torch.zeros(
        (batch_size, 2, nb, block_length - delay_max), dtype=torch.float32
    )

    for i in range(batch_size):
        for j in range(nb):
            received_signal_cut[i, :, j, :] = x_delay[
                i, :, j, delay[i] : delay[i] + block_length - delay_max
            ]

    # received_signal_cut has the shape (batch_size, 2, nb, n // nb * N_up), where nb is the number of blocks
    # we want to stack the blocks on the 3rd dimension to get the shape (batch_size, 2, n * N_up)
    y_delay_removed = received_signal_cut.reshape(
        (batch_size, 2, nb * (block_length - delay_max))
    )

    return y_delay_removed
