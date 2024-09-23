# This file contains the implementation of the CNN AutoEncoder model using PyTorch (Xi Zhang's Master Thesis)
# The model consists of an Transmitter and a Receiver
#! Note this is An AE-based Joint Synchronization, Equalization, and Decoding System
#! The transmitter is the same as before, and the receiver was designed to have a iterative architecture


import sys

sys.path.append("src/")
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.components.CNN_block import CNN_block
from channel.Block_Fading_Channel import Block_fading_channel
from models.components.cutoff import cutoff


class ReshapeLayer(nn.Module):
    def __init__(self, k_mod, n):
        super(ReshapeLayer, self).__init__()
        self.k_mod = k_mod
        self.n = n

    def forward(self, x):
        return x.view(-1, self.k_mod, self.n)


class Transmitter(nn.Module):
    def __init__(self, M1, M2, N_prime, k, L, n, k_mod):
        """
        Initializes the Transmitter model.
        Args:
            M1 (int): Number of output channels for the first convolutional layer.
            M2 (int): Number of output channels for the modulator block.
            N_prime (int): Number of output channels for the third convolutional layer.
            k (int): Kernel size for the first convolutional layer.
            L (int): Length parameter for the first convolutional layer.
            n (int): Reshape parameter for the reshape layer.
            k_mod (int): Number of input channels for the modulator block.
        Layers:
            encoding1 (nn.Conv1d): First convolutional layer.
            batch_norm1 (nn.BatchNorm1d): Batch normalization for the first convolutional layer.
            ELU1 (nn.ELU): ELU activation for the first convolutional layer.
            encoding2 (CNN_block): Second convolutional block.
            encoding3 (nn.Conv1d): Third convolutional layer.
            encoding3_batchnorm (nn.BatchNorm1d): Batch normalization for the third convolutional layer.
            encoding3_ELU (nn.ELU): ELU activation for the third convolutional layer.
            reshape_layer (nn.Lambda): Layer to reshape the tensor.
            modulator1 (CNN_block): First modulator block.
            modulator2 (nn.Conv1d): Second modulator convolutional layer.
            modulator2_batchnorm (nn.BatchNorm1d): Batch normalization for the second modulator layer.
            modulator2_linear (nn.Identity): Identity layer for the modulator.
        """

        super(Transmitter, self).__init__()

        # Encoder part
        # input size = (batch_size, 1, k), output size = (batch_size, M1, L)
        self.encoding1 = nn.Conv1d(
            in_channels=1, out_channels=M1, kernel_size=(k + 1 - L), padding="valid"
        )
        self.batch_norm1 = nn.BatchNorm1d(M1)
        self.ELU1 = nn.ELU()

        # input size = (batch_size, M1, L), output size = (batch_size, M1, L)
        self.encoding2 = CNN_block(
            in_channels=M1, out_channels=M1, kernel_size=5, num_blocks=3, padding="same"
        )

        # input size = (batch_size, M1, L), output size = (batch_size, N_prime, L)
        self.encoding3 = nn.Conv1d(
            in_channels=M1, out_channels=N_prime, kernel_size=5, padding="same"
        )
        self.encoding3_batchnorm = nn.BatchNorm1d(N_prime)
        self.encoding3_ELU = nn.ELU()

        # reshape the tensor (batch_size, N_prime, L) to (batch_size, k_mod, n)
        # self.reshape_layer = nn.Lambda(lambda x: x.view(-1, k_mod, n))
        self.reshape_layer = ReshapeLayer(k_mod, n)

        # Modulator part
        # input size = (batch_size, k_mod, n), output size = (batch_size, M2, n)
        self.modulator1 = CNN_block(
            in_channels=k_mod,
            out_channels=M2,
            kernel_size=5,
            num_blocks=4,
            padding="same",
        )

        # input size = (batch_size, M2, n), output size = (batch_size, 2, n)
        #! This part is different from the Xi Zhang's implementation, Xi Zhang has the out_channels = 1
        self.modulator2 = nn.Conv1d(
            in_channels=M2, out_channels=2, kernel_size=1, padding="same"
        )
        self.modulator2_batchnorm = nn.BatchNorm1d(2)
        self.modulator2_linear = nn.Identity()

    def forward(self, x):

        # Encoding part
        # input size = (batch_size, 1, k), output size = (batch_size, M1, L)
        x = self.encoding1(x)
        x = self.batch_norm1(x)
        x = self.ELU1(x)

        # input size = (batch_size, M1, L), output size = (batch_size, M1, L)
        x = self.encoding2(x)

        # input size = (batch_size, M1, L), output size = (batch_size, N_prime, L)
        x = self.encoding3(x)
        x = self.encoding3_batchnorm(x)
        x = self.encoding3_ELU(x)

        # Reshape the tensor (batch_size, N_prime, L) to (batch_size, k_mod, n)
        x = self.reshape_layer(x)

        # Modulator part
        # input size = (batch_size, k_mod, n), output size = (batch_size, M2, n)
        x = self.modulator1(x)

        # input size = (batch_size, M2, n), output size = (batch_size, 2, n)
        x = self.modulator2(x)
        x = self.modulator2_batchnorm(x)
        x = self.modulator2_linear(x)

        return x


class EQ_CNN_block(nn.Module):
    def __init__(self, M2, F, num_blocks=5):
        super(EQ_CNN_block, self).__init__()

        self.cnn_block1 = CNN_block(
            in_channels=M2,
            out_channels=M2,
            kernel_size=5,
            num_blocks=num_blocks,
            padding="same",
        )
        self.cnn_block2 = nn.Conv1d(
            in_channels=M2, out_channels=F, kernel_size=1, padding="same"
        )

    def forward(self, x):
        # input size = (batch_size, M2, n), output size = (batch_size, M2, n)
        x = self.cnn_block1(x)

        # input size = (batch_size, M2, n), output size = (batch_size, F, n)
        x = self.cnn_block2(x)

        return x


class DEC_CNN_block(nn.Module):
    def __init__(self, M1, F, num_blocks=5):
        super(EQ_CNN_block, self).__init__()

        self.cnn_block1 = CNN_block(
            in_channels=F,
            out_channels=M1,
            kernel_size=5,
            num_blocks=num_blocks,
            padding="same",
        )
        self.cnn_block2 = nn.Conv1d(
            in_channels=M1, out_channels=F, kernel_size=1, padding="same"
        )

    def forward(self, x):
        # input size = (batch_size, F, n), output size = (batch_size, M1, n)
        x = self.cnn_block1(x)

        # input size = (batch_size, M1, n), output size = (batch_size, F, n)
        x = self.cnn_block2(x)

        return x


class DEC_CNN_last_block(nn.Module):
    def __init__(self, M1, F, n, k, num_blocks=5):
        super(EQ_CNN_block, self).__init__()

        self.cnn_block1 = CNN_block(
            in_channels=F,
            out_channels=M1,
            kernel_size=5,
            num_blocks=num_blocks,
            padding="same",
        )

        # input size = (batch_size, M1, n), output size = (batch_size, M1*n)
        self.flatten = nn.Flatten()

        self.decoder_ouput = nn.Linear(in_features=M1 * n, out_features=k)

    def forward(self, x):
        # input size = (batch_size, F, n), output size = (batch_size, M1, n)
        x = self.cnn_block1(x)

        # flattten layer
        # input size = (batch_size, M1, n), output size = (batch_size, M1*n)
        x = self.flatten(x)

        # final decoder output
        # input size = (batch_size, M1*n), output size = (batch_size, k)
        x = self.decoder_ouput(x)

        # add one dimension to the output (second dimension) to make sure the output has the shape (batch_size, 1, k)
        x = x.unsqueeze(1)

        return x


class Sync_Block(nn.Module):
    """
    Synchronization block for a CNN AutoEncoder.

    Args:
        n (int): number of complex channel uses.
        N_up (int): Upsampling factor.
        delay_max (int): Maximum delay.
        nb (int): Number of blocks.
        num_blocks (int, optional): Number of CNN blocks. Default is 5.

    Attributes:
        cnn_block1 (CNN_block): The first CNN block with specified parameters.
        cnn_block2 (nn.Conv1d): A 1D convolutional layer.
        flatten (nn.Flatten): A flattening layer.
        sync_output (nn.Linear): A linear layer for synchronization output.
        softmax (nn.Softmax): A softmax layer for output normalization.

    Methods:
        forward(x):
            Forward pass of the synchronization block.

            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, nb, n // nb * N_up + delay_max, 2).

            Returns:
                torch.Tensor: Output tensor of shape (batch_size, delay_max + 1).
    """

    def __init__(self, n, N_up, delay_max, nb, num_blocks=5):
        super(Sync_Block, self).__init__()

        self.cnn_block1 = CNN_block(
            in_channels=2,
            out_channels=100,
            kernel_size=5,
            num_blocks=num_blocks,
            padding="same",
        )
        self.cnn_block2 = nn.Conv1d(
            in_channels=100, out_channels=1, kernel_size=1, padding="same"
        )

        self.flatten = nn.Flatten()

        self.sync_output = nn.Linear(
            in_features=n * N_up + delay_max * nb, out_features=delay_max + 1
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        # reshape the input to (batch_size, 2, P), where P = n * N_up + delay_max * nb
        x = x.view(x.size(0), 2, -1)

        # input size = (batch_size, 2, P)
        # output size = (batch_size, 100, P)
        x = self.cnn_block1(x)

        # input size = (batch_size, 100, P), output size = (batch_size, 1, P)
        x = self.cnn_block2(x)

        # flatten layer
        x = self.flatten(x)

        # final sync output
        # input size = (batch_size, P), output size = (batch_size, delay_max + 1)
        x = self.sync_output(x)

        # softmax layer
        # input size = (batch_size, delay_max + 1), output size = (batch_size, delay_max + 1)
        x = self.softmax(x)

        return x


class Receiver(nn.Module):
    def __init__(
        self,
        M1,
        M2,
        F,
        n,
        k,
        N_up,
        delay_max,
        nb,
    ):
        super(Receiver, self).__init__()

        # Sync Block
        # input size = (batch_size, 2, nb, n // nb * N_up + delay_max)
        # output size = (batch_size, 1, delay_max + 1)
        self.sync_block = Sync_Block(n, N_up, delay_max, nb, num_blocks=5)

        # input size = (batch_size, 2, nb, n // nb * N_up + delay_max), output size = (batch_size, 2, n * N_up)
        self.dec_input = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=M2, kernel_size=5, stride=N_up),
            nn.BatchNorm1d(M2),
            nn.ELU(),
        )

        # input size = (batch_size, M2, n), output size = (batch_size, F, n)
        self.eq_block = EQ_CNN_block(M2, F)

        # input size = (batch_size, F, n), output size = (batch_size, M1, n)
        self.dec_block = DEC_CNN_block(M1, F)

        # input size = (batch_size, F, n), output size = (batch_size, 1, k)
        self.dec_last_block = DEC_CNN_last_block(M1, F, n, k)

    def forward(self, x_delay, true_delay_onehot, num_iteration, training=True):
        # Sync Block
        # input size = (batch_size, 2, nb, n // nb * N_up + delay_max)
        # output size = (batch_size, 1, delay_max + 1)
        estimated_delay = self.sync_block(x_delay)

        # cutoff & concatenate the received signal
        # input size = (batch_size, 2, nb, n // nb * N_up + delay_max)
        # output size = (batch_size, 2, n * N_up)
        if training:

            y_delay_removed = cutoff(x_delay, estimated_delay)

        else:
            y_delay_removed = cutoff(x_delay, true_delay_onehot)

        # input size = (batch_size, 2, n * N_up), output size = (batch_size, M2, n)
        x = self.dec_input(x)

        # initiate i_c
        i_c_pri = 0

        # iterative EQ-DEC block
        for i in range(num_iteration):

            # EQ-CNN block
            # input size = (batch_size, M2, n), output size = (batch_size, F, n)
            i_c = self.eq_block(y_delay_removed)

            # update i_c
            i_c = i_c - i_c_pri

            # DEC-CNN block
            # input size = (batch_size, F, n), output size = (batch_size, M1, n)
            i_b = self.dec_block(i_c)

            # update i_c_pri
            i_c_pri = i_b - i_c

        #! last iteration
        # EQ-CNN last block
        # input size = (batch_size, M2, n), output size = (batch_size, F, n)
        i_c = self.eq_block(y_delay_removed)

        # update i_c
        i_c = i_c - i_c_pri

        # DEC-CNN last block
        # input size = (batch_size, F, n), output size = (batch_size, 1, k)
        y_decoded = self.dec_last_block()

        return estimated_delay, y_decoded


class CNN_AutoEncoder(nn.Module):
    def __init__(self, M1, M2, k, N, L, k_mod, F, delay_max, nb, N_up, tp):
        super(CNN_AutoEncoder, self).__init__()

        self.k_prime = k / L
        self.N_prime = N / L
        self.n = N / k_mod
        self.tp = tp
        self.N_up = N_up
        self.nb = nb
        self.delay_max = delay_max

        self.transmitter = Transmitter(M1, M2, self.N_prime, k, L, self.n, k_mod)
        self.receiver = Receiver(M1, M2, F, self.n, k, N_up, self.delay_max, nb)

    def forward(
        self, x, true_delay, true_delay_onehot, SNR_db, training=True, num_iteration=5
    ):

        # Transmitter part
        # input size = (batch_size, 1, k), output size = (batch_size, 2, n)
        x = self.transmitter(x)

        # Channel part (including upsampling and pulse shaping)
        # input size = (batch_size, 2, n)
        # output size = (batch_size, nb, n // nb * N_up + delay_max, 2)
        #! Todo: probably miss power normalization
        x_delay = Block_fading_channel(
            transmitted_signal=x,
            tp=self.tp,
            N_up=self.N_up,
            nb=self.nb,
            delay=true_delay,
            SNR_db=SNR_db,
            delay_max=self.delay_max,
        )

        # Receiver part
        # input size = (batch_size, nb, n // nb * N_up + delay_max, 2)
        if training:

            # output size
            # estimated_delay: (batch_size, 1, delay_max + 1)
            # x: (batch_size, 1, k)
            estimated_delay, y_decoded = self.receiver(
                x_delay, true_delay_onehot, num_iteration, training=True
            )

            return estimated_delay, y_decoded

        else:
            # output size
            # estimated_delay: (batch_size, 1, delay_max + 1)
            # x: (batch_size, 1, k)
            estimated_delay, y_decoded = self.receiver(
                x_delay, true_delay_onehot, num_iteration, training=False
            )

            return estimated_delay, y_decoded
