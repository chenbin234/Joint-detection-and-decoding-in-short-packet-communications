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
    def __init__(self, in_channels, M2, F, num_blocks):
        super(EQ_CNN_block, self).__init__()

        self.cnn_block1 = CNN_block(
            in_channels=in_channels,
            out_channels=M2,
            kernel_size=5,
            num_blocks=num_blocks,
            padding="same",
        )
        self.cnn_block2 = nn.Conv1d(
            in_channels=M2, out_channels=F, kernel_size=1, padding="same"
        )

    def forward(self, x):
        # input size = (batch_size, 2, n), output size = (batch_size, M2, n)
        x = self.cnn_block1(x)

        # input size = (batch_size, M2, n), output size = (batch_size, F, n)
        x = self.cnn_block2(x)

        return x


class DEC_CNN_block(nn.Module):
    def __init__(self, in_channels, M1, F, num_blocks):
        super(EQ_CNN_block, self).__init__()

        self.cnn_block1 = CNN_block(
            in_channels=in_channels,
            out_channels=M1,
            kernel_size=5,
            num_blocks=num_blocks,
            padding="same",
        )
        self.cnn_block2 = nn.Conv1d(
            in_channels=M1, out_channels=F, kernel_size=1, padding="same"
        )

    def forward(self, x):
        # input size = (batch_size, 2, n), output size = (batch_size, M2, n)
        x = self.cnn_block1(x)

        # input size = (batch_size, M2, n), output size = (batch_size, F, n)
        x = self.cnn_block2(x)

        return x


class DEC_CNN_last_block(nn.Module):
    def __init__(self, in_channels, M1, F, L, k, num_blocks):
        super(EQ_CNN_block, self).__init__()

        self.cnn_block1 = CNN_block(
            in_channels=in_channels,
            out_channels=M1,
            kernel_size=5,
            num_blocks=num_blocks,
            padding="same",
        )

        # input size = (batch_size, M1, L), output size = (batch_size, 1, k) ??
        self.flatten = nn.Flatten()

        self.decoder_ouput = nn.Linear(in_features=M1 * L, out_features=k)

    def forward(self, x):
        # input size = (batch_size, 2, n), output size = (batch_size, M2, n)
        x = self.cnn_block1(x)

        # flattten layer
        x = self.flatten(x)

        # final decoder output
        x = self.decoder_ouput(x)

        return x


class Sync_Block(nn.Module):
    def __init__(self, in_channels, M2, dmax, num_blocks=5):
        super(Sync_Block, self).__init__()

        self.cnn_block1 = CNN_block(
            in_channels=in_channels,
            out_channels=M2,
            kernel_size=5,
            num_blocks=num_blocks,
            padding="same",
        )
        self.cnn_block2 = nn.Conv1d(
            in_channels=M2, out_channels=1, kernel_size=1, padding="same"
        )

        self.flatten = nn.Flatten()

        self.sync_output = nn.Linear(in_features=dmax, out_features=dmax + 1)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        # input size = (batch_size, nb, n // nb * N_up + delay_max, 2)
        # output size = (batch_size, 1, delay_max + 1)
        x = self.cnn_block1(x)

        # input size = (batch_size, M2, n), output size = (batch_size, F, n)
        x = self.cnn_block2(x)

        # flatten layer
        x = self.flatten(x)

        # final sync output
        x = self.sync_output(x)
        x = self.softmax(x)

        return x


class Receiver(nn.Module):
    def __init__(self, M1, M2, k_mod, L, N_prime):
        super(Receiver, self).__init__()

        self.sync_block = Sync_Block(in_channels=2, M2=2, num_blocks=4, dmax=4)

        self.eq_block = EQ_CNN_block(in_channels=2, M2=2, F=2, num_blocks=4)

        self.dec_block = DEC_CNN_block(in_channels=2, M1=2, F=2, num_blocks=4)

    def forward(self, x, num_iteration, training=True):

        if training:

            # Sync Block
            # input size = (batch_size, nb, n // nb * N_up + delay_max, 2)
            # output size = (batch_size, 1, delay_max + 1)
            estimated_delay = self.sync_block(x)

            # EQ Block
            x = self.eq_block(x)

            # DEC Block
            x = self.dec_block(x)

            return estimated_delay, x
        else:
            # Sync Block
            x = self.sync_block(x)

            # EQ Block
            x = self.eq_block(x)

            # DEC Block
            x = self.dec_block(x)

            return x


class CNN_AutoEncoder(nn.Module):
    def __init__(self, M1, M2, N_prime, k, L, n, k_mod, tp, N_up, nb, delay_max):
        super(CNN_AutoEncoder, self).__init__()

        self.tp = tp
        self.N_up = N_up
        self.nb = nb
        self.delay_max = delay_max

        self.transmitter = Transmitter(M1, M2, N_prime, k, L, n, k_mod)
        self.receiver = Receiver(M1, M2, k_mod, L, N_prime)

    def forward(self, x, SNR_db, training=True):

        # Transmitter part
        # input size = (batch_size, 1, k), output size = (batch_size, 2, n)
        x = self.transmitter(x)

        # randomly generate delay for each message
        # delay has the size (batch_size, 1),
        # delay_onehot has the size (batch_size, 1, delay_max + 1)
        delay, delay_onehot = generate_random_delay(
            batch_size=x.shape[0], delay_max=self.delay_max
        )

        # Channel part (including upsampling and pulse shaping)
        # input size = (batch_size, 2, n)
        # output size = (batch_size, nb, n // nb * N_up + delay_max, 2)
        #! Todo: probably miss power normalization
        x = Block_fading_channel(
            transmitted_signal=x,
            tp=self.tp,
            N_up=self.N_up,
            nb=self.nb,
            delay=delay,
            SNR_db=SNR_db,
            delay_max=self.delay_max,
        )

        # Receiver part
        # input size = (batch_size, nb, n // nb * N_up + delay_max, 2)
        if training:

            # output size
            # estimated_delay: (batch_size, 1, delay_max + 1)
            # x: (batch_size, 1, k)
            estimated_delay, x = self.receiver(x, training=True)

            return estimated_delay, x

        else:
            # output size = (batch_size, 1, k)
            x = self.receiver(x, training=False)

            return x


def generate_random_delay(batch_size, delay_max):
    # generate random delay for each message of size (batch_size, 1), the delay is uniform distributed in [0, delay_max]
    delay = torch.randint(low=0, high=delay_max + 1, size=(batch_size, 1))

    # convert delay to one-hot encoding
    # delay_onehot has the size (batch_size, 1, delay_max + 1)
    delay_onehot = F.one_hot(delay, num_classes=delay_max + 1)

    return delay, delay_onehot
