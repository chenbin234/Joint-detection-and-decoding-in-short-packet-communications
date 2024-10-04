# This file contains the implementation of the CNN AutoEncoder model using PyTorch (Xi Zhang's Master Thesis)
# The model consists of an Transmitter and a Receiver
# refer to the paper for more details: https://arxiv.org/abs/2306.09258
#! Note that the model does not include synchronization and Equation

import sys

sys.path.append("src/")
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.components.CNN_block import CNN_block
from channel.AWGN_Channel import AWGN_Channel


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
            in_channels=M1, out_channels=M1, kernel_size=5, num_blocks=1, padding="same"
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
            num_blocks=1,
            padding="same",
        )

        # input size = (batch_size, M2, n), output size = (batch_size, 2, n)
        #! This part is different from the Xi Zhang's implementation, Xi Zhang has the out_channels = 1
        self.modulator2 = nn.Conv1d(
            in_channels=M2, out_channels=2, kernel_size=1, padding="same"
        )
        self.modulator2_batchnorm = nn.BatchNorm1d(2)
        self.modulator2_linear = nn.Identity()

        # ! In order to reproduce Xi's results, we now set the out_channels = 1
        # self.modulator2 = nn.Conv1d(
        #     in_channels=M2, out_channels=1, kernel_size=1, padding="same"
        # )
        # self.modulator2_batchnorm = nn.BatchNorm1d(1)
        # self.modulator2_linear = nn.Identity()

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


class Receiver(nn.Module):
    def __init__(self, M1, M2, k_mod, L, N_prime):
        super(Receiver, self).__init__()

        # Demodulator part
        # input size = (batch_size, 2, n), output size = (batch_size, M2, n)
        self.demodulator1 = CNN_block(
            in_channels=2, out_channels=M2, kernel_size=5, num_blocks=1, padding="same"
        )

        # # ! In order to reproduce Xi's results, we now set the in_channels = 1
        # self.demodulator1 = CNN_block(
        #     in_channels=1, out_channels=M2, kernel_size=5, num_blocks=3, padding="same"
        # )

        # input size = (batch_size, M2, n), output size = (batch_size, k_mod, n)
        self.demodulator2 = nn.Conv1d(
            in_channels=M2, out_channels=k_mod, kernel_size=5, padding="same"
        )
        self.demodulator2_batchnorm = nn.BatchNorm1d(k_mod)
        self.demodulator2_linear = nn.Identity()

        # reshape the tensor (batch_size, k_mod, n) to (batch_size, N_prime, L)
        self.reshape_layer = ReshapeLayer(N_prime, L)
        # self.reshape_layer = nn.Lambda(lambda x: x.view(-1, N_prime, L))

        # Decoder part
        # input size = (batch_size, N_prime, L), output size = (batch_size, M1, L)
        self.decoder1 = CNN_block(
            in_channels=N_prime,
            out_channels=M1,
            kernel_size=5,
            num_blocks=2,
            padding="same",
        )

        # input size = (batch_size, M1, L), output size = (batch_size, 1, k)
        #! in this case, we set L = k = 64
        self.decoder2 = nn.Conv1d(
            in_channels=M1, out_channels=1, kernel_size=1, padding="same"
        )
        # sigmoid activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Demodulator part
        # input size = (batch_size, 2, n), output size = (batch_size, M2, n)
        x = self.demodulator1(x)

        # input size = (batch_size, M2, n), output size = (batch_size, k_mod, n)
        x = self.demodulator2(x)
        x = self.demodulator2_batchnorm(x)
        x = self.demodulator2_linear(x)

        # Reshape the tensor (batch_size, k_mod, n) to (batch_size, N_prime, L)
        x = self.reshape_layer(x)

        # Decoder part
        # input size = (batch_size, N_prime, L), output size = (batch_size, M1, L)
        x = self.decoder1(x)

        # input size = (batch_size, M1, L), output size = (batch_size, 1, k)
        x = self.decoder2(x)
        x = self.sigmoid(x)

        return x


class CNN_AutoEncoder(nn.Module):
    def __init__(self, M1, M2, N_prime, k, L, n, k_mod):
        super(CNN_AutoEncoder, self).__init__()

        self.transmitter = Transmitter(M1, M2, N_prime, k, L, n, k_mod)
        self.receiver = Receiver(M1, M2, k_mod, L, N_prime)

        # self.channel = AWGN_Channel()

    def forward(self, x, SNR_db):
        x = self.transmitter(x)
        x = AWGN_Channel(x, SNR_db=SNR_db)
        x = self.receiver(x)

        return x
