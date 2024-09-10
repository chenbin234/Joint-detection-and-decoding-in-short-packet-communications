# This file contains the implementation of the CNN AutoEncoder model using PyTorch (Xi Zhang's Master Thesis)
# The model consists of an Transmitter and a Receiver
# refer to the paper for more details: https://arxiv.org/abs/2306.09258

import sys

sys.path.append("src/")
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.components.CNN_block import CNN_block


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
        self.reshape_layer = nn.Lambda(lambda x: x.view(-1, k_mod, n))

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
        self.modulator2_batchnorm = nn.BatchNorm1d(N_prime)
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


class Receiver(nn.Module):
    def __init__(self, M1, M2, k_mod, L, N_prime):
        super(Receiver, self).__init__()

        # Demodulator part
        demodulator_layers = []
        for i in range(4):
            demodulator_layers.append(
                nn.Conv1d(
                    in_channels=M2 if i > 0 else 1,
                    out_channels=M2,
                    kernel_size=5,
                    padding="same",
                )
            )
            demodulator_layers.append(nn.BatchNorm1d(M2))
            demodulator_layers.append(nn.ELU())

        demodulator_layers.append(
            nn.Conv1d(in_channels=M2, out_channels=k_mod, kernel_size=5, padding="same")
        )
        demodulator_layers.append(nn.BatchNorm1d(k_mod))
        demodulator_layers.append(
            nn.Identity()
        )  # Equivalent to 'linear' activation in TensorFlow

        self.demodulator = nn.Sequential(*demodulator_layers)
        self.reshape_layer = nn.Lambda(
            lambda x: x.view(-1, L, N_prime)
        )  # Reshape layer

        # Decoder part
        decoder_layers = []
        for i in range(4):
            decoder_layers.append(
                nn.Conv1d(
                    in_channels=M1 if i > 0 else k_mod,
                    out_channels=M1,
                    kernel_size=5,
                    padding="same",
                )
            )
            decoder_layers.append(nn.BatchNorm1d(M1))
            decoder_layers.append(nn.ELU())

        decoder_layers.append(nn.Conv1d(in_channels=M1, out_channels=1, kernel_size=1))
        decoder_layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = self.demodulator(x)
        x = self.reshape_layer(x)
        x = self.decoder(x)
        return x
