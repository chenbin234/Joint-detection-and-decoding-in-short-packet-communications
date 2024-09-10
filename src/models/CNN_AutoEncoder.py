import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, M1, M2, N_prime, k, L, n, k_mod):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=M1, kernel_size=(k + 1 - L), padding="valid"
            ),
            nn.BatchNorm1d(M1),
            nn.ELU(),
            nn.Conv1d(M1, M1, kernel_size=5, padding="same"),
            nn.BatchNorm1d(M1),
            nn.ELU(),
            nn.Conv1d(M1, M1, kernel_size=5, padding="same"),
            nn.BatchNorm1d(M1),
            nn.ELU(),
            nn.Conv1d(M1, M1, kernel_size=5, padding="same"),
            nn.BatchNorm1d(M1),
            nn.ELU(),
            nn.Conv1d(M1, N_prime, kernel_size=5, padding="same"),
            nn.BatchNorm1d(N_prime),
            nn.ELU(),
            nn.Reshape((n, k_mod)),
        )

        self.modulator = nn.Sequential(
            nn.Conv1d(N_prime, M2, kernel_size=5, padding="same"),
            nn.BatchNorm1d(M2),
            nn.ELU(),
            nn.Conv1d(M2, M2, kernel_size=5, padding="same"),
            nn.BatchNorm1d(M2),
            nn.ELU(),
            nn.Conv1d(M2, M2, kernel_size=5, padding="same"),
            nn.BatchNorm1d(M2),
            nn.ELU(),
            nn.Conv1d(M2, M2, kernel_size=5, padding="same"),
            nn.BatchNorm1d(M2),
            nn.ELU(),
            nn.Conv1d(M2, 1, kernel_size=1, padding="same"),
            nn.BatchNorm1d(1),
            nn.Identity(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.modulator(x)
        return x


class Decoder(nn.Module):
    def __init__(self, M1, M2, k_mod, L, N_prime):
        super(Decoder, self).__init__()

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
