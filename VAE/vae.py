"""Implementation of a variational autoencoder in pytorch
https://arxiv.org/pdf/1312.6114.pdf"""

import torch
from torch import nn


class VAE(nn.Module):
    def __init__(self, encoder_sizes=(784, 400), latent_size=20):
        """MODEL DESCRIPTION"""
        super(VAE, self).__init__()

        # Model parameters.
        self.encoder_sizes = encoder_sizes
        self.decoder_sizes = (latent_size,) + encoder_sizes[:: -1]
        self.latent_size = latent_size

        # Populating the encoder.
        self.encoder_layers = list()

        for layer_i in range(len(self.encoder_sizes) - 1):
            self.encoder_layers.append(nn.ReLU(nn.Linear(
                in_features=self.encoder_sizes[layer_i],
                out_features=self.encoder_sizes[layer_i + 1])))

        # Populating the decoder.
        self.decoder_layers = list()

        for layer_i in range(len(self.decoder_sizes) - 1):
            self.decoder_layers.append(nn.ReLU(nn.Linear(
                in_features=self.decoder_sizes[layer_i],
                out_features=self.decoder_sizes[layer_i + 1])))

        # Assembling the encoder, decoder and latent space.
        # Dimensions of mu and logvar.
        dims = (self.encoder_sizes[-1], latent_size)

        self.encoder = nn.Sequential(*self.encoder_layers)
        self.mu, self.logvar = nn.Linear(*dims), nn.Linear(*dims)
        self.decoder = nn.Sequential(*self.decoder_layers)

    def z_space(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + std * epsilon

    def sample(self, z):
        return self.decoder(z)

    def forward(self, x):
        x = self.encoder(x)
        mu, logvar = self.mu(x), self.logvar(x)
        z = self.z_space(mu, logvar)
        x = self.decoder(z)

        return x, mu, logvar


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = VAE()
    model.to(device)
    print(model)
