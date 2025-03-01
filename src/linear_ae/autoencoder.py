import torch.nn as nn

from .linear import DecoderMLP, EncoderMLP


class Autoencoder(nn.Module):
    """Combining Encoder and Decoder"""

    def __init__(self, input_dim, encoder_sizes, latent_dim, decoder_sizes, output_dim):
        super(Autoencoder, self).__init__()
        self.encoder = EncoderMLP(input_dim, encoder_sizes, latent_dim)
        self.decoder = DecoderMLP(latent_dim, decoder_sizes, output_dim)

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)
