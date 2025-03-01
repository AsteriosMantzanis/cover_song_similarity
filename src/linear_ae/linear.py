import torch.nn as nn


class LinearLayer(nn.Module):
    """Basic Linear Block with Linear, BatchNorm, ReLU, and Dropout"""

    def __init__(self, in_features, out_features, dropout=0.01):
        super(LinearLayer, self).__init__()
        self._validate_features(in_features, out_features)

        self.linear = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        self._validate_input(x, self.linear.in_features)
        x = self.linear(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return self.dropout(x)

    @staticmethod
    def _validate_features(in_features, out_features):
        assert in_features > 0 and out_features > 0, "Feature sizes must be positive."

    @staticmethod
    def _validate_input(x, expected_features):
        assert x.shape[1] == expected_features, (
            f"Expected {expected_features} input features, got {x.shape[1]}"
        )


class EncoderMLP(nn.Module):
    """MLP Encoder using nn.ModuleList()"""

    def __init__(self, input_dim, layer_sizes, latent_dim):
        super(EncoderMLP, self).__init__()
        layers = []
        for out_features in layer_sizes:
            layers.append(LinearLayer(input_dim, out_features))
            input_dim = out_features  # Update for next layer
        layers.append(LinearLayer(input_dim, latent_dim))
        self.encoder = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        return x


class DecoderMLP(nn.Module):
    """MLP Decoder using nn.ModuleList()"""

    def __init__(self, latent_dim, layer_sizes, output_dim):
        super(DecoderMLP, self).__init__()
        layers = []
        input_dim = latent_dim
        for out_features in layer_sizes:
            layers.append(LinearLayer(input_dim, out_features))
            input_dim = out_features  # Update for next layer
        layers.append(LinearLayer(input_dim, output_dim))
        self.decoder = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.decoder:
            x = layer(x)
        return x
