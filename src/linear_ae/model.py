import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from src.linear_ae.autoencoder import Autoencoder


class AutoencoderModel(pl.LightningModule):
    def __init__(self, autoencoder: Autoencoder, learning_rate=1e-3):
        super().__init__()
        self.model = autoencoder
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch[0]
        x_recon = self(x)
        loss = F.mse_loss(x_recon, x)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        x_recon = self(x)
        loss = F.mse_loss(x_recon, x)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Pass input through the encoder to get latent representation."""
        return self.model.encoder(x)
