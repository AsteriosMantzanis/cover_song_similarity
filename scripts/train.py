import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.linear_ae.autoencoder import Autoencoder
from src.linear_ae.dataset import Dataset
from src.linear_ae.model import AutoencoderModel

# Paths
TRAIN_DATASET_PATH = "data/cover_song_features_expanded.csv"
VAL_DATASET_PATH = "data/cover_benchmark_expanded.csv"
SAVE_DIR = "checkpoints"

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MAX_EPOCHS = 500

# Define architecture
input_dim = 52
encoder_sizes = [64, 128, 256]
latent_dim = 512
decoder_sizes = [256, 128, 64]
output_dim = 52


def train():
    """Train the autoencoder model with PyTorch Lightning."""
    os.makedirs(SAVE_DIR, exist_ok=True)  # Ensure save directory exists

    # Load datasets & dataloaders
    train_dataset = Dataset(TRAIN_DATASET_PATH, exclude_cols=["work", "performance"])
    val_dataset = Dataset(VAL_DATASET_PATH, exclude_cols=["work", "performance"])

    train_dataloader = train_dataset.get_loader(batch_size=BATCH_SIZE)
    val_dataloader = val_dataset.get_loader(batch_size=BATCH_SIZE)

    # Initialize model
    autoencoder = Autoencoder(
        input_dim, encoder_sizes, latent_dim, decoder_sizes, output_dim
    )
    model = AutoencoderModel(autoencoder, learning_rate=LEARNING_RATE)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=SAVE_DIR,
        filename="linear_expanded-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",  # Auto-detect CUDA/CPU
        callbacks=[checkpoint_callback, early_stopping],
    )

    # Train model
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    train()
