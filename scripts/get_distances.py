import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from src.linear_ae.autoencoder import Autoencoder
from src.linear_ae.model import AutoencoderModel

# Load saved model
input_dim = 52
encoder_sizes = [64, 128, 256]
latent_dim = 512
decoder_sizes = [256, 128, 64]
output_dim = 52
MODEL_PATH = "checkpoints\linear_expanded-epoch=112-val_loss=168.62.ckpt"

autoencoder = Autoencoder(
    input_dim, encoder_sizes, latent_dim, decoder_sizes, output_dim
)
model = AutoencoderModel.load_from_checkpoint(MODEL_PATH, autoencoder=autoencoder)
model.eval()

# Load dataset
df = pd.read_csv("data/cover_benchmark_expanded.csv")


def compute_distances(df, model):
    """Computes in-clique and cross-clique distances with sampling."""
    selected_work = (
        df["work"].value_counts().index[0]
    )  # get the work with the highest value counts
    clique_df = df[df["work"] == selected_work]
    other_df = df[
        df["work"] != selected_work
    ]  # and compare it with N samples out oh its clicque
    sample_size = clique_df.shape[0]  # where N = the value counts of the selected work

    # Sampling
    other_df = other_df.sample(sample_size)

    clique_embeds = model.encode(
        torch.tensor(
            clique_df.drop(columns=["work", "performance"]).values, dtype=torch.float32
        )
    )
    other_embeds = model.encode(
        torch.tensor(
            other_df.drop(columns=["work", "performance"]).values, dtype=torch.float32
        )
    )

    # Compute pairwise distances
    in_clique_distances = []
    cross_clique_distances = []

    if len(clique_embeds) > 1:
        in_clique_distances.extend(torch.cdist(clique_embeds, clique_embeds).flatten())

    if len(clique_embeds) > 0 and len(other_embeds) > 0:
        cross_clique_distances.extend(
            torch.cdist(clique_embeds, other_embeds).flatten()
        )

    return in_clique_distances, cross_clique_distances, selected_work


# Compute distances
in_clique, cross_clique, selected_work = compute_distances(df, model)

# Ensure distances are tensors before plotting
in_clique_tensor = torch.tensor(in_clique, dtype=torch.float32)
cross_clique_tensor = torch.tensor(cross_clique, dtype=torch.float32)

# Convert to NumPy arrays
in_clique_np = in_clique_tensor.detach().cpu().numpy()
cross_clique_np = cross_clique_tensor.detach().cpu().numpy()

# Plot
# Set seaborn style
sns.set_style("whitegrid")
# Create figure
plt.figure(figsize=(10, 6))
# Plot histograms with transparency and KDE
sns.histplot(
    in_clique_np,
    color="royalblue",
    label="In-Clique Distance",
    bins=20,
    kde=True,
    alpha=0.7,
    edgecolor="black",
)
sns.histplot(
    cross_clique_np,
    color="darkorange",
    label="Cross-Clique Distance",
    bins=20,
    kde=True,
    alpha=0.7,
    edgecolor="black",
)
# Labels and title
plt.legend(fontsize=12)
plt.xlabel("Distance", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.title(f"Histogram of Distances for Work: {selected_work}", fontsize=16)
# Remove top and right spines for a cleaner look
sns.despine()
# Show plot
plt.show()
