import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


class Dataset:
    def __init__(self, csv_path: str, exclude_cols=list):
        # Ensure path is valid
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        if exclude_cols:
            # Load CSV using Pandas
            df.drop(exclude_cols, axis=1, inplace=True)

        # Convert to NumPy (float32 for PyTorch compatibility)
        np_data = df.to_numpy().astype(np.float32)

        # Store as a PyTorch TensorDataset
        self.tensor_data = TensorDataset(torch.from_numpy(np_data))

    def get_loader(self, batch_size: int = 512) -> DataLoader:
        return DataLoader(dataset=self.tensor_data, batch_size=batch_size)
