from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


DATA_DIR = Path("data/processed")

INPUT_LENGTH = 48
HORIZON = 6
BATCH_SIZE = 64


class OccupancyDataset(Dataset):
    def __init__(
        self,
        values,
        target_start_min,
        target_start_max,
        input_length=48,
        horizon=6,
    ):
        self.values = torch.tensor(
            values, dtype=torch.float32
        )

        self.input_length = input_length
        self.horizon = horizon
        self.n_zones = values.shape[1]

        # Inclusive range of forecast target starts.
        self.origins = np.arange(
            target_start_min,
            target_start_max + 1,
        )

    def __len__(self):
        return len(self.origins) * self.n_zones

    def __getitem__(self, index):
        origin_index = index // self.n_zones
        zone_index = index % self.n_zones

        target_start = int(self.origins[origin_index])

        input_start = target_start - self.input_length
        target_end = target_start + self.horizon

        x = self.values[
            input_start:target_start,
            zone_index,
        ]

        y = self.values[
            target_start:target_end,
            zone_index,
        ]

        # GRU expects [sequence_length, features].
        x = x.unsqueeze(-1)

        return x, y


def main():
    train = pd.read_csv(
        DATA_DIR / "train.csv",
        index_col="time",
        parse_dates=True,
    )

    val = pd.read_csv(
        DATA_DIR / "val.csv",
        index_col="time",
        parse_dates=True,
    )

    assert train.columns.equals(val.columns)
    assert (
        train.index.max() + pd.Timedelta(hours=1)
        == val.index.min()
    )

    # Train-only normalization parameters.
    train_mean = train.mean(axis=0).to_numpy()
    train_std = train.std(
        axis=0, ddof=0
    ).to_numpy()

    # Avoid division by zero for constant zones.
    train_std = np.where(
        train_std < 1e-8, 1.0, train_std
    )

    train_values = (
        train.to_numpy(dtype=np.float64) - train_mean
    ) / train_std

    val_values = (
        val.to_numpy(dtype=np.float64) - train_mean
    ) / train_std

    # Training windows use training data only.
    train_dataset = OccupancyDataset(
        values=train_values,
        target_start_min=INPUT_LENGTH,
        target_start_max=len(train) - HORIZON,
        input_length=INPUT_LENGTH,
        horizon=HORIZON,
    )

    # Validation inputs may use the last 48 hours
    # of training data. Targets stay in validation.
    combined_values = np.concatenate(
        [train_values, val_values],
        axis=0,
    )

    val_dataset = OccupancyDataset(
        values=combined_values,
        target_start_min=len(train),
        target_start_max=len(combined_values) - HORIZON,
        input_length=INPUT_LENGTH,
        horizon=HORIZON,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print("\n=== DATASET SIZES ===")
    print("Zones:", train.shape[1])
    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    print("\n=== FIRST TRAINING SAMPLE ===")
    x, y = train_dataset[0]
    print("Input shape:", tuple(x.shape))
    print("Target shape:", tuple(y.shape))
    print("Input dtype:", x.dtype)

    print("\n=== FIRST TRAINING BATCH ===")
    batch_x, batch_y = next(iter(train_loader))
    print("Batch input shape:", tuple(batch_x.shape))
    print("Batch target shape:", tuple(batch_y.shape))

    print("\n=== NORMALIZATION CHECK ===")
    print(
        "Zone", train.columns[0],
        "normalized train mean:",
        round(float(train_values[:, 0].mean()), 6),
    )
    print(
        "Zone", train.columns[0],
        "normalized train std:",
        round(float(train_values[:, 0].std()), 6),
    )

    print("\n=== VALIDATION ALIGNMENT ===")
    print("Validation forecast origins:", len(val_dataset.origins))
    print("Expected baseline origins:", len(val) - HORIZON + 1)


if __name__ == "__main__":
    main()
