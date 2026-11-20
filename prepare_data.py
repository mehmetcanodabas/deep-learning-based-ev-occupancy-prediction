from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path("data/occupancy.csv")
OUTPUT_DIR = Path("data/processed")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
N_ZONES = 30


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load and validate the dataset.
    df = pd.read_csv(DATA_PATH, parse_dates=["time"])
    df = df.set_index("time").sort_index()

    assert not df.index.has_duplicates, "Duplicate timestamps found."
    assert not df.isna().any().any(), "Missing values found."
    assert df.index.to_series().diff().dropna().eq(
        pd.Timedelta(hours=1)
    ).all(), "Time intervals must be hourly."

    # Chronological split boundaries.
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    # Select zones using training data only.
    train_all = df.iloc[:train_end]

    zone_means = train_all.mean().sort_values()
    ranked_zones = zone_means.index.to_list()

    selected_indices = np.linspace(
        0, len(ranked_zones) - 1, N_ZONES, dtype=int
    )
    selected_zones = [ranked_zones[i] for i in selected_indices]

    # Keep the same columns across all splits.
    selected = df[selected_zones]

    train = selected.iloc[:train_end]
    val = selected.iloc[train_end:val_end]
    test = selected.iloc[val_end:]

    # Save splits.
    train.to_csv(OUTPUT_DIR / "train.csv")
    val.to_csv(OUTPUT_DIR / "val.csv")
    test.to_csv(OUTPUT_DIR / "test.csv")

    # Save zone metadata.
    zone_info = pd.DataFrame({
        "zone_id": selected_zones,
        "train_mean": zone_means.loc[selected_zones].values,
        "train_std": train_all[selected_zones].std().values,
    })

    zone_info.to_csv(
        OUTPUT_DIR / "selected_zones.csv",
        index=False
    )

    print("\n=== SELECTED ZONES ===")
    print(zone_info.round(2).to_string(index=False))

    print("\n=== CHRONOLOGICAL SPLITS ===")

    for name, split in [
        ("Train", train),
        ("Validation", val),
        ("Test", test),
    ]:
        print(
            f"{name}: {split.shape} | "
            f"{split.index.min()} -> {split.index.max()}"
        )

    print("\nSaved files to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
