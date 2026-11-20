from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_PATH = Path("data/occupancy.csv")
FIGURES_DIR = Path("figures")

TRAIN_RATIO = 0.70


def main():
    FIGURES_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")

    # Use training data only for exploratory zone selection.
    train_end = int(len(df) * TRAIN_RATIO)
    train = df.iloc[:train_end]

    print("\n=== TRAINING PERIOD ===")
    print(f"Start: {train.index.min()}")
    print(f"End:   {train.index.max()}")
    print(f"Hours: {len(train)}")

    # Calculate zone-level statistics.
    stats = pd.DataFrame({
        "mean": train.mean(),
        "std": train.std(),
        "min": train.min(),
        "max": train.max(),
        "zero_pct": (train == 0).mean() * 100,
    })

    print("\n=== ZONE STATISTICS ===")
    print(stats.describe().round(2).to_string())

    print("\n=== TOP 10 ZONES BY MEAN OCCUPANCY ===")
    print(
        stats.sort_values("mean", ascending=False)
        .head(10)
        .round(2)
        .to_string()
    )

    print("\n=== TOP 10 ZONES BY STANDARD DEVIATION ===")
    print(
        stats.sort_values("std", ascending=False)
        .head(10)
        .round(2)
        .to_string()
    )

    # Plot the distribution of zone mean occupancy.
    plt.figure(figsize=(9, 5))
    plt.hist(stats["mean"], bins=30, edgecolor="black")
    plt.xlabel("Mean occupancy")
    plt.ylabel("Number of zones")
    plt.title("Distribution of Zone Mean Occupancy (Train)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "zone_mean_distribution.png", dpi=150)
    plt.close()

    # Choose three zones at different activity levels.
    ranked_zones = stats.sort_values("mean").index

    selected_zones = [
        ranked_zones[len(ranked_zones) // 4],
        ranked_zones[len(ranked_zones) // 2],
        ranked_zones[3 * len(ranked_zones) // 4],
    ]

    print("\n=== EXAMPLE ZONES ===")
    print(stats.loc[selected_zones].round(2).to_string())

    # Plot their first seven days of training observations.
    sample = train[selected_zones].iloc[:168]

    fig, axes = plt.subplots(
        3, 1, figsize=(12, 8), sharex=True
    )

    for ax, zone in zip(axes, selected_zones):
        ax.plot(sample.index, sample[zone])
        ax.set_title(f"Zone {zone}")
        ax.set_ylabel("Occupancy")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Time")
    fig.suptitle("Hourly Occupancy: First 7 Training Days")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "zone_time_series.png", dpi=150)
    plt.close()

    print("\nFigures saved in:", FIGURES_DIR.resolve())


if __name__ == "__main__":
    main()
