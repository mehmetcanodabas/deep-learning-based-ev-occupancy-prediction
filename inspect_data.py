from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/occupancy.csv")


def main():
    df = pd.read_csv(DATA_PATH)

    print("\n=== DATASET SHAPE ===")
    print(df.shape)

    print("\n=== FIRST 5 ROWS ===")
    print(df.head().to_string())

    print("\n=== COLUMN INFORMATION ===")
    print(f"Total columns: {len(df.columns)}")
    print(f"First 10 columns: {df.columns[:10].tolist()}")

    print("\n=== DATA TYPES ===")
    print(df.dtypes.value_counts())

    print("\n=== MISSING VALUES ===")
    print(f"Total missing values: {df.isna().sum().sum()}")

    print("\n=== OCCUPANCY STATISTICS ===")
    occupancy = df.drop(columns=["time"])
    print(occupancy.stack().describe())

    print("\n=== TIMESTAMP RANGE ===")
    timestamps = pd.to_datetime(df["time"])
    print(f"Start: {timestamps.min()}")
    print(f"End:   {timestamps.max()}")

    print("\n=== TIME INTERVALS ===")
    print(timestamps.diff().value_counts().head())


if __name__ == "__main__":
    main()