from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model import OccupancyLSTM
from prepare_torch_data import OccupancyDataset


DATA_DIR = Path("data/processed")
METRICS_DIR = Path("metrics")
MODEL_PATH = Path("models/lstm.pt")

BATCH_SIZE = 64


def calculate_metrics(y_true, y_pred):
    errors = y_true - y_pred

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    denominator = np.sum(np.abs(y_true))
    wape = (
        np.sum(np.abs(errors)) / denominator * 100
        if denominator > 0
        else np.nan
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "WAPE (%)": float(wape),
    }


def load_data():
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

    test = pd.read_csv(
        DATA_DIR / "test.csv",
        index_col="time",
        parse_dates=True,
    )

    assert train.columns.equals(val.columns)
    assert train.columns.equals(test.columns)

    assert (
        train.index[-1] + pd.Timedelta(hours=1)
        == val.index[0]
    )

    assert (
        val.index[-1] + pd.Timedelta(hours=1)
        == test.index[0]
    )

    for df in (train, val, test):
        assert not df.isna().any().any()

    return train, val, test


def load_model():
    # Load only our own trusted training checkpoint.
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model = OccupancyLSTM(
        input_size=1,
        hidden_size=checkpoint["hidden_size"],
        horizon=checkpoint["horizon"],
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()

    return model, checkpoint


def generate_predictions(
    model,
    checkpoint,
    historical_data,
    target_data,
):
    train_mean = np.asarray(
        checkpoint["train_mean"],
        dtype=np.float64,
    )
    train_std = np.asarray(
        checkpoint["train_std"],
        dtype=np.float64,
    )

    input_length = checkpoint["input_length"]
    horizon = checkpoint["horizon"]
    zone_ids = [
        str(z) for z in checkpoint["zone_ids"]
    ]

    combined = pd.concat(
        [historical_data, target_data]
    )

    assert combined.columns.tolist() == zone_ids
    assert len(train_mean) == len(zone_ids)
    assert len(train_std) == len(zone_ids)
    assert np.all(train_std > 0)

    values = combined.to_numpy(dtype=np.float64)

    # Reuse training-only normalization.
    normalized = (
        values - train_mean
    ) / train_std

    first_origin = len(historical_data)
    last_origin = len(combined) - horizon

    dataset = OccupancyDataset(
        values=normalized,
        target_start_min=first_origin,
        target_start_max=last_origin,
        input_length=input_length,
        horizon=horizon,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    predictions = []

    with torch.no_grad():
        for x, _ in loader:
            predictions.append(model(x).numpy())

    n_origins = len(dataset.origins)
    n_zones = len(zone_ids)

    # Dataset order: origin, zone, horizon.
    y_pred = np.concatenate(
        predictions, axis=0
    ).reshape(
        n_origins, n_zones, horizon
    )

    # Shape: origin, horizon, zone.
    y_pred = y_pred.transpose(0, 2, 1)

    # Inverse normalization.
    y_pred = (
        y_pred.astype(np.float64)
        * train_std[None, None, :]
        + train_mean[None, None, :]
    )

    # Occupancy cannot be negative.
    y_pred = np.maximum(y_pred, 0.0)

    # Actual occupancy in original units.
    y_true = np.stack([
        values[start:start + horizon]
        for start in dataset.origins
    ])

    assert y_true.shape == y_pred.shape

    return (
        y_true,
        y_pred,
        dataset.origins,
        combined.index,
        zone_ids,
    )


def build_horizon_report(y_true, y_pred):
    rows = []

    for h in range(y_true.shape[1]):
        rows.append({
            "Horizon": h + 1,
            **calculate_metrics(
                y_true[:, h, :],
                y_pred[:, h, :],
            ),
        })

    return pd.DataFrame(rows)


def build_zone_report(y_true, y_pred, zone_ids):
    rows = []

    for z, zone_id in enumerate(zone_ids):
        actual = y_true[:, :, z]
        predicted = y_pred[:, :, z]

        rows.append({
            "Zone": zone_id,
            "Actual Mean": float(actual.mean()),
            "Predicted Mean": float(predicted.mean()),
            **calculate_metrics(
                actual, predicted
            ),
        })

    return pd.DataFrame(rows).sort_values(
        "WAPE (%)",
        ascending=True,
        na_position="last",
    ).reset_index(drop=True)


def build_prediction_report(
    y_true,
    y_pred,
    origins,
    time_index,
    zone_ids,
):
    rows = []

    n_origins, horizon, n_zones = y_true.shape

    for i in range(n_origins):
        forecast_start = time_index[origins[i]]

        for h in range(horizon):
            target_time = time_index[origins[i] + h]

            for z in range(n_zones):
                actual = y_true[i, h, z]
                predicted = y_pred[i, h, z]

                rows.append({
                    "Zone": zone_ids[z],
                    "Forecast Start": forecast_start,
                    "Target Time": target_time,
                    "Horizon": h + 1,
                    "Actual": actual,
                    "LSTM Prediction": predicted,
                    "Absolute Error": abs(
                        actual - predicted
                    ),
                })

    return pd.DataFrame(rows)


def evaluate_split(
    split_name,
    model,
    checkpoint,
    historical_data,
    target_data,
):
    output_dir = METRICS_DIR / f"lstm_{split_name}"
    output_dir.mkdir(
        parents=True, exist_ok=True
    )

    (
        y_true,
        y_pred,
        origins,
        time_index,
        zone_ids,
    ) = generate_predictions(
        model,
        checkpoint,
        historical_data,
        target_data,
    )

    overall_df = pd.DataFrame([{
        "Model": "LSTM",
        **calculate_metrics(y_true, y_pred),
    }])

    horizons_df = build_horizon_report(
        y_true, y_pred
    )

    zones_df = build_zone_report(
        y_true, y_pred, zone_ids
    )

    predictions_df = build_prediction_report(
        y_true,
        y_pred,
        origins,
        time_index,
        zone_ids,
    )

    overall_df.to_csv(
        output_dir / "overall.csv",
        index=False,
    )

    horizons_df.to_csv(
        output_dir / "horizons.csv",
        index=False,
    )

    zones_df.to_csv(
        output_dir / "zones.csv",
        index=False,
    )

    predictions_df.to_csv(
        output_dir / "predictions.csv",
        index=False,
    )

    print(f"\n=== {split_name.upper()} RESULTS ===")
    print(overall_df.round(4).to_string(index=False))

    print(f"\n=== {split_name.upper()} HORIZONS ===")
    print(horizons_df.round(4).to_string(index=False))

    print(f"\n=== {split_name.upper()} ZONES ===")
    print(zones_df.round(4).to_string(index=False))

    print(f"\n=== {split_name.upper()} SUMMARY ===")
    print("Forecast origins:", len(origins))
    print("Zones:", len(zone_ids))
    print("Horizon:", y_true.shape[1])
    print("Prediction rows:", len(predictions_df))
    print("Saved to:", output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LSTM occupancy forecasts."
    )

    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        required=True,
    )

    args = parser.parse_args()

    train, val, test = load_data()
    model, checkpoint = load_model()

    if args.split == "validation":
        historical_data = train
        target_data = val
    else:
        historical_data = pd.concat([train, val])
        target_data = test

    evaluate_split(
        split_name=args.split,
        model=model,
        checkpoint=checkpoint,
        historical_data=historical_data,
        target_data=target_data,
    )


if __name__ == "__main__":
    main()
