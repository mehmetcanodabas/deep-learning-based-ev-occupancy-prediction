from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PREDICTIONS_FILE = Path("metrics/lstm_test/predictions.csv")
TRAINING_HISTORY_FILE = Path(
    "metrics/lstm_training_history.csv"
)

OUTPUT_DIR = Path("docs/results")

FORECAST_START = "2023-02-15 13:00:00"
MIN_OCCUPANCY_RANGE = 2
NUMBER_OF_ZONES = 5

ACTUAL_COLOR = "#5B8FBE"
PREDICTED_COLOR = "#299579"
BAR_COLOR = "#299579"

TRAIN_COLOR = "#5B8FBE"
VALIDATION_COLOR = "#299579"

TITLE_COLOR = "#203A36"
SUBTITLE_COLOR = "#547269"


# --------------------------------------------------
# Load and select forecast data
# --------------------------------------------------

def load_case_study_data():
    df = pd.read_csv(PREDICTIONS_FILE)

    selected = df[
        df["Forecast Start"] == FORECAST_START
    ].copy()

    if selected.empty:
        raise ValueError(
            "Selected forecast window not found."
        )

    selected["Target Time"] = pd.to_datetime(
        selected["Target Time"]
    )

    results = (
        selected.groupby("Zone")
        .agg(
            Actual_Mean=("Actual", "mean"),
            Predicted_Mean=("LSTM Prediction", "mean"),
            Actual_Min=("Actual", "min"),
            Actual_Max=("Actual", "max"),
            Actual_Sum=("Actual", "sum"),
            Absolute_Error_Sum=("Absolute Error", "sum"),
            Hours=("Horizon", "nunique"),
        )
        .reset_index()
    )

    if len(results) != 30 or not results["Hours"].eq(6).all():
        raise ValueError(
            "Expected 30 zones with 6 forecast hours each."
        )

    results["Occupancy_Range"] = (
        results["Actual_Max"] - results["Actual_Min"]
    )

    results["WAPE"] = np.where(
        results["Actual_Sum"] > 0,
        100
        * results["Absolute_Error_Sum"]
        / results["Actual_Sum"],
        np.nan,
    )

    eligible = results[
        (results["Occupancy_Range"] >= MIN_OCCUPANCY_RANGE)
        & results["WAPE"].notna()
    ].copy()

    top5 = (
        eligible.sort_values("WAPE")
        .head(NUMBER_OF_ZONES)
        .reset_index(drop=True)
    )

    if len(top5) != NUMBER_OF_ZONES:
        raise ValueError(
            "Not enough eligible zones."
        )

    top_zones = top5["Zone"].tolist()

    selected_top5 = selected[
        selected["Zone"].isin(top_zones)
    ].copy()

    for zone in top_zones:
        zone_data = selected_top5[
            selected_top5["Zone"] == zone
        ]

        if (
            len(zone_data) != 6
            or sorted(zone_data["Horizon"].tolist())
            != [1, 2, 3, 4, 5, 6]
        ):
            raise ValueError(
                f"Zone {zone}: Expected horizons 1-6."
            )

    return top5, selected_top5


# --------------------------------------------------
# Figure 1: Five-Zone Case Study
# --------------------------------------------------

def plot_five_zone_case_study(top5):
    fig = plt.figure(
        figsize=(14, 8),
        facecolor="white",
    )

    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[3.2, 1.7],
        left=0.08,
        right=0.96,
        top=0.79,
        bottom=0.14,
        hspace=0.20,
    )

    ax = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])

    # Main title
    fig.text(
        0.5,
        0.955,
        "6-Hour EV Charging Point Occupancy Forecast",
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
        color=TITLE_COLOR,
    )

    fig.text(
        0.5,
        0.910,
        "Five-Zone Application Case Study",
        ha="center",
        va="top",
        fontsize=13,
        color=SUBTITLE_COLOR,
    )

    fig.text(
        0.5,
        0.858,
        "Forecast date: 15 February 2023"
        "   |   Forecast hours: 13:00-18:00"
        "   |   Forecast horizon: 6 hours",
        ha="center",
        va="top",
        fontsize=10.5,
        color="#4B5563",
        bbox={
            "boxstyle": "round,pad=0.65",
            "facecolor": "#F0F5F2",
            "edgecolor": "none",
        },
    )

    # WAPE bars
    zone_labels = [
        f"Zone {int(zone)}"
        for zone in top5["Zone"]
    ]

    wape_values = top5["WAPE"].to_numpy()
    x = np.arange(len(top5))

    bars = ax.bar(
        x,
        wape_values,
        width=0.56,
        color=BAR_COLOR,
    )

    for bar, value in zip(bars, wape_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.18,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color=TITLE_COLOR,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        zone_labels,
        fontsize=11,
    )

    ax.set_ylabel(
        "WAPE (%)",
        fontsize=11,
    )

    ax.set_ylim(
        0,
        max(wape_values) * 1.22,
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.18,
    )

    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Summary table
    ax_table.axis("off")

    table_rows = []

    for _, row in top5.iterrows():
        table_rows.append([
            str(int(row["Zone"])),
            f'{row["Actual_Mean"]:.2f}',
            f'{row["Predicted_Mean"]:.2f}',
            f'{row["WAPE"]:.2f}%',
        ])

    table = ax_table.table(
        cellText=table_rows,
        colLabels=[
            "Zone",
            "Avg. Actual Occupied\nCharging Points (6h)",
            "Avg. Predicted Occupied\nCharging Points (6h)",
            "WAPE",
        ],
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.12, 0.34, 0.34, 0.20],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.65)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D7E2DB")
        cell.set_linewidth(0.8)

        if row == 0:
            cell.set_facecolor("#DFEFE7")
            cell.set_text_props(
                fontweight="bold",
                color=TITLE_COLOR,
            )
            cell.set_height(
                cell.get_height() * 1.35
            )

        elif row % 2 == 0:
            cell.set_facecolor("#F5F8F6")

        else:
            cell.set_facecolor("#FFFFFF")

    # Zone-selection note
    fig.text(
        0.5,
        0.055,
        "Zone Selection: The five zones with the lowest "
        "WAPE among those showing a change of at least "
        "2 occupied charging points during the "
        "selected 6-hour period.",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#666666",
    )

    output_path = OUTPUT_DIR / "five_zone_case_study.png"

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)
    print(f"Saved: {output_path}")


# --------------------------------------------------
# Figure 2: Actual vs Predicted
# --------------------------------------------------

def plot_actual_vs_predicted(top5, selected_top5):
    top_zones = top5["Zone"].tolist()

    fig, axes = plt.subplots(
        nrows=5,
        ncols=1,
        figsize=(13, 15),
        sharex=True,
        facecolor="white",
    )

    fig.subplots_adjust(
        left=0.13,
        right=0.94,
        top=0.87,
        bottom=0.14,
        hspace=0.48,
    )

    # Titles
    fig.text(
        0.5,
        0.965,
        "6-Hour EV Charging Point Occupancy Forecast",
        ha="center",
        va="top",
        fontsize=19,
        fontweight="bold",
        color=TITLE_COLOR,
    )

    fig.text(
        0.5,
        0.935,
        "Five-Zone Application Case Study | Actual vs Predicted",
        ha="center",
        va="top",
        fontsize=13,
        color=SUBTITLE_COLOR,
    )

    fig.text(
        0.5,
        0.898,
        "15 February 2023  |  Forecast hours: 13:00-18:00",
        ha="center",
        va="top",
        fontsize=11,
        color="#4B5563",
    )

    max_value = max(
        selected_top5["Actual"].max(),
        selected_top5["LSTM Prediction"].max(),
    )

    x_limit = max_value * 1.18

    for ax, zone in zip(axes, top_zones):
        zone_data = (
            selected_top5[
                selected_top5["Zone"] == zone
            ]
            .sort_values("Horizon")
        )

        actual = zone_data["Actual"].to_numpy()
        predicted = (
            zone_data["LSTM Prediction"].to_numpy()
        )

        hour_labels = (
            zone_data["Target Time"]
            .dt.strftime("%H:%M")
            .tolist()
        )

        y = np.arange(6)
        bar_height = 0.34

        actual_bars = ax.barh(
            y + bar_height / 2,
            actual,
            height=bar_height,
            color=ACTUAL_COLOR,
            label="Actual Occupancy",
        )

        predicted_bars = ax.barh(
            y - bar_height / 2,
            predicted,
            height=bar_height,
            color=PREDICTED_COLOR,
            label="Predicted Occupancy",
        )

        # Actual labels
        for bar, value in zip(actual_bars, actual):
            ax.text(
                value + max_value * 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.0f}",
                va="center",
                ha="left",
                fontsize=9,
                color="#263238",
            )

        # Predicted labels
        for bar, value in zip(predicted_bars, predicted):
            ax.text(
                value + max_value * 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                va="center",
                ha="left",
                fontsize=9,
                color="#263238",
            )

        ax.set_yticks(y)
        ax.set_yticklabels(
            hour_labels,
            fontsize=10,
        )

        ax.invert_yaxis()

        # WAPE for this six-hour forecast
        actual_sum = actual.sum()
        absolute_error = np.abs(
            actual - predicted
        ).sum()

        wape = (
            100 * absolute_error / actual_sum
            if actual_sum > 0
            else float("nan")
        )

        ax.set_title(
            f"Zone {int(zone)}  |  WAPE: {wape:.2f}%",
            fontsize=12,
            fontweight="bold",
            loc="left",
            pad=10,
        )

        ax.set_xlim(0, x_limit)

        ax.grid(
            axis="x",
            linestyle="--",
            alpha=0.15,
        )

        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.tick_params(
            axis="y",
            length=0,
        )

    # X-axis label
    axes[-1].set_xlabel(
        "Occupied Charging Points",
        fontsize=12,
        labelpad=18,
    )

    # Legend: two rows at bottom-right.
    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.05),
        ncol=1,
        frameon=False,
        fontsize=11,
    )

    output_path = OUTPUT_DIR / "actual_vs_predicted.png"

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)
    print(f"Saved: {output_path}")


# --------------------------------------------------
# Figure 3: LSTM Training & Validation Loss
# --------------------------------------------------

def plot_training_history():
    if not TRAINING_HISTORY_FILE.exists():
        raise FileNotFoundError(
            f"Training history not found: "
            f"{TRAINING_HISTORY_FILE}"
        )

    history = pd.read_csv(TRAINING_HISTORY_FILE)

    # Accept common training-history column names.
    normalized_columns = {
        str(col).lower().strip().replace(" ", "_"): col
        for col in history.columns
    }

    def find_column(candidates):
        for candidate in candidates:
            if candidate in normalized_columns:
                return normalized_columns[candidate]

        return None

    epoch_col = find_column([
        "epoch",
        "epochs",
    ])

    train_col = find_column([
        "train_loss",
        "training_loss",
        "train_mse",
        "training_mse",
        "train_normalized_mse",
    ])

    val_col = find_column([
        "val_loss",
        "validation_loss",
        "val_mse",
        "validation_mse",
        "val_normalized_mse",
    ])

    if train_col is None or val_col is None:
        raise ValueError(
            "Could not identify train/validation loss "
            "columns in training history.\n"
            f"Available columns: {history.columns.tolist()}"
        )

    if epoch_col is not None:
        epochs = history[epoch_col].to_numpy()
    else:
        epochs = np.arange(1, len(history) + 1)

    train_loss = pd.to_numeric(
        history[train_col],
        errors="raise",
    ).to_numpy()

    val_loss = pd.to_numeric(
        history[val_col],
        errors="raise",
    ).to_numpy()

    if (
        len(history) == 0
        or not np.isfinite(train_loss).all()
        or not np.isfinite(val_loss).all()
    ):
        raise ValueError(
            "Training history contains invalid loss values."
        )

    # Best validation checkpoint
    best_index = int(np.argmin(val_loss))
    best_epoch = int(epochs[best_index])
    best_val_loss = float(val_loss[best_index])

    final_epoch = int(epochs[-1])

    # --------------------------------------------------
    # Create figure
    # --------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 6.5),
        facecolor="white",
    )

    fig.subplots_adjust(
        left=0.12,
        right=0.94,
        top=0.80,
        bottom=0.15,
    )

    fig.text(
        0.5,
        0.95,
        "LSTM Training & Validation Loss",
        ha="center",
        va="top",
        fontsize=19,
        fontweight="bold",
        color=TITLE_COLOR,
    )

    fig.text(
        0.5,
        0.89,
        "Normalized Mean Squared Error (MSE)",
        ha="center",
        va="top",
        fontsize=12,
        color=SUBTITLE_COLOR,
    )

    # Training loss
    ax.plot(
        epochs,
        train_loss,
        color=TRAIN_COLOR,
        linewidth=2.3,
        label="Training Loss",
    )

    # Validation loss
    ax.plot(
        epochs,
        val_loss,
        color=VALIDATION_COLOR,
        linewidth=2.3,
        label="Validation Loss",
    )

    # Best validation epoch
    ax.axvline(
        best_epoch,
        color="#D97736",
        linestyle="--",
        linewidth=1.5,
        label=f"Best Epoch: {best_epoch}",
    )

    ax.scatter(
        [best_epoch],
        [best_val_loss],
        color="#D97736",
        s=65,
        zorder=5,
    )

    ax.annotate(
        f"Best Val MSE: {best_val_loss:.4f}",
        xy=(best_epoch, best_val_loss),
        xytext=(10, 18),
        textcoords="offset points",
        fontsize=10,
        color="#A85B25",
        fontweight="bold",
    )

    # Last epoch / early stopping
    ax.axvline(
        final_epoch,
        color="#888888",
        linestyle=":",
        linewidth=1.4,
        label=f"Training End: Epoch {final_epoch}",
    )

    ax.set_xlabel(
        "Epoch",
        fontsize=12,
        labelpad=10,
    )

    ax.set_ylabel(
        "Normalized MSE Loss",
        fontsize=12,
        labelpad=10,
    )

    ax.set_xlim(
        min(epochs),
        max(epochs) + 0.8,
    )

    ax.set_ylim(bottom=0)

    ax.grid(
        linestyle="--",
        alpha=0.20,
    )

    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=10,
    )

    output_path = OUTPUT_DIR / "training_history.png"

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation MSE: {best_val_loss:.6f}")
    print(f"Final training epoch: {final_epoch}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    top5, selected_top5 = load_case_study_data()

    print("\n=== FIVE-ZONE APPLICATION CASE STUDY ===")
    print("Date: 15 February 2023")
    print("Forecast hours: 13:00-18:00")

    print("\nSelected zones:")
    print(
        top5[
            [
                "Zone",
                "Actual_Mean",
                "Predicted_Mean",
                "Occupancy_Range",
                "WAPE",
            ]
        ]
        .round(2)
        .to_string(index=False)
    )

    print("\nGenerating figures...\n")

    plot_five_zone_case_study(top5)
    plot_actual_vs_predicted(top5, selected_top5)
    plot_training_history()

    print("\nAll three figures generated successfully.")


if __name__ == "__main__":
    main()
