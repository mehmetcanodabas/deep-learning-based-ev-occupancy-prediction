def plot_training_history():
    history_file = Path("metrics/lstm_training_history.csv")

    if not history_file.exists():
        raise FileNotFoundError(
            f"Training history not found: {history_file}"
        )

    history = pd.read_csv(history_file)

    required_columns = [
        "Epoch",
        "Train MSE",
        "Validation MSE",
    ]

    missing = [
        col for col in required_columns
        if col not in history.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}\n"
            f"Available columns: {history.columns.tolist()}"
        )

    epochs = history["Epoch"].to_numpy()
    train_loss = history["Train MSE"].to_numpy()
    val_loss = history["Validation MSE"].to_numpy()

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

    ax.plot(
        epochs,
        train_loss,
        color=TRAIN_COLOR,
        linewidth=2.3,
        label="Training Loss",
    )

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

    # Training end / early stopping
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
