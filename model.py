import torch
import torch.nn as nn


class OccupancyLSTM(nn.Module):
    def __init__(
        self,
        input_size=1,
        hidden_size=32,
        horizon=6,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.fc = nn.Linear(
            in_features=hidden_size,
            out_features=horizon,
        )

    def forward(self, x):
        # x: [batch_size, 48, 1]

        _, (hidden, cell) = self.lstm(x)

        # Last hidden state: [batch_size, 32]
        last_hidden = hidden[-1]

        # Predict the next 6 hours directly.
        predictions = self.fc(last_hidden)

        return predictions


if __name__ == "__main__":
    model = OccupancyLSTM()

    x = torch.randn(64, 48, 1)

    with torch.no_grad():
        predictions = model(x)

    parameter_count = sum(
        p.numel() for p in model.parameters()
    )

    print("\n=== LSTM MODEL ===")
    print(model)

    print("\n=== TENSOR SHAPES ===")
    print("Input:", tuple(x.shape))
    print("Output:", tuple(predictions.shape))

    print("\n=== PARAMETERS ===")
    print("Total trainable parameters:", parameter_count)
