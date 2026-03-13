from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data.dataset import MetagratingDataset
from models.surrogate_mlp import SurrogateMLP


def train_surrogate(
    x_path: str,
    y_path: str,
    output_model: str,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    val_split: float = 0.2,
    seed: int = 42,
    device: str | None = None,
) -> dict[str, float]:
    if epochs <= 0:
        raise ValueError("epochs must be > 0.")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")
    if not 0.0 < val_split < 1.0:
        raise ValueError("val_split must be in (0, 1).")

    torch.manual_seed(seed)
    run_device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MetagratingDataset(x_path=x_path, y_path=y_path)
    n_total = len(dataset)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val
    if n_train <= 0:
        raise ValueError(f"Dataset too small for split: {n_total} samples.")

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = SurrogateMLP().to(run_device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    os.makedirs(os.path.dirname(output_model) or ".", exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        train_count = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(run_device)
            y_batch = y_batch.to(run_device)

            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * x_batch.size(0)
            train_count += x_batch.size(0)

        train_loss = running_train_loss / train_count

        model.eval()
        running_val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(run_device)
                y_batch = y_batch.to(run_device)
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)
                running_val_loss += loss.item() * x_batch.size(0)
                val_count += x_batch.size(0)

        val_loss = running_val_loss / val_count
        print(f"Epoch {epoch:03d}/{epochs} | train_mse={train_loss:.6f} | val_mse={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_model)
            print(f"  Saved improved checkpoint -> {output_model}")

    return {"best_val_loss": best_val_loss}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train surrogate model for metagrating spectra prediction.")
    parser.add_argument("--x-path", default="data/raw/X_inputs.npy", help="Path to geometry input numpy file.")
    parser.add_argument("--y-path", default="data/raw/Y_outputs.npy", help="Path to spectral output numpy file.")
    parser.add_argument(
        "--output-model",
        default="models/best_surrogate.pth",
        help="Checkpoint file path for best validation model.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="Optional torch device override.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    metrics = train_surrogate(
        x_path=args.x_path,
        y_path=args.y_path,
        output_model=args.output_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_split=args.val_split,
        seed=args.seed,
        device=args.device,
    )
    print(f"Training complete. Best validation MSE: {metrics['best_val_loss']:.6f}")


if __name__ == "__main__":
    main()

