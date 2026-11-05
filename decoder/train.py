"""Train RPC-Net on the tensors produced by ``decoder.feature_extraction``.

Hyper-parameters are read from ``models/best_hyperparameters.json`` (written
by ``decoder.tune_hyperparameters``); if the file is missing, the values of
the original paper are used. The checkpoint with the lowest validation MSE is
saved to ``models/rpc_net_weights.pth`` and the learning curve to
``results/figures/loss_convergence.png``.

Usage:
    python -m decoder.train [--epochs 50]
"""
import argparse
import json

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import config
from decoder.rpc_net import ADAM_BETAS, PAPER_HYPERPARAMETERS, RPCNet


def load_split(path):
    data = torch.load(path, weights_only=False)
    return TensorDataset(data["X_emg"], data["X_ang"], data["Y_target"])


def load_hyperparameters():
    hps = dict(PAPER_HYPERPARAMETERS)
    if config.BEST_HYPERPARAMETERS.exists():
        print(f"Loading tuned hyper-parameters from '{config.BEST_HYPERPARAMETERS.name}'...")
        with open(config.BEST_HYPERPARAMETERS, encoding="utf-8") as f:
            hps.update(json.load(f).get("hyperparameters", {}))
    else:
        print("Hyper-parameter file not found: using the RPC-Net paper defaults.")
    hps["batch_size"] = int(hps["batch_size"])
    return hps


def train_model(train_file, val_file, epochs=50):
    print(f"Loading training tensors from {train_file.name}...")
    train_dataset = load_split(train_file)
    print(f"Loading validation tensors from {val_file.name}...")
    val_dataset = load_split(val_file)

    hps = load_hyperparameters()
    train_loader = DataLoader(train_dataset, batch_size=hps["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hps["batch_size"], shuffle=False)

    model = RPCNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=hps["lr"], eps=hps["eps"],
                           weight_decay=hps["weight_decay"], betas=ADAM_BETAS)

    print(f"Training for {epochs} epochs...")
    best_val_loss = float("inf")
    train_losses, val_losses = [], []
    config.MODELS_DIR.mkdir(exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_emg, batch_ang, batch_target in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_emg, batch_ang), batch_target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_emg, val_ang, val_target in val_loader:
                val_loss += criterion(model(val_emg, val_ang), val_target).item()
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch + 1}/{epochs} | Train loss: {avg_train_loss:.6f} | Val loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config.RPC_NET_WEIGHTS)
            print(f"  --> New best model saved (val loss: {best_val_loss:.6f})")

    print(f"Training complete. Best weights: {config.rel(config.RPC_NET_WEIGHTS)}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label="Train loss", color="blue", marker="o")
    plt.plot(range(1, epochs + 1), val_losses, label="Validation loss", color="red", marker="x")
    plt.title("Loss convergence (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = config.FIGURES_DIR / "loss_convergence.png"
    plt.savefig(plot_path)
    print(f"Learning curve saved to {config.rel(plot_path)}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Train RPC-Net.")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    train_model(config.TENSORS_DIR / "train_tensors.pt", config.TENSORS_DIR / "val_tensors.pt",
                epochs=args.epochs)


if __name__ == "__main__":
    main()
