"""Bayesian hyper-parameter optimization of RPC-Net with Optuna.

Searches learning rate, weight decay, batch size and Adam epsilon with the
TPE sampler; unpromising trials are stopped early by a ``MedianPruner``.
The study is persisted in ``models/optuna_study.db`` (it can be resumed), the
best configuration goes to ``models/best_hyperparameters.json`` and the
interactive reports to ``results/optuna_*.html``.

Usage:
    python -m decoder.tune_hyperparameters [--trials 50]
"""
import argparse
import json
import sys

import optuna
import optuna.visualization as vis
import torch
import torch.nn as nn
import torch.optim as optim
from optuna.trial import TrialState
from torch.utils.data import DataLoader

import config
from decoder.rpc_net import ADAM_BETAS, RPCNet
from decoder.train import load_split

STUDY_NAME = "RPC_Net_Hyperparameter_Optimization"
EPOCHS_PER_TRIAL = 15   # enough to rank configurations

TRAIN_DATASET = None
VAL_DATASET = None


def objective(trial):
    # Apple-silicon GPU if available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    eps = trial.suggest_float("eps", 1e-8, 1e-3, log=True)
    print(f"\n---> Trial {trial.number}: lr={lr:.2e}, wd={weight_decay:.2e}, "
          f"batch={batch_size}, eps={eps:.2e} | device: {device}")

    train_loader = DataLoader(TRAIN_DATASET, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(VAL_DATASET, batch_size=batch_size, shuffle=False)

    model = torch.jit.script(RPCNet().to(device))   # TorchScript speeds up the 24 sub-nets
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay,
                           eps=eps, betas=ADAM_BETAS)

    best_val_loss = float("inf")
    for epoch in range(EPOCHS_PER_TRIAL):
        model.train()
        for batch_emg, batch_ang, batch_target in train_loader:
            batch_emg, batch_ang, batch_target = (batch_emg.to(device), batch_ang.to(device),
                                                  batch_target.to(device))
            optimizer.zero_grad()
            loss = criterion(model(batch_emg, batch_ang), batch_target)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_emg, val_ang, val_target in val_loader:
                preds = model(val_emg.to(device), val_ang.to(device))
                val_loss += criterion(preds, val_target.to(device)).item()
        avg_val_loss = val_loss / len(val_loader)
        best_val_loss = min(best_val_loss, avg_val_loss)

        if (epoch + 1) % 3 == 0 or epoch == 0:
            print(f"     [Trial {trial.number}] Epoch {epoch + 1}/{EPOCHS_PER_TRIAL} | "
                  f"Val loss: {avg_val_loss:.6f}")

        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            print(f"     [!] Trial {trial.number} pruned at epoch {epoch + 1}.")
            raise optuna.exceptions.TrialPruned()

    print(f"---> Trial {trial.number} done | best val loss: {best_val_loss:.6f}")
    return best_val_loss


def save_results(study):
    best = study.best_trial
    with open(config.BEST_HYPERPARAMETERS, "w", encoding="utf-8") as f:
        json.dump({
            "study_name": study.study_name,
            "best_validation_loss": best.value,
            "best_trial_number": best.number,
            "hyperparameters": best.params,
        }, f, indent=4)
    print(f"Best hyper-parameters saved to {config.rel(config.BEST_HYPERPARAMETERS)}")

    reports = {
        "optuna_optimization_history.html": vis.plot_optimization_history,
        "optuna_param_importances.html": vis.plot_param_importances,
        "optuna_slice_plot.html": vis.plot_slice,
    }
    config.RESULTS_DIR.mkdir(exist_ok=True)
    for filename, plot_fn in reports.items():
        plot_fn(study).write_html(config.RESULTS_DIR / filename)
        print(f" - results/{filename}")


def main():
    global TRAIN_DATASET, VAL_DATASET
    parser = argparse.ArgumentParser(description="Optuna search for RPC-Net.")
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()

    train_file = config.TENSORS_DIR / "train_tensors.pt"
    val_file = config.TENSORS_DIR / "val_tensors.pt"
    if not (train_file.exists() and val_file.exists()):
        sys.exit(f"Error: training/validation tensors not found in {config.TENSORS_DIR}")

    print("Loading training and validation tensors...")
    TRAIN_DATASET = load_split(train_file)
    VAL_DATASET = load_split(val_file)

    study = optuna.create_study(
        direction="minimize",
        study_name=STUDY_NAME,
        storage=f"sqlite:///{config.OPTUNA_DB}",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=6),
    )
    try:
        study.optimize(objective, n_trials=args.trials)
    except KeyboardInterrupt:
        print("\nOptimization interrupted: writing partial reports...")

    pruned = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print("\n================== STUDY SUMMARY ==================")
    print(f" Trials run: {len(study.trials)} | completed: {len(complete)} | pruned: {len(pruned)}")
    print(f" Best validation loss: {study.best_trial.value:.6f}")
    print("===================================================")
    save_results(study)


if __name__ == "__main__":
    main()
