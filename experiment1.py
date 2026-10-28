"""Deney 1 – Temel MLP modeli.

Çalıştırma:
    python experiment1.py
    python experiment1.py --epochs 20 --lr 0.001 --hidden 256 128
"""
import argparse
import json
import os

import torch

from src.plots import plot_history
from src.train import train_and_evaluate

RESULTS = "results"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--optimizer", default="adam", choices=["adam", "sgd"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--val-size", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    config = {
        "hidden_sizes": args.hidden,
        "dropout": args.dropout,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "val_size": args.val_size,
        "seed": args.seed,
    }

    model, history, summary = train_and_evaluate(config)

    os.makedirs(RESULTS, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": config}, os.path.join(RESULTS, "mlp_baseline.pt"))
    with open(os.path.join(RESULTS, "experiment1_summary.json"), "w") as f:
        json.dump({"summary": summary, "history": history}, f, indent=2)
    plot_history(history, os.path.join(RESULTS, "experiment1_curves.png"))
    print(f"\nModel, özet ve grafikler '{RESULTS}/' klasörüne kaydedildi.")


if __name__ == "__main__":
    main()
