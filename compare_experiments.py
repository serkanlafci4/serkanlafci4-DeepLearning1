"""Farklı MLP yapılandırmalarının / hiperparametrelerin karşılaştırılması.

Her konfigürasyon aynı train/validation/test ayrımı ve aynı seed ile eğitilir.
Model seçimi validation'a göre yapılır; test doğruluğu sadece raporlama içindir.

Çalıştırma:
    python compare_experiments.py
"""
import json
import os

from src.plots import plot_comparison
from src.train import train_and_evaluate

BASE = {
    "hidden_sizes": [256, 128],
    "dropout": 0.0,
    "optimizer": "adam",
    "lr": 1e-3,
    "weight_decay": 0.0,
    "batch_size": 128,
    "epochs": 15,
    "val_size": 5000,
    "seed": 42,
}

EXPERIMENTS = {
    "baseline (256-128, Adam 1e-3)": {},
    "küçük (64-32)": {"hidden_sizes": [64, 32]},
    "geniş-derin (512-256-128)": {"hidden_sizes": [512, 256, 128]},
    "dropout 0.3": {"dropout": 0.3},
    "SGD+momentum lr=0.05": {"optimizer": "sgd", "lr": 0.05},
    "SGD+momentum lr=0.001": {"optimizer": "sgd", "lr": 0.001},
    "Adam lr=0.01": {"lr": 0.01},
    "batch size 32": {"batch_size": 32},
}


def main():
    os.makedirs("results", exist_ok=True)
    histories, rows = {}, []
    for name, override in EXPERIMENTS.items():
        print(f"\n=== {name} ===")
        _, hist, s = train_and_evaluate({**BASE, **override}, verbose=False)
        histories[name] = hist
        rows.append({"name": name, **s})
        print(f"train acc {s['final_train_acc']:.4f} | val acc {s['final_val_acc']:.4f} "
              f"(best {s['best_val_acc']:.4f} @ep{s['best_epoch']}) | test acc {s['test_acc']:.4f} "
              f"| {s['train_time_sec']:.1f}s")

    with open("results/comparison.json", "w") as f:
        json.dump(rows, f, indent=2)
    plot_comparison(histories, "results/comparison_curves.png")

    print("\n| Deney | Param | Train loss | Val loss | Train acc | Val acc (best) | Test acc | Süre (s) |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['name']} | {r['num_params']:,} | {r['final_train_loss']:.4f} | {r['final_val_loss']:.4f} "
              f"| {r['final_train_acc']*100:.2f}% | {r['best_val_acc']*100:.2f}% | {r['test_acc']*100:.2f}% "
              f"| {r['train_time_sec']:.1f} |")


if __name__ == "__main__":
    main()
