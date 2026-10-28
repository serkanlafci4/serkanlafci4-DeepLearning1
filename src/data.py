"""MNIST verisinin indirilmesi ve train / validation / test ayrımı.

- MNIST'in resmi 60.000'lik eğitim kümesi sabit bir seed ile
  55.000 train + 5.000 validation olarak ikiye bölünür.
- MNIST'in resmi 10.000'lik test kümesi ayrı tutulur ve yalnızca
  eğitim tamamen bittikten sonra bir kez değerlendirilir.
- Normalizasyon için ortalama / standart sapma SADECE train kısmından
  hesaplanır (validation ve test bilgisi sızmaz).
"""
import json
import os
import ssl

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

try:  # macOS python.org kurulumlarında SSL sertifika hatasını önler
    import certifi

    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
except ImportError:
    pass

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
STATS_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "normalization.json")


def _to_tensor(ds):
    x = ds.data.float().div(255.0).unsqueeze(1)  # (N, 1, 28, 28), [0, 1]
    y = ds.targets.long()
    return x, y


def get_loaders(batch_size=128, val_size=5000, seed=42):
    train_full = datasets.MNIST(DATA_DIR, train=True, download=True)
    test_ds = datasets.MNIST(DATA_DIR, train=False, download=True)

    x_all, y_all = _to_tensor(train_full)
    x_test, y_test = _to_tensor(test_ds)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(x_all), generator=g)
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    x_train, y_train = x_all[train_idx], y_all[train_idx]
    x_val, y_val = x_all[val_idx], y_all[val_idx]

    mean, std = x_train.mean().item(), x_train.std().item()
    os.makedirs(os.path.dirname(STATS_PATH), exist_ok=True)
    with open(STATS_PATH, "w") as f:
        json.dump({"mean": mean, "std": std}, f, indent=2)

    norm = lambda x: (x - mean) / std
    x_train, x_val, x_test = norm(x_train), norm(x_val), norm(x_test)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size,
                              shuffle=True, generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=1024)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=1024)

    sizes = {"train": len(x_train), "val": len(x_val), "test": len(x_test)}
    return train_loader, val_loader, test_loader, (mean, std), sizes


def load_normalization():
    with open(STATS_PATH) as f:
        s = json.load(f)
    return s["mean"], s["std"]
