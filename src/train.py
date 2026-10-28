import time

import torch
import torch.nn as nn

from .data import get_loaders
from .model import MLP, count_parameters


def get_device():
    # Küçük bir MLP için CPU genelde yeterince hızlıdır ve sonuçlar daha tekrarlanabilirdir.
    return torch.device("cpu")


def run_epoch(model, loader, criterion, device, optimizer=None):
    """optimizer verilirse eğitim, verilmezse değerlendirme modu."""
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def build_optimizer(name, params, lr, momentum=0.9, weight_decay=0.0):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"Bilinmeyen optimizer: {name}")


def train_and_evaluate(config, verbose=True):
    """Bir konfigürasyonu eğitir; model seçimi validation'a göre yapılır,
    test kümesi yalnızca en sonda bir kez kullanılır."""
    torch.manual_seed(config["seed"])
    device = get_device()

    train_loader, val_loader, test_loader, (mean, std), sizes = get_loaders(
        batch_size=config["batch_size"], val_size=config["val_size"], seed=config["seed"])

    model = MLP(hidden_sizes=config["hidden_sizes"], dropout=config["dropout"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(config["optimizer"], model.parameters(), config["lr"],
                                weight_decay=config["weight_decay"])

    if verbose:
        print(f"Veri: train={sizes['train']} val={sizes['val']} test={sizes['test']} "
              f"| normalizasyon mean={mean:.4f} std={std:.4f}")
        print(model)
        print(f"Parametre sayısı: {count_parameters(model):,}")

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "epoch_time": []}
    best_val_acc, best_state, best_epoch = -1.0, None, 0

    start = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        dt = time.perf_counter() - t0

        for k, v in zip(history, (tr_loss, va_loss, tr_acc, va_acc, dt)):
            history[k].append(v)

        if va_acc > best_val_acc:
            best_val_acc, best_epoch = va_acc, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if verbose:
            print(f"Epoch {epoch:02d}/{config['epochs']} | "
                  f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                  f"val loss {va_loss:.4f} acc {va_acc:.4f} | {dt:.1f}s")
    train_time = time.perf_counter() - start

    # En iyi validation doğruluğuna sahip ağırlıklarla test (yalnızca bir kez)
    model.load_state_dict(best_state)
    test_loss, test_acc = run_epoch(model, test_loader, criterion, device)

    summary = {
        "config": config,
        "num_params": count_parameters(model),
        "best_epoch": best_epoch,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "final_train_acc": history["train_acc"][-1],
        "final_val_acc": history["val_acc"][-1],
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "train_time_sec": train_time,
    }
    if verbose:
        print(f"\nEn iyi epoch: {best_epoch} (val acc {best_val_acc:.4f})")
        print(f"Test loss {test_loss:.4f} | Test accuracy {test_acc:.4f}")
        print(f"Toplam eğitim süresi: {train_time:.1f} s")
    return model, history, summary
