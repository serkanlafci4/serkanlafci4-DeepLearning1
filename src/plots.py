import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_history(history, path, title="Deney 1 – Temel MLP"):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, history["train_loss"], "o-", label="Train loss")
    ax1.plot(epochs, history["val_loss"], "s-", label="Validation loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-entropy loss")
    ax1.set_title("Loss")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "o-", label="Train accuracy")
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "s-", label="Validation accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy")
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparison(results, path):
    """Birden fazla deneyin validation accuracy eğrilerini karşılaştırır."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, hist in results.items():
        ep = range(1, len(hist["val_loss"]) + 1)
        ax1.plot(ep, hist["val_loss"], label=name)
        ax2.plot(ep, [a * 100 for a in hist["val_acc"]], label=name)
    ax1.set_title("Validation loss")
    ax2.set_title("Validation accuracy (%)")
    for ax in (ax1, ax2):
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
