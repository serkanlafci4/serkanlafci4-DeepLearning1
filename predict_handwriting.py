"""Eğitilmiş MLP modelini kendi el yazısı görüntüleri üzerinde test eder.

Kullanım:
    # Öğrenci numarasının tamamı tek fotoğrafta (rakamlar arasında boşluk bırakın):
    python predict_handwriting.py handwriting/raw/ogrenci_no.jpg --expected 2021123456

    # Her rakam ayrı dosyada (dosya adının ilk karakteri gerçek etiket kabul edilir, örn. 3_a.png):
    python predict_handwriting.py handwriting/raw/digits/ --single

Çıktılar:
    handwriting/processed/<isim>_<i>.png  -> MNIST formatına getirilmiş 28x28 rakamlar
    results/handwriting_<isim>.png        -> orijinal görüntü, işlenmiş rakamlar ve tahminler
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.data import load_normalization
from src.model import MLP
from src.preprocess import image_to_digits, load_grayscale, normalize

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".heic", ".webp")


def load_model(path):
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt["config"]
    model = MLP(hidden_sizes=cfg["hidden_sizes"], dropout=cfg["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict(model, digits, mean, std):
    x = torch.tensor(np.stack(normalize(digits, mean, std)), dtype=torch.float32).unsqueeze(1)
    probs = torch.softmax(model(x), dim=1)
    conf, pred = probs.max(1)
    return pred.tolist(), conf.tolist()


def visualize(original, digits, preds, confs, labels, path, title):
    n = len(digits)
    fig = plt.figure(figsize=(max(6, 1.4 * n), 4.2))
    ax = fig.add_subplot(2, 1, 1)
    ax.imshow(original, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
    for i, (d, p, c) in enumerate(zip(digits, preds, confs)):
        a = fig.add_subplot(2, n, n + i + 1)
        a.imshow(d, cmap="gray", vmin=0, vmax=1)
        ok = labels is None or labels[i] == p
        a.set_title(f"{p} ({c:.0%})", color="green" if ok else "red", fontsize=10)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Görüntü dosyası veya klasör")
    ap.add_argument("--model", default="results/mlp_baseline.pt")
    ap.add_argument("--expected", help="Beklenen rakam dizisi (örn. öğrenci numarası)")
    ap.add_argument("--single", action="store_true", help="Her görüntü tek bir rakam içeriyor")
    ap.add_argument("--no-thicken", action="store_true", help="Çizgi kalınlaştırmayı kapat")
    args = ap.parse_args()

    model = load_model(args.model)
    mean, std = load_normalization()
    os.makedirs("handwriting/processed", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    if os.path.isdir(args.input):
        files = sorted(os.path.join(args.input, f) for f in os.listdir(args.input)
                       if f.lower().endswith(IMG_EXT))
    else:
        files = [args.input]

    all_correct = all_total = 0
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        digits = image_to_digits(f, single=args.single, thicken=not args.no_thicken)
        if not digits:
            print(f"{f}: rakam bulunamadı")
            continue
        preds, confs = predict(model, digits, mean, std)

        for i, d in enumerate(digits):
            Image.fromarray((d * 255).astype(np.uint8)).save(f"handwriting/processed/{name}_{i}.png")

        labels = None
        if args.expected and not args.single:
            labels = [int(c) for c in args.expected]
            if len(labels) != len(digits):
                print(f"Uyarı: {len(labels)} rakam beklendi, {len(digits)} bulundu "
                      f"(rakamlar arasında daha fazla boşluk bırakmayı deneyin).")
                labels = None
        elif args.single and name[0].isdigit():
            labels = [int(name[0])]

        pred_str = "".join(map(str, preds))
        line = f"{f}: tahmin = {pred_str}  güven = {[round(c, 2) for c in confs]}"
        if labels is not None:
            correct = sum(int(p == l) for p, l in zip(preds, labels))
            all_correct += correct
            all_total += len(labels)
            line += f"  doğru = {correct}/{len(labels)}"
        print(line)

        title = f"Tahmin: {pred_str}" + (f"   |   Gerçek: {''.join(map(str, labels))}" if labels else "")
        visualize(load_grayscale(f), digits, preds, confs, labels,
                  f"results/handwriting_{name}.png", title)

    if all_total:
        print(f"\nToplam rakam doğruluğu: {all_correct}/{all_total} = {all_correct / all_total:.2%}")


if __name__ == "__main__":
    main()
