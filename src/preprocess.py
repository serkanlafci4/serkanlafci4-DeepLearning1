"""Kendi el yazısı görüntülerini MNIST formatına dönüştürme.

MNIST'in hazırlanış biçimi taklit edilir:
  1. Gri tonlamaya çevir.
  2. Arka plan açıksa (beyaz kağıt) renkleri ters çevir -> siyah arka plan, beyaz rakam.
  3. Otsu eşiği ile kağıt dokusu / gölge gürültüsünü temizle.
  4. (Çok rakamlı görüntüde) sütun izdüşümü ile rakamları ayır.
  5. Her rakamı sınırlayıcı kutuya göre kırp, kalem çizgisini MNIST kalınlığına yaklaştır.
  6. En-boy oranını koruyarak uzun kenarı 20 piksele ölçekle.
  7. 28x28 siyah tuvale yerleştir ve kütle merkezini (14, 14)'e kaydır.
  8. [0, 1] aralığına getir ve eğitimdeki mean/std ile normalize et.
"""
import numpy as np
from PIL import Image, ImageFilter, ImageOps


def load_grayscale(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)  # telefon fotoğraflarının yönünü düzelt
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img.convert("RGBA"))
    return np.asarray(img.convert("L"), dtype=np.float32)


def otsu_threshold(arr):
    hist, _ = np.histogram(arr, bins=256, range=(0, 256))
    total = arr.size
    sum_all = np.dot(np.arange(256), hist)
    w_b = sum_b = 0.0
    best, thr = -1.0, 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b, m_f = sum_b / w_b, (sum_all - sum_b) / w_f
        between = w_b * w_f * (m_b - m_f) ** 2
        if between > best:
            best, thr = between, t
    return thr


def to_ink_image(gray):
    """Siyah arka plan / beyaz mürekkep görüntüsü ve mürekkep maskesi döndürür."""
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    if np.median(border) > 127:  # açık arka plan -> MNIST gibi ters çevir
        gray = 255.0 - gray

    # Hafif bulanıklaştırma ile kamera gürültüsünü azalt
    gray = np.asarray(Image.fromarray(gray.astype(np.uint8)).filter(ImageFilter.GaussianBlur(1)),
                      dtype=np.float32)
    thr = otsu_threshold(gray)
    mask = gray > thr
    ink = np.where(mask, gray, 0.0)
    if ink.max() > thr:  # kontrastı germe: mürekkep [0, 255] aralığına
        ink = np.where(mask, (ink - thr) / (ink.max() - thr) * 255.0, 0.0)
    return ink, mask


def _runs(profile, min_gap):
    """profile > 0 olan ardışık bölgeleri bulur; min_gap'ten küçük boşlukları birleştirir."""
    idx = np.flatnonzero(profile > 0)
    if idx.size == 0:
        return []
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > min_gap:
            runs.append((start, prev + 1))
            start = i
        prev = i
    runs.append((start, prev + 1))
    return runs


def segment_digits(ink, mask, min_rel_ink=0.05):
    """Yan yana yazılmış rakamları soldan sağa ayırır."""
    h, w = mask.shape
    col_profile = mask.sum(axis=0)
    # Gürültü sütunlarını yok say
    col_profile = np.where(col_profile > max(1, 0.01 * h), col_profile, 0)
    runs = _runs(col_profile, min_gap=max(1, w // 200))
    if not runs:
        return []

    crops = []
    largest = max(mask[:, x0:x1].sum() for x0, x1 in runs)
    for x0, x1 in runs:
        sub = mask[:, x0:x1]
        if sub.sum() < min_rel_ink * largest:
            continue  # küçük leke / nokta
        rows = np.flatnonzero(sub.sum(axis=1) > 0)
        y0, y1 = rows[0], rows[-1] + 1
        crops.append((ink[y0:y1, x0:x1], mask[y0:y1, x0:x1]))
    return crops


def _stroke_width(mask):
    """Çizgi kalınlığı tahmini: alan / (çevre / 2)."""
    area = mask.sum()
    if area == 0:
        return 0.0
    padded = np.pad(mask, 1)
    interior = (padded[1:-1, 1:-1] & padded[:-2, 1:-1] & padded[2:, 1:-1]
                & padded[1:-1, :-2] & padded[1:-1, 2:])
    perimeter = max(1, (mask & ~interior).sum())
    return 2.0 * area / perimeter


def crop_to_mnist(ink, mask, thicken=True):
    """Tek bir rakam kırpıntısını 28x28 MNIST görüntüsüne çevirir. Çıktı [0, 1] aralığında."""
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    ink = ink[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    mask = mask[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

    # Kare tuvale kenar boşluğu ile yerleştir (dilate işlemi kenara taşmasın)
    long_side = max(ink.shape)
    pad = long_side // 4
    side = long_side + 2 * pad
    canvas = np.zeros((side, side), dtype=np.float32)
    y = (side - ink.shape[0]) // 2
    x = (side - ink.shape[1]) // 2
    canvas[y:y + ink.shape[0], x:x + ink.shape[1]] = ink
    img = Image.fromarray(canvas.astype(np.uint8))

    if thicken:
        # MNIST'te 20 piksellik kutuda çizgi kalınlığı ~2-3 piksel (uzun kenarın ~1/8'i)
        target = long_side * 2.5 / 20.0
        k = int(round(target - _stroke_width(mask))) + 1
        k = min(k, max(3, long_side // 6))
        if k >= 3:
            k += (k + 1) % 2  # MaxFilter tek sayı ister
            img = img.filter(ImageFilter.MaxFilter(k))

    # Mürekkep kutusunu tekrar bul ve uzun kenarı 20 piksele ölçekle
    arr = np.asarray(img)
    rows, cols = np.flatnonzero(arr.max(1) > 0), np.flatnonzero(arr.max(0) > 0)
    arr = arr[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    hh, ww = arr.shape
    scale = 20.0 / max(hh, ww)
    nh, nw = max(1, round(hh * scale)), max(1, round(ww * scale))
    small = np.asarray(Image.fromarray(arr).resize((nw, nh), Image.LANCZOS), dtype=np.float32)
    small = np.clip(small, 0, 255)

    out = np.zeros((28, 28), dtype=np.float32)
    y0, x0 = (28 - nh) // 2, (28 - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = small

    # Kütle merkezini görüntü merkezine kaydır (MNIST'in yaptığı gibi)
    total = out.sum()
    if total > 0:
        ys, xs = np.mgrid[0:28, 0:28]
        cy, cx = (ys * out).sum() / total, (xs * out).sum() / total
        dy, dx = int(round(13.5 - cy)), int(round(13.5 - cx))
        out = np.roll(np.roll(out, dy, axis=0), dx, axis=1)

    out /= max(out.max(), 1e-6)  # [0, 1]
    return out


def image_to_digits(path, single=False, thicken=True):
    """Görüntü dosyasından 28x28 rakam dizisi (soldan sağa) üretir."""
    gray = load_grayscale(path)
    ink, mask = to_ink_image(gray)
    if single:
        crops = [(ink, mask)]
    else:
        crops = segment_digits(ink, mask)
    return [crop_to_mnist(i, m, thicken=thicken) for i, m in crops if m.any()]


def normalize(digits, mean, std):
    return [(d - mean) / std for d in digits]
