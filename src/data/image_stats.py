from __future__ import annotations

import io

import numpy as np


def _open(record):
    from PIL import Image
    if record.get("bytes") is not None:
        return Image.open(io.BytesIO(record["bytes"]))
    return Image.open(record["path"])


def scan_images(ds, split_name: str, channel_stats_indices: set[int] | None = None,
                quality_size: int = 256):
    """Return (DataFrame of per-image stats, channel mean/std over the given indices)."""
    import cv2
    import imagehash
    import pandas as pd
    from datasets import Image as HFImage
    from tqdm import tqdm

    raw = ds.cast_column("image", HFImage(decode=False))
    labels = ds["label"]
    ids = ds["id"] if "id" in ds.column_names else list(range(len(ds)))
    rows = []
    ch_sum = np.zeros(3)
    ch_sq = np.zeros(3)
    n_pix = 0

    for i in tqdm(range(len(raw)), desc=f"scanning {split_name}"):
        rec = raw[i]["image"]
        row = {"split": split_name, "index": i, "id": ids[i], "label": int(labels[i])}
        try:
            img = _open(rec)
            img.load()
        except Exception as exc:
            row.update(corrupt=True, error=str(exc)[:200])
            rows.append(row)
            continue

        w, h = img.size
        row.update(corrupt=False, width=w, height=h, mode=img.mode, aspect=w / h,
                   file_kb=len(rec["bytes"]) / 1024 if rec.get("bytes") else np.nan)
        rgb = img.convert("RGB")
        arr = np.asarray(rgb, dtype=np.float32) / 255.0

        pixel_range = arr.max(axis=-1) - arr.min(axis=-1)
        row["colorful_fraction"] = float((pixel_range > 0.08).mean())

        if channel_stats_indices is not None and i in channel_stats_indices:
            flat = arr.reshape(-1, 3)
            ch_sum += flat.sum(0)
            ch_sq += (flat ** 2).sum(0)
            n_pix += flat.shape[0]

        gray = cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        scale = quality_size / max(w, h)
        small = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))))
        row["brightness"] = float(small.mean() / 255.0)
        row["contrast"] = float(small.std() / 255.0)
        row["blur_laplacian_var"] = float(cv2.Laplacian(small, cv2.CV_64F).var())
        row["phash"] = str(imagehash.phash(rgb))
        rows.append(row)

    df = pd.DataFrame(rows)
    channel = None
    if n_pix:
        mean = ch_sum / n_pix
        std = np.sqrt(ch_sq / n_pix - mean ** 2)
        channel = {"mean": mean.round(4).tolist(), "std": std.round(4).tolist(),
                   "num_images": len(channel_stats_indices)}
    return df, channel


def find_near_duplicates(hashes_a: list[str], hashes_b: list[str] | None = None,
                         max_distance: int = 4, chunk: int = 512) -> list[tuple[int, int, int]]:
    """Pairs (i, j, hamming) with perceptual-hash distance <= max_distance."""
    a = np.array([int(h, 16) for h in hashes_a], dtype=np.uint64)
    same = hashes_b is None
    b = a if same else np.array([int(h, 16) for h in hashes_b], dtype=np.uint64)

    def popcount(x: np.ndarray) -> np.ndarray:
        if hasattr(np, "bitwise_count"):
            return np.bitwise_count(x)
        return np.unpackbits(x.view(np.uint8).reshape(*x.shape, 8), axis=-1).sum(-1)

    pairs = []
    for start in range(0, len(a), chunk):
        dist = popcount(a[start:start + chunk, None] ^ b[None, :])
        ii, jj = np.nonzero(dist <= max_distance)
        for i, j in zip(ii, jj):
            gi = int(i) + start
            if same and int(j) <= gi:
                continue
            pairs.append((gi, int(j), int(dist[i, j])))
    return pairs
