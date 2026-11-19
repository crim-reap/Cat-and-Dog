"""Dataset scanning, validation and cleaning helpers."""
import hashlib
import io
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
MODE_CHANNELS = {"1": 1, "L": 1, "P": 1, "I": 1, "I;16": 1, "F": 1, "LA": 2,
                 "RGB": 3, "YCbCr": 3, "LAB": 3, "HSV": 3, "RGBA": 4, "CMYK": 4}


def parallel_map(fn, items, workers=8, desc=""):
    """Order-preserving threaded map with a progress bar."""
    items = list(items)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(tqdm(ex.map(fn, items), total=len(items), desc=desc))


def infer_label(rel_parts):
    """Class label from the nearest parent folder whose name starts with cat/dog."""
    for part in reversed(rel_parts[:-1]):
        p = part.lower()
        if p.startswith("cat"):
            return "Cat"
        if p.startswith("dog"):
            return "Dog"
    return None


def read_header(path):
    """Cheap header-only read: width, height, mode, format, error."""
    try:
        with Image.open(path) as im:
            return im.width, im.height, im.mode, im.format, ""
    except Exception as e:  # noqa: BLE001
        return None, None, None, None, f"{type(e).__name__}: {e}"


def scan_directory(root, workers=8):
    """Recursively scan root and return (metadata DataFrame, list of non-image files)."""
    root = Path(root)
    rows, others = [], []
    for dirpath, dirnames, files in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__MACOSX"]   # junk folder created by macOS zips
        for f in files:
            if f.startswith("._"):
                continue
            p = Path(dirpath) / f
            if p.suffix.lower() in IMAGE_EXTS:
                rel = p.relative_to(root).parts
                rows.append(dict(
                    path=str(p),
                    label=infer_label(rel),
                    source_folder=str(Path(*rel[:-1])) if len(rel) > 1 else ".",
                    extension=p.suffix.lower(),
                    file_size_kb=p.stat().st_size / 1024,
                ))
            else:
                others.append(str(p))
    df = pd.DataFrame(rows)
    heads = parallel_map(read_header, df["path"], workers, "Reading image headers")
    hd = pd.DataFrame(heads, columns=["width", "height", "mode", "pil_format", "header_error"], index=df.index)
    df = pd.concat([df, hd], axis=1)
    df["channels"] = df["mode"].map(MODE_CHANNELS)
    return df, others


def check_image(path):
    """Full decode check. Returns dict with valid flag, error text, md5 and 64-bit dHash."""
    out = dict(path=path, valid=False, error="", md5=None, dhash=None)
    try:
        if os.path.getsize(path) == 0:
            out["error"] = "empty file (0 bytes)"
            return out
        with open(path, "rb") as f:
            data = f.read()
        out["md5"] = hashlib.md5(data).hexdigest()
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            im.load()  # forces a full decode -> catches truncated files
            g = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
        bits = (g[:, 1:] > g[:, :-1]).flatten()
        out["dhash"] = format(int("".join("1" if b else "0" for b in bits), 2), "016x")
        out["valid"] = True
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def detect_corrupted(paths, workers=8):
    """Run check_image on every path. Returns a DataFrame (valid, error, md5, dhash)."""
    return pd.DataFrame(parallel_map(check_image, paths, workers, "Validating images"))


def to_rgb(im):
    """Standardise any PIL image (L, P, LA, RGBA, CMYK, ...) to 3-channel RGB."""
    try:
        im = ImageOps.exif_transpose(im)
    except Exception:  # noqa: BLE001
        pass
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        return bg.convert("RGB")
    return im.convert("RGB")


def save_clean_copy(args):
    """Write a standardised RGB JPEG copy (longest side capped). Returns '' on success."""
    src, dst, max_side = args
    try:
        with Image.open(src) as im:
            im.load()
            im = to_rgb(im)
        im.thumbnail((max_side, max_side), Image.LANCZOS)
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, "JPEG", quality=95)
        return ""
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"
