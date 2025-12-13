
"""
data_prep.py — minimal knobs
============================
Only one input required: --dataset_dir
Outputs to: ./artifacts/rbc_data.npz
"""

# ---- Editable constants (change here if needed) ----
IMG_SIZE = 224
VAL_FRAC = 0.15
TEST_FRAC = 0.15
LIMIT_PER_CLASS = None
OUT_NPZ = "./artifacts/rbc_data.npz"
# ----------------------------------------------------

import argparse, os, json
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image, ImageOps, ImageFilter

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def get_class_names(dataset_dir: str) -> List[str]:
    names = [d.name for d in Path(dataset_dir).iterdir() if d.is_dir()]
    names = sorted(names)
    if not names:
        raise ValueError(f"No class folders found under {dataset_dir}")
    return names

def ids_to_names(id_array: np.ndarray, class_names: List[str]):
    return [class_names[i - 1] for i in id_array]

def load_images(dataset_dir: str, class_names: List[str]):
    X_list, y_list = [], []
    for cls_idx, cls_name in enumerate(class_names, start=1):
        folder = Path(dataset_dir) / cls_name
        files = sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS])
        if LIMIT_PER_CLASS is not None:
            files = files[:LIMIT_PER_CLASS]
        for p in files:
            im = Image.open(p)
            im = im.convert("RGB")
            im = im.resize((IMG_SIZE, IMG_SIZE))
            # Optional toggles:
            # im = ImageOps.equalize(im)
            # im = ImageOps.autocontrast(im)
            # im = im.filter(ImageFilter.MedianFilter(size=3))
            arr = np.asarray(im, dtype=np.float32) / 255.0
            im.close()
            X_list.append(arr); y_list.append(cls_idx)
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    return X, y

def simple_splits(X, y):
    rng = np.random
    classes = np.unique(y)
    idx_tr, idx_va, idx_te = [], [], []
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(round(TEST_FRAC * n))
        n_val  = int(round(VAL_FRAC  * n))
        n_train = n - n_test - n_val
        idx_tr.extend(idx[:n_train])
        idx_va.extend(idx[n_train:n_train+n_val])
        idx_te.extend(idx[n_train+n_val:])
    return np.array(idx_tr), np.array(idx_va), np.array(idx_te)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)

    class_names = get_class_names(args.dataset_dir)
    print("Classes:", class_names)

    X, y = load_images(args.dataset_dir, class_names)
    print("Loaded:", X.shape, y.shape)

    i_tr, i_va, i_te = simple_splits(X, y)
    X_tr, y_tr = X[i_tr], y[i_tr]
    X_va, y_va = X[i_va], y[i_va]
    X_te, y_te = X[i_te], y[i_te]

    meta = {"class_names": class_names, "img_size": IMG_SIZE}
    np.savez_compressed(OUT_NPZ, X_train=X_tr, y_train=y_tr, X_val=X_va, y_val=y_va, X_test=X_te, y_test=y_te, meta=json.dumps(meta))
    print("Saved:", os.path.abspath(OUT_NPZ))

if __name__ == "__main__":
    main()
