"""Index raw datasets, make stratified 70/15/15 splits and cache every image as
224x224 uint8 (bicubic) so training never re-decodes high-resolution JPEGs.

usage: python src/prepare_data.py kvasir hyperkvasir gastrovision cvc kvasirseg
"""
import glob
import os
import sys
import zipfile
from collections import Counter
from multiprocessing import Pool

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE, IMG, MIN_PER_CLASS, RAW, save_json  # noqa: E402

EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def _load(p, mask=False):
    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE if mask else cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(p)
    if mask:
        return (cv2.resize(im, (IMG, IMG), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
    return cv2.cvtColor(cv2.resize(im, (IMG, IMG), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)


def _load_mask(p):
    return _load(p, True)


def imgs_in(d):
    return sorted(p for p in glob.glob(os.path.join(d, "**", "*"), recursive=True)
                  if p.lower().endswith(EXT) and "/." not in p)


def find_dir(root, name):
    for dp, dn, _ in os.walk(root):
        dn[:] = sorted(d for d in dn if not d.startswith("."))  # skip HF .cache
        if os.path.basename(dp) == name:
            return dp
    raise FileNotFoundError(f"{name} under {root}")


def index_classification(ds):
    if ds == "kvasir":
        z = os.path.join(RAW, "zips", "kvasir-dataset-v2.zip")
        out = os.path.join(RAW, "kvasir_v2")
        if not os.path.isdir(out):
            zipfile.ZipFile(z).extractall(out)
        root = find_dir(out, "kvasir-dataset-v2")
        items = [(p, os.path.basename(os.path.dirname(p))) for p in imgs_in(root)]
    elif ds == "hyperkvasir":
        root = find_dir(os.path.join(RAW, "hyperkvasir"), "labeled-images")
        items = [(p, os.path.basename(os.path.dirname(p))) for p in imgs_in(root)]
    elif ds == "gastrovision":
        root = find_dir(os.path.join(RAW, "gastrovision"), "Gastrovision")
        items = [(p, os.path.basename(os.path.dirname(p))) for p in imgs_in(root)]
    cnt = Counter(c for _, c in items)
    dropped = {c: n for c, n in cnt.items() if n < MIN_PER_CLASS}
    items = [(p, c) for p, c in items if c not in dropped]
    return items, dropped


def build_classification(ds):
    items, dropped = index_classification(ds)
    classes = sorted({c for _, c in items})
    paths = [p for p, _ in items]
    labels = np.array([classes.index(c) for _, c in items])
    idx = np.arange(len(paths))
    tr, rest = train_test_split(idx, test_size=0.30, stratify=labels, random_state=42)
    va, te = train_test_split(rest, test_size=0.50, stratify=labels[rest], random_state=42)
    split = np.empty(len(paths), dtype=object)
    split[tr], split[va], split[te] = "train", "val", "test"
    with Pool(os.cpu_count()) as pool:
        x = np.stack(pool.map(_load, paths, chunksize=32))
    np.save(os.path.join(CACHE, f"{ds}_x.npy"), x)
    save_json(dict(classes=classes, labels=labels.tolist(), split=split.tolist(),
                   paths=[os.path.relpath(p, RAW) for p in paths], dropped=dropped,
                   counts={c: int((labels == i).sum()) for i, c in enumerate(classes)}),
              os.path.join(CACHE, f"{ds}_meta.json"))
    print(ds, x.shape, len(classes), "classes; dropped", dropped,
          {s: int((split == s).sum()) for s in ("train", "val", "test")})


def build_segmentation(ds):
    """Polyp datasets with pixel masks (external test + Grad-CAM localisation)."""
    if ds == "cvc":
        root = os.path.join(RAW, "cvc_clinicdb")
        ims, mks = imgs_in(find_dir(root, "Original")), imgs_in(find_dir(root, "Ground Truth"))
    else:  # kvasirseg (HyperKvasir segmented-images == Kvasir-SEG, 1000 polyps)
        root = find_dir(os.path.join(RAW, "kvasirseg"), "segmented-images")
        ims, mks = imgs_in(os.path.join(root, "images")), imgs_in(os.path.join(root, "masks"))
    key = lambda p: os.path.splitext(os.path.basename(p))[0]
    mk = {key(p): p for p in mks}
    ims = [p for p in ims if key(p) in mk]
    with Pool(os.cpu_count()) as pool:
        x = np.stack(pool.map(_load, ims, chunksize=16))
        m = np.stack(pool.map(_load_mask, [mk[key(p)] for p in ims], chunksize=16))
    np.save(os.path.join(CACHE, f"{ds}_x.npy"), x)
    np.save(os.path.join(CACHE, f"{ds}_mask.npy"), m)
    save_json(dict(ids=[key(p) for p in ims], paths=[os.path.relpath(p, RAW) for p in ims]),
              os.path.join(CACHE, f"{ds}_meta.json"))
    print(ds, x.shape, m.shape, "mean polyp area", float(m.mean()))


if __name__ == "__main__":
    for ds in sys.argv[1:]:
        (build_segmentation if ds in ("cvc", "kvasirseg") else build_classification)(ds)
