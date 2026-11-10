"""Classical hand-crafted baseline: dense+keypoint SIFT bag-of-visual-words
(k=256) + HSV colour histogram + LBP texture histogram -> RBF-SVM.

usage: python src/classical.py kvasir
"""
import os
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.cluster import MiniBatchKMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(__file__))
from common import RES, load_split, metrics, save_json  # noqa: E402

K = 256


def sift_desc(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    s = cv2.SIFT_create()
    kp = [cv2.KeyPoint(float(x), float(y), 12) for y in range(8, 224, 12) for x in range(8, 224, 12)]
    _, d = s.compute(g, kp)
    return d.astype(np.float32)


def global_feats(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
    h = h / (h.sum() + 1e-8)
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(g, 8, 1, "uniform")
    lh = np.histogram(lbp, bins=10, range=(0, 10))[0].astype(np.float32)
    return np.concatenate([h, lh / lh.sum()])


def _both(img):
    return sift_desc(img), global_feats(img)


def main(ds):
    run = f"{ds}_classical"
    x, y, split, classes, _ = load_split(ds)
    t0 = time.time()
    with Pool(os.cpu_count()) as p:
        out = p.map(_both, [np.asarray(im) for im in x], chunksize=64)
    descs = [o[0] for o in out]; glob = np.stack([o[1] for o in out])
    tr, te = np.where(split == "train")[0], np.where(split == "test")[0]
    rng = np.random.default_rng(0)
    sample = np.concatenate([descs[i][rng.choice(len(descs[i]), 40, replace=False)] for i in tr])
    km = MiniBatchKMeans(K, random_state=0, batch_size=4096, n_init=3).fit(sample)
    bow = np.stack([np.bincount(km.predict(d), minlength=K) for d in descs]).astype(np.float32)
    bow = np.sqrt(bow / bow.sum(1, keepdims=True))  # Hellinger
    feat = np.concatenate([bow, glob], 1)
    clf = make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", probability=True, random_state=0))
    clf.fit(feat[tr], y[tr])
    prob = clf.predict_proba(feat[te])
    m = metrics(y[te], prob, len(classes))
    np.savez_compressed(os.path.join(RES, f"{run}_test.npz"), prob=prob, y=y[te])
    save_json(dict(run=run, ds=ds, model="classical", name="SIFT-BoVW+SVM", train_time_s=time.time() - t0,
                   test=m, classes=classes), os.path.join(RES, f"{run}.json"))
    print("TEST", run, {k: round(v, 4) for k, v in m.items()})


if __name__ == "__main__":
    for d in sys.argv[1:]:
        main(d)
