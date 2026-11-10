"""Shared utilities: paths, dataset registry, GPU augmentation, endoscopic
artifact simulation, synthetic temporal windows, focal loss and metrics."""
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "data", "raw")
CACHE = os.path.join(ROOT, "data", "cache")
CKPT = os.path.join(ROOT, "checkpoints")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "paper", "figures")
for _d in (CACHE, CKPT, RES, FIG):
    os.makedirs(_d, exist_ok=True)

IMG = 224
K_WINDOW = 5
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Classes treated as "anatomically normal" by the Stage-1 gate. Everything
# else (pathology, dye/resection interventions, tools) is routed to Stage 2.
NORMAL_CLASSES = {
    "kvasir": {"normal-cecum", "normal-pylorus", "normal-z-line"},
    "hyperkvasir": {"cecum", "ileum", "retroflex-rectum", "pylorus", "retroflex-stomach",
                    "z-line", "bbps-0-1", "bbps-2-3", "impacted-stool"},
    "gastrovision": {"Cecum", "Duodenal bulb", "Gastroesophageal_junction_normal z-line",
                     "Ileocecal valve", "Normal esophagus",
                     "Normal mucosa and vascular pattern in the large bowel",
                     "Normal stomach", "Pylorus", "Retroflex rectum",
                     "Small bowel_terminal ileum"},
}
DATASET_NAMES = {"kvasir": "Kvasir v2", "hyperkvasir": "HyperKvasir",
                 "gastrovision": "GastroVision", "cvc": "CVC-ClinicDB",
                 "kvasirseg": "Kvasir-SEG"}
MIN_PER_CLASS = 30


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else
                        "cuda" if torch.cuda.is_available() else "cpu")


def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def load_split(ds):
    """Returns uint8 image array (N,H,W,3), labels, split array, class names."""
    x = np.load(os.path.join(CACHE, f"{ds}_x.npy"), mmap_mode="r")
    meta = json.load(open(os.path.join(CACHE, f"{ds}_meta.json")))
    return x, np.array(meta["labels"]), np.array(meta["split"]), meta["classes"], meta


def abnormal_mask(ds, classes):
    normal = NORMAL_CLASSES[ds]
    return np.array([c not in normal for c in classes])


def to_input(xb_uint8, dev):
    """uint8 NHWC (cpu tensor) -> float NCHW in [0,1] on device."""
    return xb_uint8.to(dev, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.)


def normalize(x):
    return (x - MEAN.to(x.device)) / STD.to(x.device)


# --------------------------------------------------------------------------
# GPU augmentation (applied batch-wise, per-sample random parameters)
# --------------------------------------------------------------------------
def affine(x, angle, scale, tx, ty, flip_h=None, flip_v=None):
    n = x.shape[0]
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.zeros(n, 2, 3, device=x.device)
    sh = torch.ones(n, device=x.device) if flip_h is None else (1 - 2 * flip_h.float())
    sv = torch.ones(n, device=x.device) if flip_v is None else (1 - 2 * flip_v.float())
    theta[:, 0, 0] = cos * sh; theta[:, 0, 1] = -sin * sv; theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin * sh; theta[:, 1, 1] = cos * sv; theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def augment(x):
    """Stochastic training augmentation: flips, rotation, zoom, shift, colour jitter."""
    n, d = x.shape[0], x.device
    ang = (torch.rand(n, device=d) * 2 - 1) * math.radians(20)
    sc = 1 + (torch.rand(n, device=d) * 2 - 1) * 0.12
    tx = (torch.rand(n, device=d) * 2 - 1) * 0.08
    ty = (torch.rand(n, device=d) * 2 - 1) * 0.08
    x = affine(x, ang, sc, tx, ty, torch.rand(n, device=d) < .5, torch.rand(n, device=d) < .5)
    b = 1 + (torch.rand(n, 1, 1, 1, device=d) * 2 - 1) * 0.2
    c = 1 + (torch.rand(n, 1, 1, 1, device=d) * 2 - 1) * 0.2
    s = 1 + (torch.rand(n, 1, 1, 1, device=d) * 2 - 1) * 0.2
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - m) * c + m
    g = x.mean(1, keepdim=True)
    x = (x - g) * s + g
    return (x * b).clamp_(0, 1)


# --------------------------------------------------------------------------
# Endoscopic artifact simulation (motion blur, specular flash, defocus, noise)
# --------------------------------------------------------------------------
def _gauss_kernel(sigma, size, dev):
    ax = torch.arange(size, device=dev) - size // 2
    k = torch.exp(-(ax ** 2) / (2 * sigma ** 2)); k = k / k.sum()
    return k


def motion_blur(x, gen):
    size = int(torch.randint(11, 25, (1,), generator=gen)) | 1
    ang = float(torch.rand(1, generator=gen)) * math.pi
    k = torch.zeros(size, size)
    c = size // 2
    for t in np.linspace(-c, c, 4 * size):
        i, j = int(round(c + t * math.sin(ang))), int(round(c + t * math.cos(ang)))
        k[i, j] = 1
    k = (k / k.sum()).to(x.device).view(1, 1, size, size).repeat(3, 1, 1, 1)
    return F.conv2d(F.pad(x, [c] * 4, mode="reflect"), k, groups=3)


def defocus(x, gen):
    sigma = 2.5 + float(torch.rand(1, generator=gen)) * 3.5
    size = int(sigma * 3) * 2 + 1
    k = _gauss_kernel(sigma, size, x.device)
    x = F.conv2d(F.pad(x, [size // 2, size // 2, 0, 0], mode="reflect"), k.view(1, 1, 1, -1).repeat(3, 1, 1, 1), groups=3)
    return F.conv2d(F.pad(x, [0, 0, size // 2, size // 2], mode="reflect"), k.view(1, 1, -1, 1).repeat(3, 1, 1, 1), groups=3)


def specular_flash(x, gen):
    """Light flash / over-exposure plus saturated specular blobs."""
    n, _, h, w = x.shape
    a = 0.45 + float(torch.rand(1, generator=gen)) * 0.4
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=x.device),
                            torch.linspace(-1, 1, w, device=x.device), indexing="ij")
    blob = torch.zeros(h, w, device=x.device)
    for _ in range(int(torch.randint(2, 6, (1,), generator=gen))):
        cy, cx = (torch.rand(2, generator=gen) * 1.6 - 0.8).tolist()
        r = 0.08 + float(torch.rand(1, generator=gen)) * 0.25
        blob = torch.maximum(blob, torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r ** 2)))
    x = x * (1 - a) + a
    return torch.maximum(x, blob.view(1, 1, h, w).expand_as(x)).clamp(0, 1)


def noise_dropout(x, gen):
    """Sensor noise with partial frame drop-out (bubbles / lens occlusion)."""
    sig = 0.08 + float(torch.rand(1, generator=gen)) * 0.12
    nz = torch.randn(x.shape, generator=gen).to(x.device) * sig
    x = (x + nz).clamp(0, 1)
    _, _, h, w = x.shape
    hh, ww = int(h * 0.45), int(w * 0.45)
    y0 = int(torch.randint(0, h - hh, (1,), generator=gen)); x0 = int(torch.randint(0, w - ww, (1,), generator=gen))
    x[:, :, y0:y0 + hh, x0:x0 + ww] *= 0.1
    return x


ARTIFACTS = [motion_blur, defocus, specular_flash, noise_dropout]
ARTIFACT_NAMES = ["Motion blur", "Defocus", "Specular flash", "Noise/occlusion"]


def corrupt(frame, gen, kind=None):
    kind = int(torch.randint(0, len(ARTIFACTS), (1,), generator=gen)) if kind is None else kind
    return ARTIFACTS[kind](frame, gen)


def make_windows(x, gen, k=K_WINDOW, p_art=0.0, last_identity=True, art_kind=None, max_art=None):
    """Synthesise a k-frame clip from each image in x (N,3,H,W in [0,1]) by a
    smooth simulated camera trajectory (drifting rotation / zoom / translation).
    Frame k-1 is the current frame. Each frame is independently corrupted by
    an endoscopic artifact with probability p_art. Returns (N,k,3,H,W) and a
    (N,k) bool corruption mask."""
    n, d = x.shape[0], x.device
    r = lambda *s: (torch.rand(*s, generator=gen) * 2 - 1).to(d)
    v_ang, v_sc, v_tx, v_ty = r(n) * math.radians(4), r(n) * 0.03, r(n) * 0.03, r(n) * 0.03
    frames, mask = [], torch.zeros(n, k, dtype=torch.bool)
    for t in range(k):
        dt = (k - 1 - t)  # frames before the current one drift away from it
        if last_identity and dt == 0:
            f = x.clone()
        else:
            f = affine(x, v_ang * dt, 1 + v_sc * dt, v_tx * dt, v_ty * dt)
        frames.append(f)
    out = torch.stack(frames, 1)
    if p_art > 0:
        hit = torch.rand(n, k, generator=gen) < p_art
        if max_art is not None:  # keep at most max_art corrupted frames per clip
            for i in range(n):
                idx = torch.nonzero(hit[i]).flatten()
                if len(idx) > max_art:
                    keep = idx[torch.randperm(len(idx), generator=gen)[:max_art]]
                    hit[i] = False; hit[i, keep] = True
        for i, t in torch.nonzero(hit).tolist():
            out[i, t] = corrupt(out[i, t:t + 1], gen, art_kind)[0]
        mask = hit
    return out, mask


# --------------------------------------------------------------------------
# Loss and metrics
# --------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """FL(p_t) = -alpha_t (1-p_t)^gamma log(p_t), computed from log-softmax
    for numerical stability. alpha: per-class weights (inverse frequency)."""

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha if alpha is not None else torch.tensor([]))

    def forward(self, logits, y):
        logp = F.log_softmax(logits.float(), dim=1)
        logp_t = logp.gather(1, y[:, None]).squeeze(1)
        p_t = logp_t.exp()
        loss = -((1 - p_t) ** self.gamma) * logp_t
        if self.alpha.numel():
            loss = loss * self.alpha[y]
        return loss.mean()


def class_alpha(labels, n_cls):
    cnt = np.bincount(labels, minlength=n_cls).astype(np.float64)
    a = 1.0 / np.maximum(cnt, 1)
    a = a / a.sum() * n_cls  # mean weight = 1
    return torch.tensor(np.sqrt(a), dtype=torch.float32)  # tempered inverse frequency


def metrics(y, prob, n_cls=None):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                                 f1_score, matthews_corrcoef, precision_score, recall_score,
                                 roc_auc_score, confusion_matrix)
    n_cls = n_cls or prob.shape[1]
    pred = prob.argmax(1)
    cm = confusion_matrix(y, pred, labels=list(range(n_cls)))
    tn = cm.sum() - cm.sum(0) - cm.sum(1) + np.diag(cm)
    fp = cm.sum(0) - np.diag(cm)
    spec = np.mean(tn / np.maximum(tn + fp, 1))
    out = dict(acc=accuracy_score(y, pred), bal_acc=balanced_accuracy_score(y, pred),
               precision=precision_score(y, pred, average="macro", zero_division=0),
               recall=recall_score(y, pred, average="macro", zero_division=0),
               f1=f1_score(y, pred, average="macro", zero_division=0),
               specificity=spec, mcc=matthews_corrcoef(y, pred), kappa=cohen_kappa_score(y, pred))
    try:
        present = np.unique(y)
        p = prob[:, present] / np.clip(prob[:, present].sum(1, keepdims=True), 1e-12, None)
        yy = np.searchsorted(present, y)
        out["auc"] = roc_auc_score(yy, p, multi_class="ovr", average="macro") if len(present) > 2 else roc_auc_score(yy, p[:, 1])
    except Exception:
        out["auc"] = float("nan")
    return {k: float(v) for k, v in out.items()}


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
