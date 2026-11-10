"""Grad-CAM on the Stage-2 ResNet-50 (layer4), scaled by the Stage-3 temporal
confidence, plus quantitative localisation against expert polyp masks
(Kvasir-SEG, CVC-ClinicDB): pointing-game hit rate, energy-in-mask, IoU.

usage: python src/gradcam.py
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE, CKPT, RES, device, load_split, normalize, save_json, to_input  # noqa: E402
from models import build  # noqa: E402


class GradCAM:
    def __init__(self, model, layer):
        self.m = model; self.acts = self.grads = None
        def fwd(_, __, o):
            self.acts = o
            o.register_hook(lambda g: setattr(self, "grads", g))
        layer.register_forward_hook(fwd)

    def __call__(self, x, cls):
        self.m.zero_grad()
        logits = self.m(x)
        logits.gather(1, cls[:, None]).sum().backward()
        alpha = self.grads.mean(dim=(2, 3), keepdim=True)            # alpha_k^c = 1/Z sum_ij dy^c/dA^k_ij
        cam = F.relu((alpha * self.acts).sum(1, keepdim=True))       # ReLU(sum_k alpha_k^c A^k)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam / cam.flatten(1).max(1)[0].clamp_min(1e-8)[:, None, None]
        return cam.detach(), F.softmax(logits, 1).detach()


def loc_metrics(cam, mask):
    hit, energy, iou = [], [], []
    for c, m in zip(cam, mask):
        m = m.astype(bool)
        if m.sum() == 0:
            continue
        hit.append(bool(m.flat[c.argmax()]))
        energy.append(float(c[m].sum() / max(c.sum(), 1e-8)))
        b = c >= 0.5
        iou.append(float((b & m).sum() / max((b | m).sum(), 1)))
    return dict(pointing_game=float(np.mean(hit)), energy_in_mask=float(np.mean(energy)), iou=float(np.mean(iou)),
                mask_area=float(np.mean([m.mean() for m in mask])))


def run_model(tag, xe, cls_idx, dev, n_cls, layer_fn, model_name):
    m, _ = build(model_name, n_cls, pretrained=False)
    m.load_state_dict(torch.load(os.path.join(CKPT, f"kvasir_{model_name}{tag}.pt"), map_location="cpu"))
    m = m.to(dev).eval()
    gc = GradCAM(m, layer_fn(m)); cams, probs = [], []
    for i in range(0, len(xe), 16):
        xb = normalize(to_input(torch.from_numpy(np.ascontiguousarray(xe[i:i + 16])), dev)).requires_grad_(True)
        c, p = gc(xb, torch.full((len(xb),), cls_idx, device=dev))
        cams.append(c.cpu().numpy()); probs.append(p.cpu().numpy())
    return np.concatenate(cams), np.concatenate(probs)


def main():
    dev = device()
    _, _, _, classes, _ = load_split("kvasir")
    pol = classes.index("polyps"); n_cls = len(classes)
    cfgs = {"ResNet-50 (CE)": ("resnet50", "", lambda m: m.layer4),
            "ResNet-50 (focal, proposed Stage 2)": ("resnet50", "_focal", lambda m: m.layer4),
            "DenseNet-121 (CE)": ("densenet121", "", lambda m: m.features),
            "MobileNetV2 (focal, Stage 1)": ("mobilenetv2", "_focal", lambda m: m.features)}
    out, examples = {}, {}
    for ext in ("kvasirseg", "cvc"):
        xe = np.load(os.path.join(CACHE, f"{ext}_x.npy")); me = np.load(os.path.join(CACHE, f"{ext}_mask.npy"))
        rng = np.random.default_rng(0)
        sel = np.sort(rng.choice(len(xe), min(400, len(xe)), replace=False))
        xe, me = xe[sel], me[sel]
        out[ext] = {}
        for nm, (mn, tag, lf) in cfgs.items():
            cams, probs = run_model(tag, xe, pol, dev, n_cls, lf, mn)
            r = loc_metrics(cams, me); r["mean_p_polyp"] = float(probs[:, pol].mean())
            out[ext][nm] = r
            print(ext, nm, r, flush=True)
            if "proposed" in nm:
                # temporal-confidence weighting: scale CAM by P(polyp) (Stage-3 confidence proxy)
                examples[ext] = dict(x=xe[:12], m=me[:12], cam=cams[:12] * probs[:12, pol, None, None])
    save_json(out, os.path.join(RES, "gradcam_localisation.json"))
    np.savez_compressed(os.path.join(RES, "gradcam_examples.npz"),
                        **{f"{e}_{k}": v for e, d in examples.items() for k, v in d.items()})


if __name__ == "__main__":
    main()
