"""Evaluate every trained model:
  * robustness of single-frame baselines under the same artifact conditions
    the proposed system was tested on (current-frame corruption)
  * efficiency: parameters, GMACs, batch-1 latency / FPS on the GPU (MPS)
  * cross-dataset generalisation (Kvasir-trained) on CVC-ClinicDB and Kvasir-SEG

usage: python src/evaluate.py robust kvasir | efficiency | crossdata
"""
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from common import (CACHE, CKPT, RES, abnormal_mask, device, load_split, make_windows,  # noqa: E402
                    metrics, normalize, save_json, to_input)
from models import BASELINES, TemporalBiLSTM, build, count_params, gflops, resnet_features  # noqa: E402
from train_cnn import predict  # noqa: E402

AMP = dict(device_type="mps", dtype=torch.float16)
CONDS = ["clean", "stream_p30", "transient_0", "transient_1", "transient_2", "transient_3"]


def load(ds, model, tag, n_cls, dev):
    m, _ = build(model, n_cls, pretrained=False)
    m.load_state_dict(torch.load(os.path.join(CKPT, f"{ds}_{model}{tag}.pt"), map_location="cpu"))
    return m.to(dev).eval()


def robust(ds, only=None):
    dev = device()
    _, y, split, classes, _ = load_split(ds)
    yt = y[split == "test"]
    path = os.path.join(RES, f"{ds}_robustness.json")
    out = __import__("json").load(open(path)) if only and os.path.exists(path) else {}
    for model in (only or BASELINES):
        if not os.path.exists(os.path.join(CKPT, f"{ds}_{model}.pt")):
            continue
        m = load(ds, model, "", len(classes), dev)
        out[model] = {c: metrics(yt, predict(m, np.load(os.path.join(CACHE, f"{ds}_cond_{c}_x.npy")), dev), len(classes))
                      for c in CONDS}
        print(model, {c: round(v["acc"], 4) for c, v in out[model].items()}, flush=True)
    save_json(out, path)


def efficiency():
    """Batch-1 latency on the idle GPU (FP16, MPS) and on the CPU (FP32), so that
    the cascade can be judged both where launch overhead dominates (GPU) and
    where arithmetic dominates (CPU / edge devices)."""
    dev = device(); out = {}
    torch.set_num_threads(os.cpu_count())

    def lat(fn, d, n):
        with torch.no_grad():
            for _ in range(max(3, n // 8)): fn()
            if d.type == "mps": torch.mps.synchronize()
            t0 = time.perf_counter()
            for _ in range(n): fn()
            if d.type == "mps": torch.mps.synchronize()
        return (time.perf_counter() - t0) / n * 1e3

    def both(m, fwd):
        r = {}
        for d, n in ((dev, 100), (torch.device("cpu"), 20)):
            m = m.to(d).eval(); x = torch.rand(1, 3, 224, 224, device=d)
            amp = d.type == "mps"

            def f():
                with torch.autocast(device_type=d.type, dtype=torch.float16, enabled=amp): fwd(m, x)
            r["gpu_ms" if amp else "cpu_ms"] = lat(f, d, n)
        return r

    for model in BASELINES:
        m, _ = build(model, 8, pretrained=False)
        g = gflops(m, "cpu")
        r = both(m, lambda mm, x: mm(normalize(x)))
        out[model] = dict(name=BASELINES[model][0], params_m=count_params(m) / 1e6, gmacs=g,
                          latency_ms=r["gpu_ms"], fps=1e3 / r["gpu_ms"], cpu_ms=r["cpu_ms"], cpu_fps=1e3 / r["cpu_ms"])
        print(out[model], flush=True)
    # proposed components: gate (MobileNetV2), Stage 2+3 = ResNet-50 trunk + Bi-LSTM over cached buffer
    net = TemporalBiLSTM(2048, 256, 8)
    out["bilstm"] = dict(name="Bi-LSTM head", params_m=count_params(net) / 1e6)
    res, _ = build("resnet50", 8, pretrained=False); trunk = resnet_features(res)
    s23 = torch.nn.ModuleDict(dict(trunk=trunk, net=net))

    def fwd23(mm, x):
        f = mm["trunk"](normalize(x)).float()
        buf = torch.zeros(1, 5, 2048, device=x.device); buf[:, -1] = f
        with torch.autocast(device_type=x.device.type, enabled=False):  # LSTM runs in FP32
            mm["net"](buf)
    out["stage23"] = both(s23, fwd23)
    out["stage23"]["gmacs"] = out["resnet50"]["gmacs"]
    print("stage23", out["stage23"], flush=True)
    save_json(out, os.path.join(RES, "efficiency.json"))


@torch.no_grad()
def crossdata():
    """All CVC-ClinicDB / Kvasir-SEG frames contain a polyp; Kvasir-trained
    models should predict 'polyps' (or at least an abnormal class)."""
    dev = device(); ds = "kvasir"
    xk, yk, split, classes, meta = load_split(ds)
    abn = abnormal_mask(ds, classes); pol = classes.index("polyps")
    n_cls = len(classes)
    # remove Kvasir-SEG images that are near-duplicates of Kvasir v2 *training* images
    tr_th = thumbs(xk[split == "train"])
    out = {}
    for ext in ("cvc", "kvasirseg"):
        xe = np.load(os.path.join(CACHE, f"{ext}_x.npy"))
        th = thumbs(xe)
        sim = th @ tr_th.T
        keep = sim.max(1) < 0.995
        print(ext, "kept", keep.sum(), "of", len(xe), "after de-duplication vs Kvasir train", flush=True)
        xe = xe[keep]
        r = {"_n": int(keep.sum()), "_n_total": int(len(keep))}
        models = [(m, "") for m in BASELINES] + [("mobilenetv2", "_focal"), ("resnet50", "_focal")]
        for model, tag in models:
            p = predict(load(ds, model, tag, n_cls, dev), xe, dev)
            r[model + tag] = dict(polyp_recall=float((p.argmax(1) == pol).mean()),
                                  abnormal_recall=float(abn[p.argmax(1)].mean()))
        # proposed cascade on synthetic clean clips
        gate = load(ds, "mobilenetv2", "_focal", n_cls, dev)
        res = load(ds, "resnet50", "_focal", n_cls, dev); trunk = resnet_features(res).eval()
        net = TemporalBiLSTM(2048, 256, n_cls).to(dev)
        net.load_state_dict(torch.load(os.path.join(CKPT, f"{ds}_bilstm.pt"), map_location="cpu")); net.eval()
        tau = __import__("json").load(open(os.path.join(RES, f"{ds}_proposed.json")))["tau"]
        gen = torch.Generator().manual_seed(7); preds = []
        for i in range(0, len(xe), 24):
            xb = to_input(torch.from_numpy(xe[i:i + 24]), dev)
            clip, _ = make_windows(xb, gen)
            with torch.autocast(**AMP):
                f = trunk(normalize(clip.flatten(0, 1))).float().view(len(xb), clip.shape[1], -1)
                g = F.softmax(gate(normalize(xb)).float(), 1).cpu().numpy()
            pt = F.softmax(net(f), 1).cpu().numpy()
            route = g[:, abn].sum(1) >= tau
            gn = g.copy(); gn[:, abn] = 0
            preds.append(np.where(route[:, None], pt, gn).argmax(1))
        pr = np.concatenate(preds)
        r["proposed"] = dict(polyp_recall=float((pr == pol).mean()), abnormal_recall=float(abn[pr].mean()))
        out[ext] = r
        print(ext, {k: v for k, v in r.items()}, flush=True)
    save_json(out, os.path.join(RES, "crossdata.json"))


def thumbs(x):
    t = torch.from_numpy(np.asarray(x)).permute(0, 3, 1, 2).float()
    t = F.interpolate(t, size=(24, 24), mode="area").flatten(1)
    t = t - t.mean(1, keepdim=True)
    return F.normalize(t, dim=1).numpy()


if __name__ == "__main__":
    {"robust": lambda: [robust(d) for d in sys.argv[2:]], "efficiency": efficiency,
     "robust_googlenet": lambda: [robust(d, ["googlenet"]) for d in sys.argv[2:]],
     "crossdata": crossdata}[sys.argv[1]]()
