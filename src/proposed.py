"""Proposed cascaded spatio-temporal system.

Stage 1  MobileNetV2 (focal) gate. Every incoming frame is scored once,
         s_t = sum_{c in abnormal} P(c | I_t); the routing decision uses the
         gate probabilities averaged over the k-frame buffer (temporally
         smoothed gate), so one corrupted frame cannot force a bypass.
Stage 2  ResNet-50 (focal) trunk -> 2048-d spatial feature f_t per frame
Stage 3  Bi-LSTM + temporal attention over the buffer of k=5 features

Synthetic clips are generated from each still image by a smooth simulated
camera trajectory; endoscopic artifacts (motion blur, defocus, specular flash,
noise/occlusion) are injected into random frames.

Outputs: Bi-LSTM checkpoint, ablations, tau sweep, robustness conditions,
measured latency, and the corrupted current frames used to test baselines.
Per-frame features / probabilities are cached in data/cache so the analysis
can be re-run in seconds.

usage: python src/proposed.py kvasir
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from common import (ARTIFACT_NAMES, CACHE, CKPT, K_WINDOW, RES, FocalLoss, abnormal_mask, augment,  # noqa: E402
                    class_alpha, corrupt, device, load_split, make_windows, metrics, normalize,
                    save_json, seed_all, to_input)
from models import TemporalBiLSTM, build, resnet_features  # noqa: E402

AMP = dict(device_type="mps", dtype=torch.float16)


def load_cnn(ds, name, n_cls, dev):
    m, _ = build(name, n_cls, pretrained=False)
    m.load_state_dict(torch.load(os.path.join(CKPT, f"{ds}_{name}_focal.pt"), map_location="cpu"))
    return m.to(dev).eval()


@torch.no_grad()
def extract(x_u8, trunk, head, gate, dev, seed, views=1, bs=24, aug=False, transient=None, tag="", keep_cur=True, **wkw):
    """Build `views` k-frame clips per image and return per-frame ResNet-50
    features, per-frame ResNet-50 probabilities, per-frame gate probabilities,
    the current frames (uint8, first view) and the corruption masks.
    transient=kind corrupts only the current frame with artifact `kind`."""
    gen = torch.Generator().manual_seed(seed)
    feats, probs, gprob, cur, masks = [], [], [], [], []
    nb, t0 = views * ((len(x_u8) + bs - 1) // bs), time.time()
    for v in range(views):
        for i in range(0, len(x_u8), bs):
            xb = to_input(torch.from_numpy(np.ascontiguousarray(x_u8[i:i + bs])), dev)
            if aug:
                xb = augment(xb)
            clip, mk = make_windows(xb, gen, **wkw)                  # (B,k,3,H,W)
            if transient is not None:
                for j in range(len(clip)):
                    clip[j, -1] = corrupt(clip[j, -1:], gen, transient)[0]
                mk[:, -1] = True
            b, k = clip.shape[:2]
            flat = normalize(clip.flatten(0, 1))
            with torch.autocast(**AMP):
                f = trunk(flat).float()
                p = F.softmax(head(f).float(), 1)
                g = F.softmax(gate(flat).float(), 1)
            feats.append(f.view(b, k, -1).half().cpu()); probs.append(p.view(b, k, -1).cpu())
            gprob.append(g.view(b, k, -1).cpu()); masks.append(mk)
            if v == 0 and keep_cur:
                cur.append((clip[:, -1].permute(0, 2, 3, 1) * 255).round().byte().cpu())
            done = len(feats)
            if done % 100 == 0:
                print(f"  [{tag}] {done}/{nb} batches, {time.time() - t0:.0f}s", flush=True)
    return dict(f=torch.cat(feats), p=torch.cat(probs).numpy(), g=torch.cat(gprob).numpy(),
                cur=torch.cat(cur).numpy() if cur else None, mask=torch.cat(masks).numpy())


def cached(ds, name, fn):
    path = os.path.join(CACHE, f"{ds}_prop_{name}.pt")
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    d = fn(); torch.save(d, path); return d


def train_lstm(ftr, ytr, fva, yva, n_cls, dev, epochs=40):
    seed_all(0)
    net = TemporalBiLSTM(ftr.shape[-1], 256, n_cls).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    steps = epochs * ((len(ftr) + 127) // 128)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-5)
    crit = FocalLoss(class_alpha(ytr.numpy(), n_cls), 2.0).to(dev)
    best, state, hist = -1, None, []
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(ftr)); tl = 0
        for i in range(0, len(ftr), 128):
            idx = perm[i:i + 128]
            f = ftr[idx].to(dev).float()
            if torch.rand(1) < 0.5:  # temporal reversal augmentation
                f = f.flip(1)
            loss = crit(net(f), ytr[idx].to(dev))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl += loss.item() * len(idx)
        pv = predict_lstm(net, fva, dev)
        mv = metrics(yva.numpy(), pv, n_cls)
        hist.append(dict(epoch=ep + 1, train_loss=tl / len(ftr), val_acc=mv["acc"], val_f1=mv["f1"]))
        if mv["f1"] > best:
            best, state = mv["f1"], {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state)
    return net.eval(), hist


@torch.no_grad()
def predict_lstm(net, f, dev, bs=512, att=False):
    net.eval(); out, aa = [], []
    for i in range(0, len(f), bs):
        y, a = net(f[i:i + bs].to(dev).float(), return_att=True)
        out.append(F.softmax(y, 1).cpu()); aa.append(a.cpu())
    return (torch.cat(out).numpy(), torch.cat(aa).numpy()) if att else torch.cat(out).numpy()


def gate_probs(g, mode):
    """g: (N,k,C) per-frame gate probabilities -> (N,C) used for routing."""
    return g[:, -1] if mode == "current" else g.mean(1)


def cascade(gp, temporal_p, abn, tau):
    """Route to Stages 2-3 if s >= tau; bypassed frames get the gate's best normal class."""
    s = gp[:, abn].sum(1)
    route = s >= tau
    gnorm = gp.copy(); gnorm[:, abn] = 0
    gnorm /= gnorm.sum(1, keepdims=True).clip(1e-12)
    return np.where(route[:, None], temporal_p, gnorm), route


def pick_tau(gp, y, abn, target=0.995):
    s = gp[:, abn].sum(1); is_abn = abn[y]
    taus = np.round(np.linspace(0.01, 0.99, 99), 2)
    rec = np.array([(s[is_abn] >= t).mean() for t in taus])
    return float(taus[rec >= target].max()) if (rec >= target).any() else 0.01, taus


@torch.no_grad()
def latency(gate, trunk, head, net, dev, n=200):
    x = torch.rand(1, 3, 224, 224, device=dev)
    buf = torch.randn(1, K_WINDOW, 2048, device=dev)

    def t(fn):
        for _ in range(20): fn()
        torch.mps.synchronize(); t0 = time.perf_counter()
        for _ in range(n): fn()
        torch.mps.synchronize(); return (time.perf_counter() - t0) / n * 1e3

    def g():
        with torch.autocast(**AMP): gate(normalize(x))

    def r():
        with torch.autocast(**AMP): trunk(normalize(x))

    def l(): net(buf)

    def full():  # steady state: current-frame feature + Bi-LSTM over cached buffer
        with torch.autocast(**AMP): f = trunk(normalize(x))
        net(torch.cat([buf[:, 1:], f.float()[:, None]], 1))

    def single():
        with torch.autocast(**AMP): head(trunk(normalize(x)))
    return dict(gate_ms=t(g), resnet_ms=t(r), lstm_ms=t(l), stage23_ms=t(full), resnet_single_ms=t(single))


def main(ds, views=3):
    seed_all(42); dev = device()
    x, y, split, classes, _ = load_split(ds)
    n_cls = len(classes); abn = abnormal_mask(ds, classes)
    tr, va, te = (np.where(split == s)[0] for s in ("train", "val", "test"))
    gate = load_cnn(ds, "mobilenetv2", n_cls, dev)
    res = load_cnn(ds, "resnet50", n_cls, dev)
    trunk, head = resnet_features(res).eval(), res.fc
    t0 = time.time()
    E = lambda xs, **kw: (lambda: extract(xs, trunk, head, gate, dev, **kw))
    # training clips: 3 augmented views per image, artifacts injected with p=0.3
    dtr = cached(ds, "train", E(x[tr], seed=1, views=views, aug=True, p_art=0.3, tag="train", keep_cur=False))
    dvc = cached(ds, "val_clean", E(x[va], seed=2, tag="val"))
    dva = cached(ds, "val_art", E(x[va], seed=3, p_art=0.3, tag="val_art"))
    print("features ready", time.time() - t0, flush=True)
    ytr = torch.from_numpy(np.tile(y[tr], views))
    fva = torch.cat([dvc["f"], dva["f"]]); yv = np.concatenate([y[va], y[va]])
    net, hist = train_lstm(dtr["f"], ytr, fva, torch.from_numpy(yv), n_cls, dev)
    torch.save(net.state_dict(), os.path.join(CKPT, f"{ds}_bilstm.pt"))

    # tau on validation (clean + artifact) for both gate variants
    gv = np.concatenate([dvc["g"], dva["g"]])
    pv_lstm = predict_lstm(net, fva, dev)
    taus = {m: pick_tau(gate_probs(gv, m), yv, abn)[0] for m in ("current", "smoothed")}
    tau = taus["smoothed"]
    print("tau", taus, flush=True)
    sweep = []
    gs = gate_probs(gv, "smoothed"); sv = gs[:, abn].sum(1); is_abn = abn[yv]
    for t_ in np.round(np.linspace(0.01, 0.99, 99), 2):
        cp, route = cascade(gs, pv_lstm, abn, t_)
        sweep.append(dict(tau=float(t_), gate_recall=float((sv[is_abn] >= t_).mean()),
                          bypass=float(1 - route.mean()), acc=float((cp.argmax(1) == yv).mean())))

    conds = {"clean": dict(p_art=0.0), "stream_p30": dict(p_art=0.3)}
    for kind in range(len(ARTIFACT_NAMES)):  # transient artifact on the current frame only
        conds[f"transient_{kind}"] = dict(transient=kind)
    out = dict(ds=ds, classes=classes, tau=tau, tau_current_gate=taus["current"], lstm_history=hist,
               conditions={}, abnormal=abn.tolist(), tau_sweep_val=sweep)
    npz = {}; yt = y[te]
    for ci, (cn, kw) in enumerate(conds.items()):
        d = cached(ds, f"test_{cn}", E(x[te], seed=100 + ci, tag=cn, **kw))
        np.save(os.path.join(CACHE, f"{ds}_cond_{cn}_x.npy"), d["cur"])
        p_lstm, att = predict_lstm(net, d["f"], dev, att=True)
        p_single = d["p"][:, -1]
        p_mean = d["p"].mean(1)
        g_cur, g_sm = gate_probs(d["g"], "current"), gate_probs(d["g"], "smoothed")
        p_casc_cur, _ = cascade(g_cur, p_lstm, abn, taus["current"])
        p_casc, route = cascade(g_sm, p_lstm, abn, tau)
        r = {"ResNet-50 (focal), single frame": metrics(yt, p_single, n_cls),
             "+ temporal mean pooling": metrics(yt, p_mean, n_cls),
             "+ Bi-LSTM (no gate)": metrics(yt, p_lstm, n_cls),
             "+ gate on current frame only": metrics(yt, p_casc_cur, n_cls),
             "Proposed cascade": metrics(yt, p_casc, n_cls),
             "MobileNetV2 gate alone": metrics(yt, g_cur, n_cls)}
        s_te = g_sm[:, abn].sum(1); mk = d["mask"]
        r["_gate"] = dict(bypass=float(1 - route.mean()), gate_recall_abn=float((s_te[abn[yt]] >= tau).mean()),
                          gate_spec_normal=float((s_te[~abn[yt]] < tau).mean()),
                          abn_detect_recall=float(abn[p_casc.argmax(1)][abn[yt]].mean()),
                          att_on_corrupted=float(att[mk].mean()) if mk.any() else None,
                          att_on_clean=float(att[~mk].mean()))
        out["conditions"][cn] = r
        npz.update({f"{cn}_casc": p_casc, f"{cn}_lstm": p_lstm, f"{cn}_single": p_single, f"{cn}_gate": g_cur,
                    f"{cn}_att": att, f"{cn}_mask": mk})
        print(cn, {k: round(v["acc"], 4) for k, v in r.items() if not k.startswith("_")}, r["_gate"], flush=True)
    npz["y"] = yt
    np.savez_compressed(os.path.join(RES, f"{ds}_proposed_test.npz"), **npz)
    np.save(os.path.join(RES, f"{ds}_test_feat.npy"), cached(ds, "test_clean", None)["f"][:, -1].numpy())
    lat = latency(gate, trunk, head, net, dev)
    byp = out["conditions"]["clean"]["_gate"]["bypass"]
    lat["cascade_ms"] = lat["gate_ms"] + (1 - byp) * lat["stage23_ms"]
    lat["cascade_fps"] = 1e3 / lat["cascade_ms"]
    out["latency"] = lat; out["time_s"] = time.time() - t0
    save_json(out, os.path.join(RES, f"{ds}_proposed.json"))
    print("latency", lat, flush=True)


if __name__ == "__main__":
    views = int(os.environ.get("VIEWS", 3))  # training clips per image (GastroVision used 1 for time)
    for d in sys.argv[1:]:
        main(d, views)
