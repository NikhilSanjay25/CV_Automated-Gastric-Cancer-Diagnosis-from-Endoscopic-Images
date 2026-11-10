"""Generate every figure and LaTeX table used in the paper from results/*.json.

usage: python src/make_figures.py
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from common import ARTIFACT_NAMES, CACHE, DATASET_NAMES, FIG, RES, load_split  # noqa: E402
from models import BASELINES  # noqa: E402

# validated reference palette (light mode) -------------------------------------------------
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                                         "#e87ba4", "#008300", "#6250d6", "#e34948")
CAT = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
GRAY, INK, INK2, GRID = "#a9a8a2", "#0b0b0b", "#52514e", "#e6e5e0"
SEQ = LinearSegmentedColormap.from_list("seqblue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DS_COLORS = {"kvasir": BLUE, "hyperkvasir": ORANGE, "gastrovision": AQUA}
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 300,
                     "savefig.bbox": "tight", "figure.dpi": 150, "lines.linewidth": 1.6})
COLW, FULLW = 3.5, 7.16  # IEEE column / page widths (in)
ORDER = ["classical", "alexnet", "vgg16", "googlenet", "resnet50", "densenet121", "mobilenetv2",
         "efficientnetb0", "vitb16"]
NAMES = {"classical": "SIFT-BoVW+SVM", **{k: v[0] for k, v in BASELINES.items()}, "proposed": "Proposed"}
DSS = [d for d in ("kvasir", "gastrovision") if os.path.exists(os.path.join(RES, f"{d}_proposed.json"))]


def J(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf")); fig.savefig(os.path.join(FIG, name + ".png"))
    plt.close(fig); print("fig", name)


def cascade_cost(ds, prevalence=None):
    """Expected per-frame cascade latency (ms) on GPU and CPU from the idle-device
    component timings and the measured routing statistics of the gate. With
    prevalence=pi (fraction of normal frames) the pass rate is re-weighted."""
    eff, p = J("efficiency.json"), J(f"{ds}_proposed.json")
    if not eff or "stage23" not in eff or not p: return None
    g = p["conditions"]["clean"]["_gate"]
    if prevalence is None:
        pass_rate = 1 - g["bypass"]
    else:
        pass_rate = prevalence * (1 - g["gate_spec_normal"]) + (1 - prevalence) * g["gate_recall_abn"]
    out = {}
    for dev, k in (("gpu", "latency_ms"), ("cpu", "cpu_ms")):
        ms = eff["mobilenetv2"][k] + pass_rate * eff["stage23"][f"{dev}_ms"]
        out[f"{dev}_ms"], out[f"{dev}_fps"] = ms, 1e3 / ms
    out["gmacs"] = eff["mobilenetv2"]["gmacs"] + pass_rate * eff["stage23"]["gmacs"]
    out["pass_rate"] = pass_rate
    return out


def results_table(ds):
    """rows: model -> test metrics (clean). Proposed = cascade on clean clips."""
    rows = {}
    for m in ORDER:
        r = J(f"{ds}_{m}.json")
        if r: rows[m] = r["test"]
    p = J(f"{ds}_proposed.json")
    if p: rows["proposed"] = p["conditions"]["clean"]["Proposed cascade"]
    return rows


# ------------------------------------------------------------------------------------------
def fig_dataset_samples():
    dss = [d for d in ("kvasir", "gastrovision") if os.path.exists(os.path.join(CACHE, f"{d}_x.npy"))]
    segs = [d for d in ("kvasirseg", "cvc") if os.path.exists(os.path.join(CACHE, f"{d}_x.npy"))]
    nc = 6
    fig, axes = plt.subplots(len(dss) + len(segs), nc, figsize=(FULLW, 0.98 * (len(dss) + len(segs))))
    fig.subplots_adjust(wspace=0.04, hspace=0.32)
    for r, d in enumerate(dss):
        x, y, _, classes, _ = load_split(d)
        rng = np.random.default_rng(1)
        cls = rng.choice(len(classes), nc, replace=False) if len(classes) > nc else range(nc)
        for c, k in enumerate(cls):
            i = rng.choice(np.where(y == k)[0])
            ax = axes[r, c]; ax.imshow(x[i]); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            ax.set_title(classes[k].replace("_", " ").replace("ulcerative-colitis", "UC")[:21], fontsize=5.5, color=INK2)
            if c == 0: ax.set_ylabel(DATASET_NAMES[d], fontsize=7)
    for r, d in enumerate(segs, start=len(dss)):
        x = np.load(os.path.join(CACHE, f"{d}_x.npy"), mmap_mode="r"); m = np.load(os.path.join(CACHE, f"{d}_mask.npy"), mmap_mode="r")
        idx = np.random.default_rng(2).choice(len(x), nc // 2, replace=False)
        for c in range(nc):
            ax = axes[r, c]; i = idx[c // 2]
            ax.imshow(x[i] if c % 2 == 0 else m[i], cmap=None if c % 2 == 0 else "gray")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            ax.set_title("image" if c % 2 == 0 else "expert mask", fontsize=6, color=INK2)
            if c == 0: ax.set_ylabel(DATASET_NAMES[d], fontsize=7)
    for ax in axes.flat:
        for s in ax.spines.values(): s.set_visible(False)
    save(fig, "fig_datasets")


def fig_baseline_bars():
    """Accuracy / macro-F1 / MCC of every model, grouped by dataset."""
    mets = [("acc", "Accuracy"), ("f1", "Macro-F1"), ("mcc", "MCC")]
    fig, axes = plt.subplots(1, 3, figsize=(FULLW, 2.6), sharey=True)
    models = ORDER + ["proposed"]
    w = 0.8 / max(len(DSS), 1)
    for ax, (k, lab) in zip(axes, mets):
        for j, d in enumerate(DSS):
            rows = results_table(d)
            vals = [rows.get(m, {}).get(k, np.nan) * 100 for m in models]
            ax.barh(np.arange(len(models)) + (j - (len(DSS) - 1) / 2) * w, vals, height=w * 0.9,
                    color=DS_COLORS[d], label=DATASET_NAMES[d], edgecolor="white", linewidth=0.5)
        ax.set_xlabel(lab + " (%)"); ax.set_xlim(40, 100); ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(models)))
    labs = axes[0].set_yticklabels([NAMES[m] for m in models]); labs[-1].set_fontweight("bold")
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower center", bbox_to_anchor=(1.6, 1.0), ncol=3)
    save(fig, "fig_baseline_comparison")


def fig_curves():
    fig, axes = plt.subplots(1, 2, figsize=(FULLW, 2.3))
    for i, m in enumerate([m for m in ORDER if m != "classical"]):
        r = J(f"kvasir_{m}.json")
        if not r: continue
        h = r["history"]; ep = [e["epoch"] for e in h]
        c = CAT[i % 8]
        axes[0].plot(ep, [e["val_acc"] * 100 for e in h], color=c, marker="o", ms=3, label=NAMES[m])
        axes[1].plot(ep, [e["val_loss"] for e in h], color=c, marker="o", ms=3)
    for tag, ls in (("resnet50_focal", "--"), ("mobilenetv2_focal", ":")):
        r = J(f"kvasir_{tag}.json")
        if r:
            h = r["history"]; ep = [e["epoch"] for e in h]
            axes[0].plot(ep, [e["val_acc"] * 100 for e in h], color=INK, ls=ls, label=NAMES[tag.split("_")[0]] + " (focal, ours)")
            axes[1].plot(ep, [e["val_loss"] for e in h], color=INK, ls=ls)
    axes[0].set_ylabel("Validation accuracy (%)"); axes[1].set_ylabel("Validation cross-entropy")
    for ax in axes: ax.set_xlabel("Epoch")
    axes[0].legend(fontsize=6, ncol=2, loc="lower right")
    save(fig, "fig_training_curves")


def fig_lstm_curve():
    fig, ax = plt.subplots(figsize=(COLW, 2.0))
    for j, d in enumerate(DSS):
        h = J(f"{d}_proposed.json")["lstm_history"]
        ax.plot([e["epoch"] for e in h], [e["val_f1"] * 100 for e in h], color=DS_COLORS[d], label=DATASET_NAMES[d])
    ax.set_xlabel("Bi-LSTM epoch"); ax.set_ylabel("Val. macro-F1 (%)"); ax.legend()
    save(fig, "fig_bilstm_curve")


def fig_confusion():
    fig, axes = plt.subplots(1, len(DSS), figsize=(FULLW, FULLW / len(DSS) + 0.4))
    axes = np.atleast_1d(axes)
    from sklearn.metrics import confusion_matrix
    for ax, d in zip(axes, DSS):
        z = np.load(os.path.join(RES, f"{d}_proposed_test.npz"))
        classes = J(f"{d}_proposed.json")["classes"]
        cm = confusion_matrix(z["y"], z["clean_casc"].argmax(1), labels=range(len(classes)))
        cmn = cm / cm.sum(1, keepdims=True).clip(1)
        ax.imshow(cmn, cmap=SEQ, vmin=0, vmax=1); ax.grid(False)
        n = len(classes)
        if n <= 10:
            for i in range(n):
                for j in range(n):
                    if cm[i, j]:
                        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=5.5,
                                color="white" if cmn[i, j] > 0.55 else INK)
        lab = [c.replace("_", " ")[:18] for c in classes]
        fs = 5.5 if n <= 10 else 3.6
        ax.set_xticks(range(n)); ax.set_xticklabels(lab, rotation=90, fontsize=fs)
        ax.set_yticks(range(n)); ax.set_yticklabels(lab, fontsize=fs)
        ax.set_title(DATASET_NAMES[d], fontsize=8); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    save(fig, "fig_confusion")


def fig_roc():
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    fig, axes = plt.subplots(1, len(DSS), figsize=(FULLW, 2.4))
    axes = np.atleast_1d(axes)
    for ax, d in zip(axes, DSS):
        show = [("classical", GRAY), ("alexnet", YELLOW), ("mobilenetv2", AQUA), ("resnet50", ORANGE), ("proposed", BLUE)]
        for m, c in show:
            if m == "proposed":
                z = np.load(os.path.join(RES, f"{d}_proposed_test.npz")); p, y = z["clean_casc"], z["y"]
            else:
                f = os.path.join(RES, f"{d}_{m}_test.npz")
                if not os.path.exists(f): continue
                z = np.load(f); p, y = z["prob"], z["y"]
            yb = label_binarize(y, classes=range(p.shape[1]))
            fpr, tpr, _ = roc_curve(yb.ravel(), p.ravel())
            ax.plot(fpr, tpr, color=c, lw=2.0 if m == "proposed" else 1.3, label=f"{NAMES[m]} ({auc(fpr, tpr):.3f})")
        ax.plot([0, 1], [0, 1], color=GRID, lw=1, ls="--")
        ax.set_xlim(-0.01, 0.4); ax.set_ylim(0.6, 1.005)
        ax.set_title(DATASET_NAMES[d] + " (micro-avg ROC, zoom)"); ax.set_xlabel("False positive rate")
        ax.legend(fontsize=5.5, loc="lower right")
    axes[0].set_ylabel("True positive rate")
    save(fig, "fig_roc")


def fig_tradeoff():
    eff = J("efficiency.json"); p = J("kvasir_proposed.json")
    if not eff or not p: return
    rows = results_table("kvasir")
    fig, ax = plt.subplots(figsize=(COLW, 2.5))
    for m in ORDER[1:]:
        if m not in rows: continue
        e = eff[m]
        ax.scatter(e["fps"], rows[m]["acc"] * 100, s=12 + e["params_m"] * 2.2, color=GRAY, edgecolor="white", linewidth=1, zorder=3)
        ax.annotate(NAMES[m], (e["fps"], rows[m]["acc"] * 100), xytext=(4, -3), textcoords="offset points", fontsize=6, color=INK2)
    c = cascade_cost("kvasir")
    ax.scatter(c["gpu_fps"], rows["proposed"]["acc"] * 100, s=60, marker="*", color=BLUE, zorder=4, edgecolor="white")
    ax.annotate("Proposed cascade", (c["gpu_fps"], rows["proposed"]["acc"] * 100), xytext=(4, 3),
                textcoords="offset points", fontsize=6.5, color=INK, fontweight="bold")
    ax.set_xscale("log"); ax.set_xlabel("GPU throughput, batch-1 (FPS, log scale)"); ax.set_ylabel("Kvasir test accuracy (%)")
    save(fig, "fig_accuracy_vs_fps")


def fig_prevalence():
    """Expected cascade cost as a function of the share of normal frames:
    measured GPU throughput (left) and hardware-independent GMACs (right)."""
    eff = J("efficiency.json")
    if not eff or "stage23" not in eff: return
    pis = np.linspace(0, 1, 51)
    fig, axes = plt.subplots(1, 2, figsize=(FULLW, 2.2))
    for ax, key, ref, title in ((axes[0], "gpu_fps", lambda m: eff[m]["fps"], "Expected GPU throughput (FPS, higher is better)"),
                                (axes[1], "gmacs", lambda m: eff[m]["gmacs"], "Expected GMACs per frame (lower is better)")):
        for d in DSS:
            ax.plot(pis * 100, [cascade_cost(d, pi)[key] for pi in pis], color=DS_COLORS[d],
                    label=f"Proposed ({DATASET_NAMES[d]} gate)")
        for m, ls in (("resnet50", "--"), ("densenet121", ":"), ("vitb16", "-.")):
            ax.axhline(ref(m), color=GRAY, ls=ls, lw=1.2, label=NAMES[m] + " (every frame)")
        ax.axvline(37.5, color=INK2, lw=0.8, ls=":")
        ax.set_xlabel("Share of normal frames in the stream (%)"); ax.set_title(title, fontsize=8)
    axes[1].set_yscale("log"); axes[0].legend(fontsize=5.5, loc="upper left")
    save(fig, "fig_prevalence")


def fig_tau():
    fig, axes = plt.subplots(1, 2, figsize=(FULLW, 2.2), sharey=True)
    for ax, k, lab in ((axes[0], "gate_recall", "Abnormal recall at gate"), (axes[1], "bypass", "Frames bypassed")):
        for d in DSS:
            p = J(f"{d}_proposed.json"); s = p["tau_sweep_val"]
            ax.plot([e["tau"] for e in s], [e[k] * 100 for e in s], color=DS_COLORS[d], label=DATASET_NAMES[d])
            ax.axvline(p["tau"], color=DS_COLORS[d], lw=0.8, ls=":")
        ax.set_xlabel(r"Gate threshold $\tau$"); ax.set_title(lab + " (%)", fontsize=8)
    axes[0].legend(); axes[0].set_ylabel("%")
    save(fig, "fig_tau_sweep")


def fig_robustness():
    """Accuracy under artifact conditions: single-frame baselines vs proposed (Kvasir)."""
    rb = J("kvasir_robustness.json"); p = J("kvasir_proposed.json")
    if not rb or not p: return
    conds = ["clean"] + [f"transient_{i}" for i in range(4)] + ["stream_p30"]
    clab = ["Clean"] + ARTIFACT_NAMES + ["Random\nstream"]
    models = [m for m in ORDER[1:] if m in rb]
    fig, ax = plt.subplots(figsize=(FULLW, 2.4))
    w = 0.8 / (len(models) + 1)
    for j, m in enumerate(models):
        ax.bar(np.arange(len(conds)) + (j - len(models) / 2) * w, [rb[m][c]["acc"] * 100 for c in conds], width=w * 0.92,
               color=GRAY, alpha=0.35 + 0.65 * j / len(models), edgecolor="white", linewidth=0.4,
               label="Single-frame baselines" if j == 0 else None)
    pv = [p["conditions"][c]["Proposed cascade"]["acc"] * 100 for c in conds]
    ax.bar(np.arange(len(conds)) + (len(models) / 2) * w, pv, width=w * 0.92, color=BLUE, label="Proposed (temporal)")
    for i, v in enumerate(pv):
        ax.text(i + (len(models) / 2) * w, v + 0.8, f"{v:.1f}", ha="center", fontsize=6, color=INK)
    ax.set_xticks(range(len(conds))); ax.set_xticklabels(clab)
    ax.set_ylabel("Kvasir test accuracy (%)"); ax.set_ylim(0, 105)
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="x", visible=False)
    save(fig, "fig_robustness")


def fig_attention():
    z = np.load(os.path.join(RES, "kvasir_proposed_test.npz"))
    att, mk = z["stream_p30_att"], z["stream_p30_mask"]
    x = np.load(os.path.join(CACHE, "kvasir_cond_stream_p30_x.npy"), mmap_mode="r")
    fig, axes = plt.subplots(1, 2, figsize=(FULLW, 1.9), gridspec_kw=dict(width_ratios=[1, 1.3]))
    # attention mass on corrupted vs clean frames, by window position
    pos = np.arange(att.shape[1])
    for lab, sel, c in (("clean frames", ~mk, BLUE), ("corrupted frames", mk, ORANGE)):
        axes[0].plot(pos + 1, [att[:, t][sel[:, t]].mean() for t in pos], marker="o", ms=4, color=c, label=lab)
    axes[0].set_xlabel("Position in window (5 = current)"); axes[0].set_ylabel("Mean attention weight")
    axes[0].legend(); axes[0].set_xticks(pos + 1)
    # histogram
    axes[1].hist(att[~mk], bins=40, color=BLUE, alpha=0.7, density=True, label="clean")
    axes[1].hist(att[mk], bins=40, color=ORANGE, alpha=0.7, density=True, label="corrupted")
    axes[1].set_xlabel("Bi-LSTM temporal attention weight"); axes[1].set_ylabel("Density"); axes[1].legend()
    save(fig, "fig_attention")


def fig_artifacts():
    """Example of a synthetic clip and every artifact type."""
    import torch
    from common import ARTIFACTS, make_windows, to_input
    x, y, split, classes, _ = load_split("kvasir")
    i = np.where((split == "test") & (y == classes.index("polyps")))[0][3]
    xb = to_input(torch.from_numpy(np.ascontiguousarray(x[i:i + 1])), "cpu")
    gen = torch.Generator().manual_seed(5)
    clip, _ = make_windows(xb, gen)
    fig, axes = plt.subplots(2, 5, figsize=(FULLW, 3.0))
    for t in range(5):
        axes[0, t].imshow(clip[0, t].permute(1, 2, 0).numpy()); axes[0, t].set_title(f"frame t-{4 - t}" if t < 4 else "frame t (current)", fontsize=7)
    axes[1, 0].imshow(clip[0, -1].permute(1, 2, 0).numpy()); axes[1, 0].set_title("clean", fontsize=7)
    for k, (fn, nm) in enumerate(zip(ARTIFACTS, ARTIFACT_NAMES)):
        axes[1, k + 1].imshow(fn(clip[0, -1:].clone(), torch.Generator().manual_seed(k))[0].permute(1, 2, 0).clamp(0, 1).numpy())
        axes[1, k + 1].set_title(nm, fontsize=7)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values(): s.set_visible(False)
    save(fig, "fig_artifacts")


def fig_gradcam():
    """Per database: image with the expert mask outline (top) and the Grad-CAM
    overlay with the same outline (bottom)."""
    z = J("gradcam_localisation.json")
    if not z: return
    from scipy import ndimage
    e = np.load(os.path.join(RES, "gradcam_examples.npz"))
    exts = [d for d in ("kvasirseg", "cvc") if f"{d}_x" in e]
    nc = 6
    fig, axes = plt.subplots(2 * len(exts), nc, figsize=(FULLW, 1.22 * 2 * len(exts)))
    fig.subplots_adjust(wspace=0.04, hspace=0.06)
    for r, d in enumerate(exts):
        for c in range(nc):
            img, m, cam = e[f"{d}_x"][c], e[f"{d}_m"][c].astype(bool), e[f"{d}_cam"][c]
            edge = m ^ ndimage.binary_erosion(m, iterations=2)
            a0, a1 = axes[2 * r, c], axes[2 * r + 1, c]
            im0 = img.copy(); im0[edge] = [255, 255, 255]
            a0.imshow(im0)
            a1.imshow(img); a1.imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
            a1.contour(m, levels=[0.5], colors="white", linewidths=0.8)
        axes[2 * r, 0].set_ylabel(DATASET_NAMES[d] + "\nimage + mask", fontsize=6.5)
        axes[2 * r + 1, 0].set_ylabel("Grad-CAM", fontsize=6.5)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values(): sp.set_visible(False)
    save(fig, "fig_gradcam")


def fig_tsne():
    from sklearn.manifold import TSNE
    f = np.load(os.path.join(RES, "kvasir_test_feat.npy")).astype(np.float32)
    z = np.load(os.path.join(RES, "kvasir_proposed_test.npz")); y = z["y"]
    classes = J("kvasir_proposed.json")["classes"]
    e = TSNE(2, random_state=0, init="pca", perplexity=30).fit_transform(f)
    fig, ax = plt.subplots(figsize=(COLW, 2.8))
    for k, c in enumerate(classes):
        ax.scatter(e[y == k, 0], e[y == k, 1], s=3, color=CAT[k], label=c, linewidths=0)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.legend(fontsize=5.5, markerscale=2.5, loc="center left", bbox_to_anchor=(1, 0.5))
    save(fig, "fig_tsne")


def fig_perclass():
    from sklearn.metrics import f1_score
    z = np.load(os.path.join(RES, "kvasir_proposed_test.npz")); y = z["y"]
    classes = J("kvasir_proposed.json")["classes"]
    fig, ax = plt.subplots(figsize=(COLW, 2.3))
    show = [("classical", GRAY, None), ("resnet50", ORANGE, None), ("proposed", BLUE, None)]
    w = 0.27
    for j, (m, c, _) in enumerate(show):
        p = z["clean_casc"] if m == "proposed" else np.load(os.path.join(RES, f"kvasir_{m}_test.npz"))["prob"]
        f1 = f1_score(y, p.argmax(1), average=None, labels=range(len(classes)))
        ax.bar(np.arange(len(classes)) + (j - 1) * w, f1 * 100, width=w * 0.92, color=c, label=NAMES[m])
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=35, ha="right", fontsize=6)
    ax.set_ylabel("Per-class F1 (%)"); ax.set_ylim(40, 101); ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1))
    ax.grid(axis="x", visible=False)
    save(fig, "fig_perclass_f1")


def fig_deck_samples():
    """One representative frame per database, for the presentation."""
    picks = [("kvasir", "polyps"), ("gastrovision", "Gastric polyps"), ("kvasirseg", None), ("cvc", None)]
    fig, axes = plt.subplots(1, 4, figsize=(8, 2.25))
    for ax, (d, c) in zip(axes, picks):
        p = os.path.join(CACHE, f"{d}_x.npy")
        if not os.path.exists(p):
            ax.axis("off"); continue
        if c:
            x, y, split, classes, _ = load_split(d)
            i = np.where(y == classes.index(c))[0][7] if c in classes else 0
            img, lab = x[i], {"esophagitis-b-d": "oesophagitis (grade B-D)"}.get(c, c)
        else:
            x = np.load(p, mmap_mode="r"); m = np.load(os.path.join(CACHE, f"{d}_mask.npy"), mmap_mode="r")
            img = x[11].copy(); edge = m[11].astype(bool) ^ np.roll(m[11].astype(bool), 2, 0)
            img[edge] = [255, 255, 255]; lab = "polyp + expert mask outline"
        ax.imshow(img); ax.set_title(f"{DATASET_NAMES[d]}\n{lab}", fontsize=8, color=INK)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values(): sp.set_visible(False)
    plt.tight_layout(pad=0.4)
    save(fig, "fig_deck_samples")


if __name__ == "__main__":
    which = sys.argv[1:] or [n for n in dir() if n.startswith("fig_")]
    for n in which:
        try:
            globals()[n]()
        except Exception as ex:  # keep going so partial results still render
            print("FAILED", n, repr(ex))
