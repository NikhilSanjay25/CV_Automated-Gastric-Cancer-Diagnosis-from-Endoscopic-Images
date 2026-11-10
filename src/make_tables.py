"""Write LaTeX tables (paper/tables/*.tex) and in-text number macros
(paper/tables/macros.tex) from results/*.json, so the paper always matches
the experiment outputs.

usage: python src/make_tables.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import ARTIFACT_NAMES, CACHE, DATASET_NAMES, RES, ROOT  # noqa: E402
from make_figures import DSS, NAMES, ORDER, J, cascade_cost, results_table  # noqa: E402

TAB = os.path.join(ROOT, "paper", "tables")
os.makedirs(TAB, exist_ok=True)
MAC = {}


def pct(v, d=2):
    return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * 100:.{d}f}"


def w(name, s):
    open(os.path.join(TAB, name + ".tex"), "w").write(s); print("table", name)


def bold_best(rows, keys, models):
    best = {k: max(rows[m][k] for m in models if m in rows) for k in keys}
    return lambda m, k, s: f"\\textbf{{{s}}}" if m in rows and abs(rows[m][k] - best[k]) < 1e-9 else s


def mac(name, val):
    MAC[name] = val


def t_datasets():
    lines = []
    use = {"kvasir": "8-class train/test (primary)", "hyperkvasir": "multi-class train/test (imbalanced)",
           "gastrovision": "multi-class train/test (upper+lower GI)", "cvc": "external polyp test + localisation",
           "kvasirseg": "external polyp test + localisation"}
    for d in ("kvasir", "gastrovision", "kvasirseg", "cvc"):
        p = os.path.join(CACHE, f"{d}_meta.json")
        if not os.path.exists(p): continue
        m = json.load(open(p))
        if "classes" in m:
            n = len(m["labels"]); c = len(m["classes"]); cnt = list(m["counts"].values())
            imb = f"{max(cnt) / min(cnt):.1f}"
            sp = " / ".join(str(sum(1 for s in m["split"] if s == k)) for k in ("train", "val", "test"))
            dropped = len(m.get("dropped", {}))
            mac(f"n{d}", f"{n:,}"); mac(f"c{d}", str(c)); mac(f"drop{d}", str(dropped))
        else:
            n = len(m["ids"]); c = "1 (+mask)"; imb = "--"; sp = "test only"
            mac(f"n{d}", f"{n:,}")
        lines.append(f"{DATASET_NAMES[d]} & {n:,} & {c} & {imb} & {sp} & {use[d]} \\\\")
    w("datasets", r"""\begin{table*}[t]
\caption{Databases used in this study (all public). Imbalance = largest/smallest class size.}
\label{tab:datasets}\centering
\begin{tabular}{lrrrll}\toprule
Database & Images & Classes & Imbalance & Train / Val / Test & Role in this study\\\midrule
""" + "\n".join(lines) + r"""
\bottomrule\end{tabular}\end{table*}
""")


def t_main():
    rows = results_table("kvasir"); eff = J("efficiency.json") or {}; p = J("kvasir_proposed.json")
    keys = ["acc", "precision", "recall", "f1", "specificity", "mcc", "kappa", "auc"]
    models = [m for m in ORDER + ["proposed"] if m in rows]
    B = bold_best(rows, keys, models)
    out = []
    for m in models:
        r = rows[m]
        cells = [B(m, k, pct(r[k])) for k in keys]
        cc = cascade_cost("kvasir")
        if m == "proposed" and cc:
            par = eff["mobilenetv2"]["params_m"] + eff["resnet50"]["params_m"] + eff["bilstm"]["params_m"]
            ef = [f"{par:.1f}", f"{cc['gmacs']:.2f}$^\\dagger$", f"{cc['gpu_fps']:.0f}$^\\dagger$", f"{cc['cpu_fps']:.1f}$^\\dagger$"]
        elif m in eff:
            ef = [f"{eff[m]['params_m']:.1f}", f"{eff[m]['gmacs']:.2f}", f"{eff[m]['fps']:.0f}", f"{eff[m]['cpu_fps']:.1f}"]
        else:
            ef = ["--", "--", "--", "--"]
        name = NAMES[m] if m != "proposed" else r"\textbf{Proposed cascade}"
        out.append(f"{name} & " + " & ".join(cells + ef) + r" \\" + (r"\midrule" if m == "vitb16" else ""))
    w("main_kvasir", r"""\begin{table*}[t]
\caption{Baseline comparison on the Kvasir v2 test set (1,200 images, 8 classes). Precision, recall, F1 and
specificity are macro-averaged; AUC is macro one-vs-rest. Best value per column in bold.
$^\dagger$Expected per-frame cost of the cascade given the measured gate pass rate (GPU: Apple M5, FP16; CPU: FP32).}
\label{tab:main}\centering\setlength{\tabcolsep}{4.2pt}
\begin{tabular}{lcccccccc|cccc}\toprule
 & \multicolumn{8}{c|}{Classification (\%)} & & & \multicolumn{2}{c}{FPS (batch 1)}\\
Model & Acc. & Prec. & Rec. & F1 & Spec. & MCC & $\kappa$ & AUC & Params (M) & GMACs & GPU & CPU\\\midrule
""" + "\n".join(out) + r"""
\bottomrule\end{tabular}\end{table*}
""")
    if p:
        r = rows["proposed"]
        for k in keys: mac(f"prop{k}", pct(r[k]))
        best_base = max((m for m in models if m != "proposed"), key=lambda m: rows[m]["acc"])
        mac("bestbase", NAMES[best_base]); mac("bestbaseacc", pct(rows[best_base]["acc"]))
        mac("bestbasemcc", pct(rows[best_base]["mcc"]))
        mac("gapbest", f"{(rows[best_base]['acc'] - rows['proposed']['acc']) * 100:.1f}")
        for m in models:
            if m != "proposed":
                mac(f"acc{m}", pct(rows[m]["acc"])); mac(f"mcc{m}", pct(rows[m]["mcc"])); mac(f"f1{m}", pct(rows[m]["f1"]))
        cc = cascade_cost("kvasir")
        if cc:
            mac("casfps", f"{cc['gpu_fps']:.0f}"); mac("casms", f"{cc['gpu_ms']:.1f}")
            mac("cascpufps", f"{cc['cpu_fps']:.1f}"); mac("cascpums", f"{cc['cpu_ms']:.1f}")
            mac("casgmacs", f"{cc['gmacs']:.2f}"); mac("caspass", pct(cc["pass_rate"], 1))
            e = eff
            mac("gatems", f"{e['mobilenetv2']['latency_ms']:.1f}"); mac("gatecpums", f"{e['mobilenetv2']['cpu_ms']:.1f}")
            mac("stagems", f"{e['stage23']['gpu_ms']:.1f}"); mac("stagecpums", f"{e['stage23']['cpu_ms']:.1f}")
            for pi_, nm in ((0.9, "ninety"), (0.8, "eighty")):
                c2 = cascade_cost("kvasir", pi_)
                mac(f"cascpu{nm}", f"{c2['cpu_fps']:.1f}"); mac(f"casgpu{nm}", f"{c2['gpu_fps']:.0f}")
                mac(f"casgmacs{nm}", f"{c2['gmacs']:.2f}")
            mac("ressinglems", f"{e['resnet50']['latency_ms']:.1f}")
        mac("tau", f"{p['tau']:.2f}"); mac("taucur", f"{p['tau_current_gate']:.2f}")
        g = p["conditions"]["clean"]["_gate"]
        mac("bypass", pct(g["bypass"], 1)); mac("gaterec", pct(g["gate_recall_abn"], 1)); mac("gatespec", pct(g["gate_spec_normal"], 1))
        mac("abnrec", pct(g["abn_detect_recall"], 2))
        for m in ("resnet50", "mobilenetv2", "vitb16", "vgg16", "efficientnetb0", "densenet121", "alexnet", "googlenet"):
            if m in eff:
                mac(f"fps{m}", f"{eff[m]['fps']:.0f}"); mac(f"cpufps{m}", f"{eff[m]['cpu_fps']:.1f}")
                mac(f"gmacs{m}", f"{eff[m]['gmacs']:.2f}"); mac(f"params{m}", f"{eff[m]['params_m']:.1f}")


def t_multi():
    keys = [("acc", "Acc."), ("f1", "F1"), ("mcc", "MCC")]
    dss = [d for d in ("gastrovision",) if d in DSS]
    if not dss: return
    tabs = {d: results_table(d) for d in dss}
    models = [m for m in ORDER + ["proposed"] if all(m in tabs[d] for d in dss)]
    Bs = {d: bold_best(tabs[d], [k for k, _ in keys] + ["bal_acc"], models) for d in dss}
    out = []
    for m in models:
        cells = []
        for d in dss:
            cells += [Bs[d](m, k, pct(tabs[d][m][k])) for k in ("acc", "bal_acc", "f1", "mcc")]
        name = NAMES[m] if m != "proposed" else r"\textbf{Proposed cascade}"
        out.append(f"{name} & " + " & ".join(cells) + r" \\" + (r"\midrule" if m == "vitb16" else ""))
    head = " & ".join(f"\\multicolumn{{4}}{{c}}{{{DATASET_NAMES[d]} ({len(J(f'{d}_proposed.json')['classes'])} classes)}}" for d in dss)
    sub = " & ".join(["Acc. & B-Acc. & F1 & MCC"] * len(dss))
    w("multi_db", r"""\begin{table*}[t]
\caption{Generalisation of the comparison to GastroVision (test split). B-Acc. = balanced accuracy
(mean per-class recall), which exposes failures on rare classes that plain accuracy hides.}
\label{tab:multidb}\centering
\begin{tabular}{l""" + "cccc" * len(dss) + r"""}\toprule
 & """ + head + r"""\\
Model & """ + sub + r"""\\\midrule
""" + "\n".join(out) + r"""
\bottomrule\end{tabular}\end{table*}
""")
    for d in dss:
        r = tabs[d]
        if "proposed" in r:
            for k in ("acc", "f1", "mcc", "bal_acc"): mac(f"{d}{k}".replace("_", ""), pct(r["proposed"][k]))
            bb = max((m for m in models if m != "proposed"), key=lambda m: r[m]["acc"])
            mac(f"{d}bestbase", NAMES[bb]); mac(f"{d}bestbaseacc", pct(r[bb]["acc"])); mac(f"{d}bestbasebacc", pct(r[bb]["bal_acc"]))


def t_ablation():
    out = []
    comps = ["ResNet-50 (focal), single frame", "+ temporal mean pooling", "+ Bi-LSTM (no gate)",
             "+ gate on current frame only", "Proposed cascade"]
    for d in DSS:
        p = J(f"{d}_proposed.json"); base = J(f"{d}_resnet50.json")
        rows = [("ResNet-50 (CE), single frame", base["test"] if base else None, None)]
        rb = J(f"{d}_robustness.json")
        if rb and "resnet50" in rb: rows[0] = (rows[0][0], rb["resnet50"]["clean"], rb["resnet50"]["stream_p30"])
        for c in comps:
            rows.append((c, p["conditions"]["clean"][c], p["conditions"]["stream_p30"][c]))
        out.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{DATASET_NAMES[d]}}}}}\\\\")
        for nm, cl, st in rows:
            if cl is None: continue
            out.append(f"\\quad {nm} & {pct(cl['acc'])} & {pct(cl['f1'])} & {pct(cl['mcc'])} & "
                       f"{pct(st['acc']) if st else '--'} & {pct(st['mcc']) if st else '--'}\\\\")
    w("ablation", r"""\begin{table}[t]
\caption{Ablation of the proposed components. ``Artifact stream'': every frame of the 5-frame window is
independently corrupted with probability 0.3.}
\label{tab:ablation}\centering\setlength{\tabcolsep}{3pt}
\begin{tabular}{lccccc}\toprule
 & \multicolumn{3}{c}{Clean} & \multicolumn{2}{c}{Artifact stream}\\
Configuration & Acc. & F1 & MCC & Acc. & MCC\\\midrule
""" + "\n".join(out) + r"""
\bottomrule\end{tabular}\end{table}
""")


def t_robust():
    rb = J("kvasir_robustness.json"); p = J("kvasir_proposed.json")
    if not rb or not p: return
    conds = ["clean"] + [f"transient_{i}" for i in range(4)] + ["stream_p30"]
    out = []
    models = [m for m in ORDER[1:] if m in rb]
    for m in models:
        vals = [rb[m][c]["acc"] for c in conds]
        out.append(f"{NAMES[m]} & " + " & ".join(pct(v, 1) for v in vals) + f" & {pct(np.mean(vals[1:5]), 1)}\\\\")
    pv = [p["conditions"][c]["Proposed cascade"]["acc"] for c in conds]
    out.append(r"\midrule\textbf{Proposed} & " + " & ".join(f"\\textbf{{{pct(v, 1)}}}" for v in pv) + f" & \\textbf{{{pct(np.mean(pv[1:5]), 1)}}}\\\\")
    mac("robtransient", pct(np.mean(pv[1:5]), 1))
    bm = max(models, key=lambda m: np.mean([rb[m][c]["acc"] for c in conds[1:5]]))
    mac("robbestbase", NAMES[bm]); mac("robbestbaseacc", pct(np.mean([rb[bm][c]["acc"] for c in conds[1:5]]), 1))
    mac("robstream", pct(pv[5], 1)); mac("robbestbasestream", pct(max(rb[m]["stream_p30"]["acc"] for m in models), 1))
    mac("resnettransient", pct(np.mean([rb["resnet50"][c]["acc"] for c in conds[1:5]]), 1))
    hdr = " & ".join(["Clean"] + ARTIFACT_NAMES + ["Stream", "Mean art."])
    w("robustness", r"""\begin{table*}[t]
\caption{Kvasir test accuracy (\%) when the \emph{current} frame is hit by a transient artifact (columns 2--5),
and on a random artifact stream ($p=0.3$ per frame). Single-frame models see only the current frame; the proposed
system additionally sees the four preceding frames.}
\label{tab:robust}\centering
\begin{tabular}{lccccccc}\toprule
Model & """ + hdr + r"""\\\midrule
""" + "\n".join(out) + r"""
\bottomrule\end{tabular}\end{table*}
""")


def t_cross():
    c = J("crossdata.json")
    if not c: return
    exts = [e for e in ("kvasirseg", "cvc") if e in c]
    models = ORDER[1:] + ["mobilenetv2_focal", "resnet50_focal", "proposed"]
    nm = dict(NAMES, mobilenetv2_focal="MobileNetV2 (focal)", resnet50_focal="ResNet-50 (focal)")
    out = []
    for m in models:
        if not all(m in c[e] for e in exts): continue
        cells = []
        for e in exts:
            cells += [pct(c[e][m]["polyp_recall"], 1), pct(c[e][m]["abnormal_recall"], 1)]
        out.append(f"{nm[m] if m != 'proposed' else chr(92) + 'textbf{Proposed cascade}'} & " + " & ".join(cells) + r"\\")
    for e in exts:
        mac(f"{e}n", str(c[e]["_n"])); mac(f"{e}ntotal", str(c[e]["_n_total"]))
        mac(f"{e}prop", pct(c[e]["proposed"]["abnormal_recall"], 1)); mac(f"{e}proppolyp", pct(c[e]["proposed"]["polyp_recall"], 1))
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{{DATASET_NAMES[e]} ($n$={c[e]['_n']})}}" for e in exts)
    w("crossdata", r"""\begin{table}[t]
\caption{Cross-database generalisation of Kvasir-trained models on unseen polyp databases: sensitivity (\%) for the
\emph{polyp} class and for \emph{any abnormal} class. Kvasir-SEG images duplicated in the Kvasir training split were removed.}
\label{tab:cross}\centering\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lcccc}\toprule
 & """ + head + r"""\\
Model & Polyp & Abnormal & Polyp & Abnormal\\\midrule
""" + "\n".join(out) + r"""
\bottomrule\end{tabular}\end{table}
""")


def t_gradcam():
    g = J("gradcam_localisation.json")
    if not g: return
    exts = [e for e in ("kvasirseg", "cvc") if e in g]
    out = []
    for m in g[exts[0]]:
        cells = []
        for e in exts:
            r = g[e][m]; cells += [pct(r["pointing_game"], 1), pct(r["energy_in_mask"], 1), pct(r["iou"], 1)]
        out.append(f"{m} & " + " & ".join(cells) + r"\\")
    for e in exts:
        mac(f"gc{e}pg", pct(g[e]["ResNet-50 (focal, proposed Stage 2)"]["pointing_game"], 1))
        mac(f"gc{e}area", pct(g[e][list(g[e])[0]]["mask_area"], 1))
    head = " & ".join(f"\\multicolumn{{3}}{{c}}{{{DATASET_NAMES[e]}}}" for e in exts)
    w("gradcam", r"""\begin{table}[t]
\caption{Quantitative Grad-CAM localisation against expert polyp masks (400 images per database): pointing-game
hit rate (PG), fraction of CAM energy inside the mask (EiM) and IoU of the CAM thresholded at 0.5.}
\label{tab:gradcam}\centering\setlength{\tabcolsep}{2.6pt}
\begin{tabular}{lcccccc}\toprule
 & """ + head + r"""\\
Model & PG & EiM & IoU & PG & EiM & IoU\\\midrule
""" + "\n".join(out) + r"""
\bottomrule\end{tabular}\end{table}
""")


def t_gate():
    out = []
    for d in DSS:
        p = J(f"{d}_proposed.json"); g = p["conditions"]["clean"]["_gate"]
        n_abn = sum(p["abnormal"]); n = len(p["abnormal"])
        out.append(f"{DATASET_NAMES[d]} & {n - n_abn}/{n_abn} & {p['tau']:.2f} & {pct(g['gate_recall_abn'], 1)} & "
                   f"{pct(g['gate_spec_normal'], 1)} & {pct(g['bypass'], 1)} & {(cascade_cost(d) or {}).get('gpu_fps', float('nan')):.0f}\\\\")
    w("gate", r"""\begin{table}[t]
\caption{Stage-1 gate on the clean test sets: normal/abnormal class split, selected threshold $\tau$, recall of
abnormal frames, specificity on normal frames, fraction of frames that bypass Stages 2--3, and expected cascade GPU FPS (batch 1).}
\label{tab:gate}\centering\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccccc}\toprule
Database & Norm./Abn. & $\tau$ & Recall & Spec. & Bypass & FPS\\\midrule
""" + "\n".join(out) + r"""
\bottomrule\end{tabular}\end{table}
""")


def t_stats():
    """Bootstrap 95% CI of accuracy (2,000 resamples) and exact McNemar test of
    each baseline against the proposed cascade, on every classification database."""
    from scipy.stats import binomtest
    out, cols = [], []
    for d in DSS:
        z = np.load(os.path.join(RES, f"{d}_proposed_test.npz")); y = z["y"]; pp = z["clean_casc"].argmax(1)
        idx = np.random.default_rng(0).integers(0, len(y), (2000, len(y)))
        res = {}
        for m in ORDER + ["proposed"]:
            f = os.path.join(RES, f"{d}_{m}_test.npz")
            if m != "proposed" and not os.path.exists(f): continue
            pb = pp if m == "proposed" else np.load(f)["prob"].argmax(1)
            lo, hi = np.percentile((pb == y)[idx].mean(1), [2.5, 97.5])
            b = int(((pb == y) & (pp != y)).sum()); c = int(((pb != y) & (pp == y)).sum())
            pv = binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) and m != "proposed" else None
            res[m] = (float((pb == y).mean()), lo, hi, pv, b > c)
        cols.append((d, res))
        if d == "kvasir":
            sig_worse = [NAMES[m] for m, r in res.items() if r[3] is not None and r[3] < 0.05 and not r[4]]
            sig_better = [NAMES[m] for m, r in res.items() if r[3] is not None and r[3] < 0.05 and r[4]]
            ns = [NAMES[m] for m, r in res.items() if r[3] is not None and r[3] >= 0.05]
            mac("sigworse", ", ".join(sig_worse) or "none"); mac("sigbetter", ", ".join(sig_better) or "none")
            mac("signs", ", ".join(ns) or "none")
            mac("propcilo", f"{res['proposed'][1] * 100:.1f}"); mac("propcihi", f"{res['proposed'][2] * 100:.1f}")
    models = [m for m in ORDER + ["proposed"] if all(m in r for _, r in cols)]
    fp = lambda p: "--" if p is None else ("$<$0.001" if p < 1e-3 else f"{p:.3f}")
    for m in models:
        cells = []
        for d, r in cols:
            a, lo, hi, pv, better = r[m]
            mark = "" if pv is None or pv >= 0.05 else ("$^{+}$" if better else "$^{-}$")
            cells += [f"{a * 100:.1f} [{lo * 100:.1f}, {hi * 100:.1f}]", fp(pv) + mark]
        nm = NAMES[m] if m != "proposed" else r"\textbf{Proposed cascade}"
        out.append(f"{nm} & " + " & ".join(cells) + r"\\" + (r"\midrule" if m == "vitb16" else ""))
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{{DATASET_NAMES[d]}}}" for d, _ in cols)
    sub = " & ".join(["Acc. [95\\% CI] & $p$"] * len(cols))
    w("stats", r"""\begin{table*}[t]
\caption{Statistical comparison with the proposed cascade on the clean test sets: accuracy with a 2,000-sample
bootstrap 95\% confidence interval and the exact McNemar $p$-value against the proposed system ($^{+}$/$^{-}$: baseline
significantly better/worse at $p<0.05$).}
\label{tab:stats}\centering\setlength{\tabcolsep}{4pt}
\begin{tabular}{l""" + "cc" * len(cols) + r"""}\toprule
 & """ + head + r"""\\
Model & """ + sub + r"""\\\midrule
""" + "\n".join(out) + r"""
\bottomrule\end{tabular}\end{table*}
""")


def write_macros():
    s = "% auto-generated by src/make_tables.py\n"
    for k, v in sorted(MAC.items()):
        k2 = "R" + "".join(ch for ch in k if ch.isalpha())
        s += f"\\newcommand{{\\{k2}}}{{{v}}}\n"
    # any macro the paper uses that has no value yet renders as a visible ?? placeholder
    import glob, re
    have = {"R" + "".join(ch for ch in k if ch.isalpha()) for k in MAC}
    used = set()
    for f in glob.glob(os.path.join(ROOT, "paper", "**", "*.tex"), recursive=True):
        if "tables" in f: continue
        used |= set(re.findall(r"\\(R[a-z]+)", open(f).read()))
    missing = sorted(used - have)
    for m in missing:
        s += f"\\newcommand{{\\{m}}}{{\\textbf{{??}}}}\n"
    open(os.path.join(TAB, "macros.tex"), "w").write(s); print(len(MAC), "macros; missing:", missing)


if __name__ == "__main__":
    for f in (t_datasets, t_main, t_multi, t_ablation, t_robust, t_cross, t_gradcam, t_gate, t_stats):
        try:
            f()
        except Exception as ex:
            import traceback; traceback.print_exc(); print("FAILED", f.__name__)
    write_macros()
