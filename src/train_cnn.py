"""Fine-tune one ImageNet-pretrained model on one dataset.

AdamW + differential learning rate (backbone lr, head 10x lr) + cosine
annealing; cross-entropy (baselines) or focal loss (proposed stages).
Best epoch is selected on validation macro-F1; test-set probabilities are saved.

usage: python src/train_cnn.py --ds kvasir --model resnet50 --epochs 5 [--loss focal --tag _focal]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from common import (CKPT, RES, FocalLoss, augment, class_alpha, device, load_split,  # noqa: E402
                    metrics, normalize, save_json, seed_all, to_input)
from models import BASELINES, build  # noqa: E402


@torch.no_grad()
def predict(model, x_u8, dev, bs=128):
    model.eval(); out = []
    for i in range(0, len(x_u8), bs):
        xb = torch.from_numpy(np.ascontiguousarray(x_u8[i:i + bs]))
        with torch.autocast("mps", dtype=torch.float16, enabled=dev.type == "mps"):
            out.append(F.softmax(model(normalize(to_input(xb, dev))).float(), 1).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--epochs", type=int, default=5); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--loss", default="ce")
    ap.add_argument("--tag", default=""); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    seed_all(a.seed); dev = device()
    run = f"{a.ds}_{a.model}{a.tag}"
    if os.path.exists(os.path.join(RES, f"{run}_test.npz")):
        print("skip (done)", run); return
    x, y, split, classes, _ = load_split(a.ds)
    n_cls = len(classes)
    tr, va, te = (np.where(split == s)[0] for s in ("train", "val", "test"))
    xtr = torch.from_numpy(np.ascontiguousarray(x[tr])); ytr = torch.from_numpy(y[tr])
    xva, xte = np.ascontiguousarray(x[va]), np.ascontiguousarray(x[te])

    model, head = build(a.model, n_cls)
    model = model.to(dev)
    hid = {id(p) for p in head}
    groups = [{"params": [p for p in model.parameters() if id(p) not in hid], "lr": a.lr},
              {"params": head, "lr": a.lr * 10}]
    opt = torch.optim.AdamW(groups, weight_decay=1e-2)
    steps = a.epochs * (len(tr) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.lr * 10], total_steps=steps,
                                                pct_start=0.1, anneal_strategy="cos")
    crit = FocalLoss(class_alpha(y[tr], n_cls), gamma=2.0).to(dev) if a.loss == "focal" else torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("mps") if dev.type == "mps" else None

    hist, best, best_state = [], -1, None
    t0 = time.time()
    for ep in range(a.epochs):
        model.train(); perm = torch.randperm(len(tr)); tl, tc, nb = 0., 0, 0
        te0 = time.time()
        for i in range(0, len(tr) - a.bs + 1, a.bs):
            idx = perm[i:i + a.bs]
            xb = augment(to_input(xtr[idx], dev)); yb = ytr[idx].to(dev)
            with torch.autocast("mps", dtype=torch.float16, enabled=dev.type == "mps"):
                logits = model(normalize(xb))
                loss = crit(logits.float(), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            sched.step()
            tl += loss.item(); tc += (logits.argmax(1) == yb).sum().item(); nb += 1
        pv = predict(model, xva, dev)
        mv = metrics(y[va], pv, n_cls)
        vloss = float(F.nll_loss(torch.log(torch.from_numpy(pv).clamp_min(1e-8)), torch.from_numpy(y[va])))
        hist.append(dict(epoch=ep + 1, train_loss=tl / nb, train_acc=tc / (nb * a.bs), val_loss=vloss,
                         val_acc=mv["acc"], val_f1=mv["f1"], time=time.time() - te0))
        print(run, hist[-1], flush=True)
        if mv["f1"] > best:
            best = mv["f1"]; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_time = time.time() - t0
    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(CKPT, f"{run}.pt"))
    pt = predict(model, xte, dev)
    m = metrics(y[te], pt, n_cls)
    np.savez_compressed(os.path.join(RES, f"{run}_test.npz"), prob=pt, y=y[te], val_prob=predict(model, xva, dev), val_y=y[va])
    save_json(dict(run=run, ds=a.ds, model=a.model, name=BASELINES[a.model][0], loss=a.loss, epochs=a.epochs,
                   train_time_s=train_time, history=hist, test=m, classes=classes),
              os.path.join(RES, f"{run}.json"))
    print("TEST", run, {k: round(v, 4) for k, v in m.items()}, flush=True)


if __name__ == "__main__":
    main()
