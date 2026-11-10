"""Phase-4 analysis on B held-out folds: grid sensitivity plot, confusion pairs, failure cases.

    python scripts/bridge_analysis.py [--out runs/p4/analysis.md] [--plot docs/figs/bridge_sensitivity.png]

Reads runs/p4/grid_s0 (seed-0 grid), runs/p4/chosen.json (from scripts/bridge_results.py final),
runs/p1/semi_s{s}, runs/p4/recheck, runs/p4/soft. B labels are used for evaluation only; B captions
are shown for failure cases (B captions are training data, so reading them is allowed).
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="runs/p4/analysis.md")
ap.add_argument("--plot", default="docs/figs/bridge_sensitivity.png")
args = ap.parse_args()
sys.path.insert(0, os.getcwd())
from omegaconf import OmegaConf  # noqa: E402

from src.data import class_prompt_names, load_classes  # noqa: E402

NAME = re.compile(r"(caption|image)_l([\d.]+)_e([\d.]+)_k([\d.]+)(?:_(uniform|caption))?$")
dcfg = OmegaConf.load("configs/data/kfold5.yaml")
names = class_prompt_names(load_classes(dcfg.manifest_dir), {})
b = pd.read_csv("data/splits/B_labeled.csv")
row_of = {p: i for i, p in enumerate(b.filepath)}
y = b.label.values
C = len(names)


def latest(pattern):
    ds = [d for d in sorted(glob.glob(pattern)) if os.path.exists(os.path.join(d, "val_logits_fold0.pt"))]
    return ds[-1] if ds else None


def mr1_acc(run):
    m = json.load(open(os.path.join(run, "metrics.json")))
    f = lambda k: 100 * np.mean([x["metrics"][k] for x in m["folds"]])
    return f("zs_acc"), (f("i2t_R@1") + f("t2i_R@1")) / 2


def pooled_pred(run):
    pred = np.full(len(b), -1)
    for k in range(5):
        d = torch.load(os.path.join(run, f"val_logits_fold{k}.pt"), weights_only=False)
        pred[[row_of[p] for p in d["filepath"]]] = d["cos"].argmax(1)
    assert (pred >= 0).all()
    return pred


lines = []
out = lines.append

# ------------------------------------------------------------------ sensitivity plot over the seed-0 grid
grid = []
for d in sorted(glob.glob("runs/p4/grid_s0/*")):
    run = latest(os.path.join(d, "*"))
    if run:
        src, lam, eps, kap, prior = NAME.match(os.path.basename(d)).groups()
        acc, mr1 = mr1_acc(run)
        grid.append(dict(source=src, lam=float(lam), eps=float(eps), kappa=float(kap), prior=prior or "-",
                         acc=acc, mR1=mr1))
grid = pd.DataFrame(grid)
m2 = [mr1_acc(latest(f"runs/p1/semi_s{s}/*")) for s in range(3)]
m2_acc0, m2_mr10 = m2[0]

if args.plot:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    styles = {(0.0, "-"): ("#555555", "κ=0"), (0.5, "uniform"): ("#1f77b4", "κ=0.5 uniform"),
              (1.0, "uniform"): ("#2ca02c", "κ=1 uniform"), (0.5, "caption"): ("#ff7f0e", "κ=0.5 caption prior"),
              (1.0, "caption"): ("#d62728", "κ=1 caption prior")}
    for j, src in enumerate(["caption", "image"]):
        for i, (met, ref) in enumerate([("acc", m2_acc0), ("mR1", m2_mr10)]):
            ax = axes[i, j]
            for (kap, prior), (col, lbl) in styles.items():
                for eps, ls in [(1.0, "-"), (0.5, "--")]:
                    sub = grid[(grid.source == src) & (grid.kappa == kap) & (grid.prior == prior) & (grid.eps == eps)]
                    if len(sub) == 0 or (kap == 0 and eps == 0.5):     # κ=0, hard targets: ε has no effect
                        continue
                    sub = sub.sort_values("lam")
                    ax.plot(sub.lam, sub[met], ls, marker="o", color=col,
                            label=f"{lbl}" + ("" if kap == 0 else f", ε={eps:g}"))
            ax.axhline(ref, color="k", lw=0.8, ls=":", label="method 2 (seed 0)")
            if met == "mR1":
                ax.axhline(ref - 0.5, color="k", lw=0.6, ls="-.", label="eligibility (−0.5)")
            ax.set_xscale("log", base=2)
            ax.set_xticks([0.25, 0.5, 1.0], ["0.25", "0.5", "1"])
            ax.set_title(f"source={src}: {'accuracy' if met == 'acc' else 'mean R@1'} (%)")
            if i == 1:
                ax.set_xlabel("λ_bridge")
    axes[0, 0].legend(fontsize=7, ncol=2)
    axes[1, 0].legend(fontsize=7, ncol=2, loc="lower left")
    fig.suptitle("Bridge grid on B held-out folds (seed 0, 5-fold mean)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.plot), exist_ok=True)
    fig.savefig(args.plot, dpi=130)

# κ effect at fixed (source, λ, ε, prior): mean over the grid
out("### Grid marginals (seed 0; mean over the other grid axes)\n")
out("| source | factor | value | acc | mR1 | configs |")
out("|---|---|---|---|---|---|")
for src in ("caption", "image"):
    g = grid[grid.source == src]
    for fac in ("lam", "eps", "kappa", "prior"):
        for v, sub in g.groupby(fac):
            out(f"| {src} | {fac} | {v} | {sub.acc.mean():.2f} | {sub.mR1.mean():.2f} | {len(sub)} |")
out(f"\nMethod 2 seed 0: acc {m2_acc0:.2f}, mR1 {m2_mr10:.2f}.\n")

# ------------------------------------------------------------------ confusions and failure cases
chosen = json.load(open("runs/p4/chosen.json"))["chosen"]
EXPS = [("#1 method 2", [latest(f"runs/p1/semi_s{s}/*") for s in range(3)])]
for lbl, key in [("#2 bridge κ=0", "#2 bridge kappa=0"), ("#3 balanced bridge (main)", "#3 balanced bridge (main)")]:
    if chosen.get(key):
        n = chosen[key]
        EXPS.append((lbl, [latest(f"runs/p4/grid_s0/{n}/*")] + [latest(f"runs/p4/recheck/{n}_s{s}/*") for s in (1, 2)]))
preds = {lbl: [pooled_pred(r) for r in runs] for lbl, runs in EXPS}

out("### Most frequent confusions (true → predicted; images per seed, mean over seeds 0–2)\n")
pairs = {}
for lbl, ps in preds.items():
    cnt = np.zeros((C, C))
    for p in ps:
        np.add.at(cnt, (y, p), 1)
    cnt /= len(ps)
    np.fill_diagonal(cnt, 0)
    pairs[lbl] = cnt
union = set()
for cnt in pairs.values():
    idx = np.dstack(np.unravel_index(np.argsort(-cnt, axis=None)[:6], cnt.shape))[0]
    union |= {tuple(i) for i in idx}
union = sorted(union, key=lambda tp: -pairs["#1 method 2"][tp])
out("| true → predicted | " + " | ".join(l.split(" ")[0] for l in pairs) + " |")
out("|---|" + "---|" * len(pairs))
for t, p in union:
    out(f"| {names[t]} → {names[p]} | " + " | ".join(f"{pairs[l][t, p]:.1f}" for l in pairs) + " |")
out("| total errors | " + " | ".join(f"{pairs[l].sum():.1f}" for l in pairs) + " |\n")

main = "#3 balanced bridge (main)"
if main in preds:
    p0 = preds[main][0]
    m1 = preds["#1 method 2"][0]
    out("### Failure cases (#3 main, seed 0; held-out B images with their own B captions)\n")
    for cname in ("playfields", "sparse residential", "center"):
        c = names.index(cname)
        idx = np.where((y == c) & (p0 != c))[0][:4]
        out(f"**{cname}**: {(p0[y == c] != c).sum()} / {(y == c).sum()} wrong for #3 seed 0 "
            f"(method 2 seed 0: {(m1[y == c] != c).sum()}).\n")
        out("| image | #3 predicted | method 2 predicted | captions (unique, first 2) |")
        out("|---|---|---|---|")
        for i in idx:
            caps = list(dict.fromkeys(b.loc[i, [f"caption_{j}" for j in range(5)]].tolist()))[:2]
            out(f"| {os.path.basename(b.filepath[i])} | {names[p0[i]]} | {names[m1[i]]} | "
                + " / ".join(x.strip() for x in caps) + " |")
        out("")

text = "\n".join(lines)
print(text)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
open(args.out, "w").write(text + "\n")
