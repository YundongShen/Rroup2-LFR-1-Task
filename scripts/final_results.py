"""Phase-5 result tables on split C (docs/bridge_protocol.md §7). Reads runs only; never runs a model on C.

    python scripts/final_results.py [--root runs/p5] [--out runs/p5/results.md]

Runs: <root>/zeroshot (deterministic, once), <root>/<row>_s{0..4} for the trained rows. Every trained run has
5 fold models, each evaluated on all of C; per seed the 5 fold-model values are averaged, then mean ± std
(population) over seeds. TV and per-class accuracy come from test_logits_fold*.pt (argmax of the cosines to
the evaluation-template class embeddings, the same scores as zs_acc).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="runs/p5")
ap.add_argument("--out", default="runs/p5/results.md")
ap.add_argument("--seeds", default="0,1,2,3,4")
args = ap.parse_args()
sys.path.insert(0, os.getcwd())
from omegaconf import OmegaConf  # noqa: E402

from scripts.paired_stats import paired_t, wilcoxon_exact  # noqa: E402
from src.data import class_prompt_names, load_classes, load_test  # noqa: E402
from src.methods.semi import unbalanced_sinkhorn  # noqa: E402

SEEDS = [int(s) for s in args.seeds.split(",")]
dcfg = OmegaConf.load("configs/data/kfold5.yaml")
names = class_prompt_names(load_classes(dcfg.manifest_dir), {})
C = len(names)
c_df = load_test(dcfg.manifest_dir)
row_of = {p: i for i, p in enumerate(c_df.filepath)}
y = c_df.label.values
true_share = np.bincount(y, minlength=C) / len(y)
TTA = dict(kappa=1.0, eps=1.0)       # #6, frozen from runs/p4/chosen.json (uniform prior)
RET = ["i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10"]


def latest(pattern):
    ds = [d for d in sorted(glob.glob(pattern)) if os.path.exists(os.path.join(d, "metrics.json"))]
    assert ds, f"no run for {pattern}"
    return ds[-1]


def tta(d):
    r = torch.softmax(torch.tensor(d["scale"] * d["cos"], dtype=torch.float64), 1)
    pi = torch.full((C,), 1.0 / C, dtype=torch.float64)
    return unbalanced_sinkhorn(r, TTA["eps"], pi, TTA["kappa"]).argmax(1).numpy()


def seed_values(run, transform=None):
    """Per-seed values (mean over the fold models of one run)."""
    m = json.load(open(os.path.join(run, "metrics.json")))
    assert m["eval_split"] == "test", run
    per = []
    for f in m["folds"]:
        d = torch.load(os.path.join(run, f"test_logits_fold{f['fold']}.pt"), weights_only=False)
        idx = np.array([row_of[p] for p in d["filepath"]])
        assert len(idx) == len(y) and len(set(idx)) == len(y)
        pred = np.empty(len(y), int)
        pred[idx] = transform(d) if transform else d["cos"].argmax(1)
        rec = np.array([(pred[y == c] == c).mean() for c in range(C)])
        share = np.bincount(pred, minlength=C) / len(pred)
        acc_logits = 100 * (pred == y).mean()
        if transform is None:   # logits must reproduce the evaluated accuracy
            assert abs(acc_logits - 100 * f["metrics"]["zs_acc"]) < 0.2, (run, acc_logits, f["metrics"]["zs_acc"])
        v = {"acc": acc_logits if transform else 100 * f["metrics"]["zs_acc"],
             "mpc": 100 * rec.mean(), "tv": 100 * 0.5 * np.abs(share - true_share).sum(), "recall": rec}
        for k in RET:
            v[k] = 100 * f["metrics"][k]
        v["mR1"] = (v["i2t_R@1"] + v["t2i_R@1"]) / 2
        per.append(v)
    return {k: np.mean([p[k] for p in per], 0) for k in per[0]}, len(per)


ROWS = [("Zero-shot CLIP", "zeroshot", None, False),
        ("Supervised (method 1)", "sup", None, True),
        ("Semi-supervised (method 2, #1)", "semi", None, True),
        ("**Ours** (#3: λ 0.25, ε_b 1, κ 0.5, uniform)", "main", None, True),
        ("Ours, no balancing (#2, κ = 0)", "k0", None, True),
        ("Ours, image-derived labels (#4)", "image", None, True),
        ("Ours, soft targets (#5)", "soft", None, True),
        ("Ours, other prior (caption prior)", "main_capprior", None, True),
        ("Semi-sup. + test-time balancing † (#6)", "semi", tta, True)]

vals, nfold = {}, {}
for label, key, tf, trained in ROWS:
    runs = [latest(os.path.join(args.root, f"{key}_s{s}", "*")) for s in SEEDS] if trained \
        else [latest(os.path.join(args.root, key, "*"))]
    out_ = [seed_values(r, tf) for r in runs]
    vals[label] = [o[0] for o in out_]
    nfold[label] = (len(runs), out_[0][1])

pm = lambda xs: f"{np.mean(xs):.2f}" + (f" ± {np.std(xs):.2f}" if len(xs) > 1 else "")
lines = []
out = lines.append
out("### Main results and ablations (C, seeds 0–4; mean ± std over seeds; per seed: mean over the 5 fold models)\n")
out("| Method | Cls. acc | Mean per-class | TV dist. | I→T R@1/5/10 | T→I R@1/5/10 | mean R@1 |")
out("| --- | --- | --- | --- | --- | --- | --- |")
for label, *_ in ROWS:
    v = vals[label]
    g = lambda k: [s[k] for s in v]
    out(f"| {label} | {pm(g('acc'))} | {pm(g('mpc'))} | {pm(g('tv'))} | "
        + " / ".join(f"{np.mean(g(k)):.2f}" for k in RET[:3]) + " | "
        + " / ".join(f"{np.mean(g(k)):.2f}" for k in RET[3:]) + f" | {pm(g('mR1'))} |")
out("\n† Transductive (balances over all unlabeled C images jointly); not comparable to other rows; retrieval is "
    "that of method 2. Zero-shot is deterministic (one run). R@k: seed means; their stds are in summary.json.\n")

# ---- paired tests (same seed)
out("### Paired tests (same seed, n = 5; two-sided paired t-test, 4 d.o.f.; exact Wilcoxon signed-rank)\n")
out("| comparison | metric | per-seed differences | mean diff | std diff | t | p (t-test) | p (Wilcoxon) |")
out("|---|---|---|---|---|---|---|---|")
tests = {}
REF = "Semi-supervised (method 2, #1)"
for other in ["**Ours** (#3: λ 0.25, ε_b 1, κ 0.5, uniform)", "Semi-sup. + test-time balancing † (#6)"]:
    for k, kname in [("acc", "acc"), ("mpc", "per-class acc"), ("mR1", "mean R@1")]:
        if other.startswith("Semi-sup.") and k == "mR1":
            continue
        a, b = [s[k] for s in vals[other]], [s[k] for s in vals[REF]]
        m, sd, t, p = paired_t(a, b)
        pw = wilcoxon_exact(a, b)
        tests[f"{other}|{k}"] = {"diffs": list(np.subtract(a, b)), "mean": m, "sd": sd, "t": t, "p_t": p, "p_w": pw}
        out(f"| {other.replace('**', '')} − method 2 | {kname} | "
            + ", ".join(f"{d:+.2f}" for d in np.subtract(a, b)) + f" | {m:+.2f} | {sd:.2f} | {t:.2f} | {p:.3f} | {pw:.3f} |")
out("\nPrimary criterion (protocol §7): acc, ours vs method 2, paired t-test at α = 0.05. "
    "With n = 5 the Wilcoxon p-value cannot fall below 0.0625.\n")

# ---- per-class accuracy
FOCUS = {"center", "stadium", "playfields", "sparse residential", "commercial", "playground", "medium residential"}
cols = ["Zero-shot CLIP", "Supervised (method 1)", REF, "**Ours** (#3: λ 0.25, ε_b 1, κ 0.5, uniform)",
        "Ours, no balancing (#2, κ = 0)", "Semi-sup. + test-time balancing † (#6)"]
short = ["Zero-shot", "Method 1", "Semi-sup.", "Ours", "Ours κ=0", "TTA †"]
out("### Per-class accuracy on C (mean over seeds and fold models, %; ★ = focus classes)\n")
out("| Class | n | " + " | ".join(short) + " |")
out("|---|---|" + "---|" * len(cols))
rec = {c: np.mean([s["recall"] for s in vals[c]], 0) for c in cols}
for i in range(C):
    out(f"| {names[i]}{' ★' if names[i] in FOCUS else ''} | {(y == i).sum()} | "
        + " | ".join(f"{100 * rec[c][i]:.1f}" for c in cols) + " |")

text = "\n".join(lines)
print(text)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
open(args.out, "w").write(text + "\n")
summary = {label: {k: {"mean": float(np.mean([s[k] for s in v])), "std": float(np.std([s[k] for s in v])),
                       "per_seed": [float(s[k]) for s in v]}
                   for k in ["acc", "mpc", "tv", *RET, "mR1"]} | {"n_runs_folds": nfold[label]}
           for label, v in vals.items()}
json.dump({"rows": summary, "paired_tests": tests,
           "per_class": {c: dict(zip(names, map(float, rec[c]))) for c in cols}},
          open(os.path.join(os.path.dirname(args.out), "summary.json"), "w"), indent=1)
