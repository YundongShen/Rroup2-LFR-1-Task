"""Aggregate scripts/diagnose_bias.py outputs over seeds and apply the pre-registered criteria (TASK.md §4.4).

    python scripts/diagnosis_report.py [--root runs/diagnosis] [--out docs/diagnosis_tables.md]
                                       [--plot docs/figs/pred_vs_true.png]

Input: <root>/s<seed>/{summary.json, per_class.csv}, one directory per seed (methods: zeroshot,
supervised = method 1, semi = method 2). Values are mean ± std over seeds (population std).
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="runs/diagnosis")
ap.add_argument("--out", default=None)
ap.add_argument("--plot", default=None)
ap.add_argument("--logits_root", default=None,
                help="e.g. runs/p1: reads <root>/{sup,semi}_s*/<latest>/val_logits_fold*.pt for confusion pairs")
args = ap.parse_args()

dirs = sorted(glob.glob(os.path.join(args.root, "s*")))
S = [json.load(open(os.path.join(d, "summary.json"))) for d in dirs]
P = [pd.read_csv(os.path.join(d, "per_class.csv")) for d in dirs]
METHODS = [("zeroshot", "zero-shot"), ("supervised", "method 1"), ("semi", "method 2")]
FOCUS = ["center", "stadium", "playfields", "sparse residential", "commercial"]
lines = []
out = lines.append


def vals(m, k):
    return np.array([s[m][k] for s in S], float)


def ms(m, k, pct=True):
    v = vals(m, k) * (100 if pct else 1)
    return f"{v.mean():.2f} ± {v.std():.2f}" if len(v) > 1 else f"{v.mean():.2f}"


out(f"Seeds: {', '.join(os.path.basename(d) for d in dirs)} (zero-shot is deterministic).\n")

# ------------------------------------------------------------------ hypothesis tables
groups = {
    "H1 class bias": [("acc", True), ("mean_per_class_acc", True), ("tv_pred_vs_true", True),
                      ("n_never_predicted", False), ("n_under_predicted(<0.5x)", False),
                      ("n_over_predicted(>2x)", False), ("err_share_into_over_predicted", True),
                      ("acc_after_balancing_uniform", True), ("mpc_after_balancing_uniform", True),
                      ("acc_after_balancing_oracle_prior", True), ("mpc_after_balancing_oracle_prior", True)],
    "H2 conversion": [("gt_captions/acc_caption_only", True), ("gt_captions/caption_right_when_image_wrong", True),
                      ("gt_captions/acc_image_x_caption", True), ("pseudo_captions/acc_caption_only", True),
                      ("pseudo_captions/caption_right_when_image_wrong", True),
                      ("pseudo_captions/acc_image_x_caption", True)],
    "H3 near-synonyms": [("err_share_into_3_text_nearest", True), ("chance_level", True),
                         ("spearman(text_sim, confusion)", False)],
}
for title, keys in groups.items():
    out(f"### {title}\n")
    out("| metric | " + " | ".join(lbl for _, lbl in METHODS) + " |")
    out("|---|" + "---|" * len(METHODS))
    for k, pct in keys:
        out(f"| {k}{' (%)' if pct else ''} | " + " | ".join(ms(m, k, pct) for m, _ in METHODS) + " |")
    out("")

out(f"Class prior: TV(uniform, true) = {100 * S[0]['class_prior']['tv_uniform_vs_true']:.2f} %, "
    f"TV(caption prior, true) = {100 * S[0]['class_prior']['tv_caption_prior_vs_true']:.2f} %.\n")

# ------------------------------------------------------------------ pre-registered criteria (method 2)
out("### Pre-registered criteria (method 2, mean over seeds; per-seed in brackets)\n")


def per_seed(f):
    return [f(s) for s in S]


gain_bal = per_seed(lambda s: s["semi"]["acc_after_balancing_uniform"] - s["semi"]["acc"])
n_under = per_seed(lambda s: s["semi"]["n_under_predicted(<0.5x)"])
crw = per_seed(lambda s: s["semi"]["pseudo_captions/caption_right_when_image_wrong"])
gain_x = per_seed(lambda s: s["semi"]["pseudo_captions/acc_image_x_caption"] - s["semi"]["acc"])
near = per_seed(lambda s: s["semi"]["err_share_into_3_text_nearest"] / s["semi"]["chance_level"])
fmt = lambda v, p=100: f"{np.mean(v) * p:.2f} [{', '.join(f'{x * p:.2f}' for x in v)}]"
h1 = np.mean(gain_bal) > 0.02 and np.mean(n_under) > 0
h2 = np.mean(crw) >= 0.30 and np.mean(gain_x) > 0.01
h3 = np.mean(near) >= 2
out("| hypothesis | criterion | value | supported |")
out("|---|---|---|---|")
out(f"| H1 bias | acc_after_balancing_uniform − acc > 2 pts | {fmt(gain_bal)} pts | "
    f"{'yes' if h1 else 'no'} |")
out(f"| | and ≥1 class never / under-predicted (<0.5× true share) | {fmt(n_under, 1)} classes | |")
out(f"| H2 conversion | pseudo caption_right_when_image_wrong ≥ 30 % | {fmt(crw)} % | {'yes' if h2 else 'no'} |")
out(f"| | and acc_image_x_caption − acc > 1 pt | {fmt(gain_x)} pts | |")
out(f"| H3 near-synonyms | err_share_into_3_text_nearest ≥ 2 × chance | {fmt(near, 1)} × chance | "
    f"{'yes' if h3 else 'no'} |")
out("")

# ------------------------------------------------------------------ per-class table
pc = P[0][["class", "n", "true_share"]].copy()
cols = ["pred_share_zeroshot", "recall_zeroshot", "pred_share_supervised", "recall_supervised",
        "pred_share_semi", "recall_semi", "recall_gt_captions_only", "recall_pseudo_captions_only"]
for c in cols:
    stack = np.stack([p[c].values for p in P])
    pc[c] = stack.mean(0)
    pc[c + "_std"] = stack.std(0)
out("### Per-class (mean over seeds, %; ★ = classes named in TASK.md)\n")
out("| class | n | true share | ZS pred share | ZS recall | M1 pred share | M1 recall | M2 pred share | "
    "M2 recall | GT-caption recall | pseudo-caption recall |")
out("|---|---|---|---|---|---|---|---|---|---|---|")
for _, r in pc.iterrows():
    star = " ★" if r["class"] in FOCUS else ""
    cells = [f"{100 * r[c]:.1f}" + (f" ± {100 * r[c + '_std']:.1f}" if c.endswith(("supervised", "semi")) else "")
             for c in cols]
    out(f"| {r['class']}{star} | {int(r['n'])} | {100 * r['true_share']:.1f} | " + " | ".join(cells) + " |")
out("")

# ------------------------------------------------------------------ top confusion pairs from the held-out logits
if args.logits_root:
    import torch
    b = pd.read_csv("data/splits/B_labeled.csv")
    row = {p: i for i, p in enumerate(b.filepath)}
    y = b.label.values
    names = list(pc["class"])
    out("### Most frequent confusions (true → predicted; images per seed, mean over seeds; share of all errors)\n")
    for prefix, lbl in [("sup", "method 1"), ("semi", "method 2")]:
        counts, n_err = {}, []
        seeds = sorted(glob.glob(os.path.join(args.logits_root, f"{prefix}_s*")))
        for sd in seeds:
            run = sorted(glob.glob(os.path.join(sd, "*", "val_logits_fold0.pt")))[-1]
            pred = np.full(len(b), -1)
            for f in glob.glob(os.path.join(os.path.dirname(run), "val_logits_fold*.pt")):
                d = torch.load(f, weights_only=False)
                pred[[row[p] for p in d["filepath"]]] = d["cos"].argmax(1)
            assert (pred >= 0).all()
            wrong = pred != y
            n_err.append(wrong.sum())
            for t, p in zip(y[wrong], pred[wrong]):
                counts[(t, p)] = counts.get((t, p), 0) + 1
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:10]
        out(f"**{lbl}** ({np.mean(n_err):.1f} errors per seed)\n")
        out("| true → predicted | images | share of errors (%) |")
        out("|---|---|---|")
        for (t, p), c in top:
            out(f"| {names[t]} → {names[p]} | {c / len(seeds):.1f} | {100 * c / sum(n_err):.1f} |")
        out("")

text = "\n".join(lines)
print(text)
if args.out:
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w").write(text + "\n")

if args.plot:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(len(pc))
    fig, ax = plt.subplots(figsize=(14, 4.5))
    wbar = 0.2
    ax.bar(x - 1.5 * wbar, 100 * pc["true_share"], wbar, label="true", color="#444444")
    for i, (m, lbl, col) in enumerate([("zeroshot", "zero-shot", "#9db4d6"), ("supervised", "method 1", "#e3a857"),
                                       ("semi", "method 2", "#5b9b6b")]):
        ax.bar(x + (i - 0.5) * wbar, 100 * pc[f"pred_share_{m}"], wbar, label=lbl, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels([c + (" ★" if c in FOCUS else "") for c in pc["class"]], rotation=60, ha="right")
    ax.set_ylabel("share of B held-out images (%)")
    ax.set_title("Predicted vs true class distribution on B held-out folds (mean over seeds)")
    ax.legend(ncol=4)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.plot), exist_ok=True)
    fig.savefig(args.plot, dpi=120)
    print(f"saved {args.plot}")
