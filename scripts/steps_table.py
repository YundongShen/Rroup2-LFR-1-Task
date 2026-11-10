"""Training-length table and curves from the phase-1 fold-0 runs (runs/steps/*).

    python scripts/steps_table.py [--root runs/steps] [--plot docs/figs/steps_curve.png]
"""
import argparse
import glob
import json
import os
import re

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="runs/steps")
ap.add_argument("--plot", default=None)
args = ap.parse_args()

rows = {}
for f in sorted(glob.glob(os.path.join(args.root, "*", "*", "metrics.json"))):
    name = os.path.basename(os.path.dirname(os.path.dirname(f)))      # semi_T2000 / sup_T2000
    meth, T = re.match(r"(semi|sup)_T(\d+)", name).groups()
    m = json.load(open(f))["folds"][0]["metrics"]
    log = json.load(open(os.path.join(os.path.dirname(f), "fold0", "train_log.json")))
    rows[(meth, int(T))] = {"mR1": 50 * (m["i2t_R@1"] + m["t2i_R@1"]), "acc": 100 * m["zs_acc"],
                            "curve": log["curve"]}

print("| T | method 2 mean R@1 | method 2 acc | method 1 mean R@1 | method 1 acc | method 2 teacher mean R@1 (end) |")
print("|---|---|---|---|---|---|")
for T in sorted({t for _, t in rows}):
    s, p = rows.get(("semi", T)), rows.get(("sup", T))
    cell = lambda r, k: f"{r[k]:.2f}" if r else "–"
    teach = f"{s['curve'][-1]['teacher_mR1']:.2f}" if s else "–"
    print(f"| {T} | {cell(s, 'mR1')} | {cell(s, 'acc')} | {cell(p, 'mR1')} | {cell(p, 'acc')} | {teach} |")

semi = {T: r["mR1"] for (meth, T), r in rows.items() if meth == "semi"}
if semi:
    best = max(semi.values())
    chosen = min(T for T, v in semi.items() if v >= best - 0.5)
    print(f"\nbest final method-2 mean R@1 = {best:.2f}; rule (within 0.5 of best, smallest T) -> T = {chosen}")

if args.plot:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, meth, title in [(axes[0], "semi", "method 2"), (axes[1], "sup", "method 1")]:
        for (mm, T), r in sorted(rows.items()):
            if mm != meth:
                continue
            c = r["curve"]
            ax.plot([x["step"] for x in c], [x["student_mR1"] for x in c], label=f"T={T}")
            if meth == "semi":
                ax.plot([x["step"] for x in c], [x["teacher_mR1"] for x in c], "--", lw=0.8,
                        color=ax.lines[-1].get_color())
        ax.set_xscale("symlog", linthresh=100)
        ax.set_title(f"{title} (fold 0, seed 0){' — dashed: EMA teacher' if meth == 'semi' else ''}")
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("B-val mean R@1 (%)")
    axes[1].legend()
    os.makedirs(os.path.dirname(args.plot), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.plot, dpi=120)
    print(f"saved {args.plot}")
