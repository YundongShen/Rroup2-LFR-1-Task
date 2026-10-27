"""Seed-aggregated table of A-only runs: groups runs/<exp_name>_s<seed>/ (latest run of each) and reports mean±std.

    python scripts/summarize_aonly.py [--root runs] [--prefix main_ sens_ gen_] [--out ../notes/results.md]
"""
import argparse
import glob
import json
import os
import re
import statistics
from collections import defaultdict

COLS = ["zs_acc", "zs_mean_per_class_acc", "i2t_R@1", "t2i_R@1", "mR", "rot_zs_acc_mean", "rot_zs_acc_min",
        "rot_mR_mean", "tta_zs_acc", "tta_mR"]

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="runs")
ap.add_argument("--prefix", nargs="+", default=["main_", "sens_", "gen_"])
ap.add_argument("--out", default=None)
args = ap.parse_args()

groups = defaultdict(list)
for exp in sorted(os.listdir(args.root)):
    if not exp.startswith(tuple(args.prefix)):
        continue
    runs = sorted(glob.glob(os.path.join(args.root, exp, "*", "metrics.json")))
    if runs:
        groups[re.sub(r"_s\d+$", "", exp)].append(json.load(open(runs[-1]))["summary"])

lines = ["| run | n | " + " | ".join(COLS) + " |", "|" + "---|" * (len(COLS) + 2)]
for name, sums in groups.items():
    cells = []
    for c in COLS:
        v = [100 * s[c]["mean"] for s in sums if c in s]
        cells.append("" if not v else f"{statistics.mean(v):.2f}" + (f"±{statistics.pstdev(v):.2f}" if len(v) > 1 else ""))
    lines.append(f"| {name} | {len(sums)} | " + " | ".join(cells) + " |")
print("\n".join(lines))
if args.out:
    open(args.out, "w").write("\n".join(lines) + "\n")
