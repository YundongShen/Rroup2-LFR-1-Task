"""Phase-1 result table on B held-out folds (data=kfold5): zero-shot, method 1, method 2.

Each seed's value is the mean over the 5 folds; the table reports mean ± std over seeds
(population std, as in README's 3-seed sweeps). Zero-shot is deterministic (one run).

    python scripts/phase1_table.py [--root runs]
"""
import argparse
import glob
import json
import os
import statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="runs")
args = ap.parse_args()

KEYS = [("acc", lambda m: 100 * m["zs_acc"]),
        ("per-class acc", lambda m: 100 * m["zs_mean_per_class_acc"]),
        ("mean R@1", lambda m: 50 * (m["i2t_R@1"] + m["t2i_R@1"])),
        ("I→T R@1", lambda m: 100 * m["i2t_R@1"]), ("I→T R@5", lambda m: 100 * m["i2t_R@5"]),
        ("I→T R@10", lambda m: 100 * m["i2t_R@10"]),
        ("T→I R@1", lambda m: 100 * m["t2i_R@1"]), ("T→I R@5", lambda m: 100 * m["t2i_R@5"]),
        ("T→I R@10", lambda m: 100 * m["t2i_R@10"])]


def latest(pattern):
    fs = sorted(glob.glob(os.path.join(args.root, pattern, "*", "metrics.json")))
    return json.load(open(fs[-1])) if fs else None


def seed_value(run, fn):
    return st.mean(fn(f["metrics"]) for f in run["folds"])


rows, max_diff = [], 0.0
zs = latest("zeroshot_vit_b32_laion2b")
if zs:
    rows.append(("Zero-shot", [f"{seed_value(zs, fn):.2f}" for _, fn in KEYS], 1))
for label, prefix in [("Method 1 (supervised)", "sup"), ("Method 2 (semi)", "semi")]:
    runs = [r for r in (latest(f"p1/{prefix}_s{s}") for s in range(5)) if r]
    if not runs:
        continue
    for r in runs:
        assert r["n_folds"] == 5 and r["eval_split"] == "val"
        max_diff = max([max_diff] + [f.get("cached_max_abs_diff", 0.0) for f in r["folds"]])
    cells = []
    for _, fn in KEYS:
        v = [seed_value(r, fn) for r in runs]
        cells.append(f"{st.mean(v):.2f} ± {st.pstdev(v):.2f}")
    rows.append((label, cells, len(runs)))

print("| method | seeds | " + " | ".join(k for k, _ in KEYS) + " |")
print("|---|---|" + "---|" * len(KEYS))
for label, cells, n in rows:
    print(f"| {label} | {n} | " + " | ".join(cells) + " |")
print(f"\nmax |cached - standard| over all folds and seeds: {max_diff:.2e}")
