"""Paired comparison of two run groups that share seeds and folds.

Within one multirun, jobs are grouped by one override key (default
training.losses.pseudo_cap.enabled). For every (seed, fold) pair present in both groups
the metric difference is taken, then reported as mean difference, the win / tie / loss
count and a sign-test p-value. Fold-level pairing removes the fold-to-fold spread that
dominates the summary means on a 219-image eval fold.

    python scripts/paired_compare.py runs/multirun/2026-10-29_18-56-04
    python scripts/paired_compare.py <multirun> --key training.losses.anc.enabled --metrics mean_r1,mean_recall
    python scripts/paired_compare.py --a <run_dir_or_multirun> --b <run_dir_or_multirun>   # two separate groups
    python scripts/paired_compare.py <multirun> --summary          # one group: per (seed, fold) table, no pairing
    python scripts/paired_compare.py --a <multirun>/0_*,<multirun>/1_* --b <other_multirun>   # pick jobs by path
"""
import argparse
import glob
import json
import math
import os

import yaml


def _jobs(path):
    """metrics.json of every job under a multirun dir, or of a single run dir."""
    if os.path.isfile(os.path.join(path, "metrics.json")):
        return [path]
    return sorted(d for d in glob.glob(os.path.join(path, "*")) if os.path.isfile(os.path.join(d, "metrics.json")))


def _override(job, key):
    for line in open(os.path.join(job, ".hydra", "overrides.yaml")):
        line = line.strip().lstrip("- ")
        if line.startswith(key + "="):
            return line.split("=", 1)[1]
    cfg = yaml.safe_load(open(os.path.join(job, ".hydra", "config.yaml")))
    node = cfg
    for part in key.split("."):
        node = node[part]
    return str(node)


def _rows(job):
    m = json.load(open(os.path.join(job, "metrics.json")))
    seed = yaml.safe_load(open(os.path.join(job, ".hydra", "config.yaml")))["seed"]
    return {(seed, f["fold"]): f["metrics"] for f in m["folds"]}


def _sign_test(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def compare(group_a, group_b, metrics, label_a, label_b):
    rows_a, rows_b = {}, {}
    for job in group_a:
        rows_a.update(_rows(job))
    for job in group_b:
        rows_b.update(_rows(job))
    keys = sorted(set(rows_a) & set(rows_b))
    if not keys:
        raise SystemExit("no (seed, fold) pair is present in both groups")
    print(f"{len(keys)} paired (seed, fold) evaluations: {label_a} minus {label_b}")
    print(f"{'metric':14s} {'A mean':>8s} {'B mean':>8s} {'diff':>8s} {'win/tie/loss':>13s} {'sign p':>8s}")
    for metric in metrics:
        if metric not in rows_a[keys[0]]:
            continue
        diffs = [rows_a[k][metric] - rows_b[k][metric] for k in keys]
        wins = sum(d > 1e-9 for d in diffs)
        losses = sum(d < -1e-9 for d in diffs)
        ties = len(diffs) - wins - losses
        a_mean = sum(rows_a[k][metric] for k in keys) / len(keys)
        b_mean = sum(rows_b[k][metric] for k in keys) / len(keys)
        print(f"{metric:14s} {a_mean:8.4f} {b_mean:8.4f} {sum(diffs) / len(diffs):+8.4f} "
              f"{wins:>4d}/{ties}/{losses:<5d} {_sign_test(wins, losses):8.3f}")
    print("per pair (seed, fold):", "  ".join(f"{k}:{rows_a[k][metrics[0]] - rows_b[k][metrics[0]]:+.3f}" for k in keys))


def summarize(jobs, metrics):
    """One group: every (seed, fold) row, then the mean over folds per seed and over all rows."""
    rows = {}
    for job in jobs:
        rows.update(_rows(job))
    keys = sorted(rows)
    metrics = [m for m in metrics if m in rows[keys[0]]]
    print(f"{len(jobs)} jobs, {len(keys)} (seed, fold) evaluations")
    print(f"{'seed':>4s} {'fold':>4s} " + " ".join(f"{m:>11s}" for m in metrics))
    for seed, fold in keys:
        print(f"{seed:>4d} {fold:>4} " + " ".join(f"{rows[(seed, fold)][m]:11.4f}" for m in metrics))
    for seed in sorted({k[0] for k in keys}):
        sel = [rows[k] for k in keys if k[0] == seed]
        print(f"{seed:>4d} {'mean':>4s} " + " ".join(f"{sum(r[m] for r in sel) / len(sel):11.4f}" for m in metrics))
    print(f"{'all':>4s} {'mean':>4s} " + " ".join(f"{sum(rows[k][m] for k in keys) / len(keys):11.4f}" for m in metrics))
    print(f"{'all':>4s} {'std':>4s} " + " ".join(
        f"{(sum((rows[k][m] - sum(rows[q][m] for q in keys) / len(keys)) ** 2 for k in keys) / max(1, len(keys) - 1)) ** 0.5:11.4f}"
        for m in metrics))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("multirun", nargs="?", help="multirun dir whose jobs split into two groups by --key")
    ap.add_argument("--key", default="training.losses.pseudo_cap.enabled")
    ap.add_argument("--a", help="group A: run dir(s) or multirun dir(s), comma-separated (instead of splitting by --key)")
    ap.add_argument("--b", help="group B")
    ap.add_argument("--metrics", default="mean_r1,mean_recall,i2t_r1,t2i_r1,mean_r5,mean_r10")
    ap.add_argument("--summary", action="store_true", help="no pairing: tabulate every (seed, fold) of one group")
    args = ap.parse_args()
    metrics = args.metrics.split(",")
    if args.summary:
        paths = [args.multirun] if args.multirun else [p for arg in (args.a, args.b) if arg for p in arg.split(",")]
        if not paths:
            raise SystemExit("--summary needs a multirun / run dir")
        summarize([j for path in paths for j in _jobs(path)], metrics)
        return
    if args.a and args.b:
        jobs_a = [j for path in args.a.split(",") for j in _jobs(path)]
        jobs_b = [j for path in args.b.split(",") for j in _jobs(path)]
        compare(jobs_a, jobs_b, metrics, args.a, args.b)
        return
    if not args.multirun:
        raise SystemExit("give a multirun dir, or --a and --b")
    groups = {}
    for job in _jobs(args.multirun):
        groups.setdefault(_override(job, args.key).lower(), []).append(job)
    if len(groups) != 2:
        raise SystemExit(f"{args.key} takes {len(groups)} values in this multirun: {sorted(groups)}; need exactly 2. "
                         "A multirun that varies only the seed is one group: pair it with another one via "
                         "--a <dir> --b <dir>, or tabulate it alone with --summary.")
    on = groups.get("true") or groups[sorted(groups)[1]]
    off = groups.get("false") or groups[sorted(groups)[0]]
    compare(on, off, metrics, f"{args.key}=true", f"{args.key}=false")


if __name__ == "__main__":
    main()
