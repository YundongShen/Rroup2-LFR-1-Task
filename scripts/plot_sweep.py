"""Sensitivity curve: a metric on the evaluation fold(s) against one hyper-parameter.

Runs are matched by task (and optionally protocol/training); the parameter value is
read from each run's resolved .hydra/config.yaml, and seeds at the same value give
the error band. Produce the runs with a Hydra sweep, for example

    python -m src.experiment.run -m training=semi training.losses.cls.weight=0.1,0.3,1,3 seed=0,1,2
    python scripts/plot_sweep.py --param training.losses.cls.weight --task classification --metric acc
    python scripts/plot_sweep.py --param training.losses.cls.pseudo.threshold --task i2t --metric R@1 --logx
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
import numpy as np
from omegaconf import OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--param", required=True, help="dotted config key, e.g. training.losses.cons.weight")
ap.add_argument("--root", default="runs")
ap.add_argument("--task", required=True)
ap.add_argument("--protocol", default=None)
ap.add_argument("--training", default=None, help="only runs with this training name")
ap.add_argument("--metric", default=None, help="metric key in metrics.json summary; default: first one")
ap.add_argument("--logx", action="store_true")
ap.add_argument("--out", default=None, help="png path; default figures/sweep_<param>_<metric>.png")
args = ap.parse_args()

points = {}
metric = args.metric
for path in sorted(glob.glob(os.path.join(args.root, "**", "metrics.json"), recursive=True)):
    run_dir = os.path.dirname(path)
    metrics = json.load(open(path))
    if metrics.get("task") != args.task:
        continue
    if args.protocol and metrics["protocol"] != args.protocol:
        continue
    if args.training and (metrics["training"] or "none") != args.training:
        continue
    cfg = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))
    value = OmegaConf.select(cfg, args.param, default=None)
    if value is None:
        continue
    metric = metric or next(iter(metrics["summary"]))
    if metric not in metrics["summary"]:
        continue
    points.setdefault(float(value), []).append(100 * metrics["summary"][metric]["mean"])

if not points:
    sys.exit(f"no runs with {args.param} and task={args.task} under {args.root}")

xs = sorted(points)
means = np.array([np.mean(points[x]) for x in xs])
stds = np.array([np.std(points[x]) for x in xs])
counts = [len(points[x]) for x in xs]
for x, m, s, n in zip(xs, means, stds, counts):
    print(f"{args.param}={x:g}  {metric}={m:.2f}±{s:.2f}  (n={n})")

figure, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(xs, means, marker="o")
ax.fill_between(xs, means - stds, means + stds, alpha=0.2)
if args.logx:
    ax.set_xscale("log")
ax.set_xlabel(args.param)
ax.set_ylabel(f"{metric} (%)  [{args.task}]")
ax.grid(True, alpha=0.3)
figure.tight_layout()
out = args.out or os.path.join("figures", f"sweep_{args.param.replace('.', '_')}_{metric.replace('@', '')}.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
figure.savefig(out, dpi=150)
print(f"saved {out}")
