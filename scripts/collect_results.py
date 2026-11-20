"""Collect runs/**/metrics.json into one table.

One row per run by default. --group merges the runs that differ only in `seed`
(same task, protocol, model, prompts, training and other overrides) into one row
with mean ± std across seeds, which is the number the report quotes.

    python scripts/collect_results.py [--root runs] [--task classification] [--protocol kfold]
    python scripts/collect_results.py --group --task classification --out results.csv
    python scripts/collect_results.py --group --latex ablation.tex
    python scripts/collect_results.py \
        --root runs/multirun/2026-10-28_14-01-05 \
        --task classification --protocol kfold --training semi_stage3_cls --out ablation.tex
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import yaml
from omegaconf import OmegaConf

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="runs")
ap.add_argument("--task", default=None)
ap.add_argument("--protocol", default=None)
ap.add_argument("--training", default=None, help="exact training name, e.g. semi_stage1_con")
ap.add_argument("--group", action="store_true", help="aggregate runs that differ only in seed")
ap.add_argument("--out", default=None, help="optional .csv or .md output path")
ap.add_argument("--latex", default=None, help="optional .tex path for a tabular snippet")
args = ap.parse_args()

IGNORED_OVERRIDES = ("seed=", "device=", "num_workers=", "batch_size=", "feature_cache=", "aug_feature_cache=",
                     "manifest_dir=", "hydra.")


def _overrides(run_dir):
    path = os.path.join(run_dir, ".hydra", "overrides.yaml")
    if not os.path.isfile(path):
        return []
    items = yaml.safe_load(open(path)) or []
    keep = []
    for item in items:
        item = str(item)
        if item.startswith(IGNORED_OVERRIDES) or item.split("=")[0] in ("model", "prompts", "task", "protocol", "training"):
            continue
        keep.append(item)
    return sorted(keep)


rows = []
for path in sorted(glob.glob(os.path.join(args.root, "**", "metrics.json"), recursive=True)):
    run_dir = os.path.dirname(path)
    metrics = json.load(open(path))
    if "task" not in metrics:
        continue
    if args.task and metrics["task"] != args.task:
        continue
    if args.protocol and metrics["protocol"] != args.protocol:
        continue
    training = metrics["training"] or "none"
    if args.training and training != args.training:
        continue
    cfg = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))
    row = {
        "run": os.path.relpath(run_dir, args.root),
        "task": metrics["task"],
        "protocol": metrics["protocol"],
        "model": cfg.model.name,
        "prompts": cfg.prompts.name,
        "training": training,
        "overrides": " ".join(_overrides(run_dir)),
        "seed": int(cfg.get("seed", 0)),
        "folds": metrics["n_folds"],
    }
    for key, value in metrics["summary"].items():
        row[key] = 100 * value["mean"]
        row[f"{key}_foldstd"] = 100 * value["std"]
    rows.append(row)

df = pd.DataFrame(rows)
if not len(df):
    raise SystemExit("no runs found")
metric_keys = [c for c in df.columns if c not in
               ("run", "task", "protocol", "model", "prompts", "training", "overrides", "seed", "folds")
               and not c.endswith("_foldstd")]

if args.group:
    keys = ["task", "protocol", "model", "prompts", "training", "overrides"]
    grouped = []
    for group_values, part in df.groupby(keys, sort=False, dropna=False):
        out = dict(zip(keys, group_values))
        out["n_seeds"] = len(part)
        out["seeds"] = ",".join(str(s) for s in sorted(part["seed"]))
        for key in metric_keys:
            mean, std = part[key].mean(), part[key].std(ddof=0)
            out[f"{key}_mean"] = mean
            out[f"{key}_std"] = std
            if np.isnan(mean):
                out[key] = ""
            else:
                out[key] = f"{mean:.2f}±{std:.2f}" if len(part) > 1 else f"{mean:.2f}"
        grouped.append(out)
    table = pd.DataFrame(grouped)
    shown = table[keys + ["n_seeds"] + metric_keys]
else:
    table = df.copy()
    for key in metric_keys:
        table[key] = ["" if np.isnan(m) else (f"{m:.2f}±{s:.2f}" if f > 1 else f"{m:.2f}")
                      for m, s, f in zip(df[key], df[f"{key}_foldstd"], df["folds"])]
    shown = table[["run", "task", "protocol", "model", "prompts", "training", "overrides", "seed", "folds"] + metric_keys]

print(shown.to_markdown(index=False))
if args.out:
    if args.out.endswith(".md"):
        shown.to_markdown(args.out, index=False)
    else:
        table.to_csv(args.out, index=False)
    print(f"saved {args.out}")
if args.latex:
    cols = ["training", "overrides"] + metric_keys
    lines = ["\\begin{tabular}{l" + "l" * (len(cols) - 1) + "}", "\\toprule",
             " & ".join(c.replace("_", "\\_") for c in cols) + " \\\\", "\\midrule"]
    for _, r in shown.iterrows():
        cells = [str(r[c]).replace("_", "\\_").replace("±", " $\\pm$ ") for c in cols]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(args.latex, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"saved {args.latex}")
