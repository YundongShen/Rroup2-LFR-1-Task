"""Collect all runs/**/metrics.json into one table.

    python scripts/collect_results.py [--root runs] [--split val|test] [--out results.csv]
"""
import argparse
import glob
import json
import os

import pandas as pd
from omegaconf import OmegaConf

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="runs")
ap.add_argument("--split", default=None, help="filter by eval split")
ap.add_argument("--out", default=None, help="optional .csv or .md output path")
args = ap.parse_args()

rows = []
for f in sorted(glob.glob(os.path.join(args.root, "**", "metrics.json"), recursive=True)):
    run_dir = os.path.dirname(f)
    m = json.load(open(f))
    if args.split and m["eval_split"] != args.split:
        continue
    cfg = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))
    row = {"run": os.path.relpath(run_dir, args.root), "method": cfg.method.name, "model": cfg.model.name,
           "prompts": cfg.prompts.name, "b_split": cfg.data.b_split.mode, "eval": m["eval_split"],
           "folds": m["n_folds"]}
    for k, v in m["summary"].items():
        row[k] = f"{100 * v['mean']:.2f}" + (f"±{100 * v['std']:.2f}" if m["n_folds"] > 1 else "")
    rows.append(row)

df = pd.DataFrame(rows)
print(df.to_markdown(index=False) if len(df) else "no runs found")
if args.out:
    df.to_markdown(args.out, index=False) if args.out.endswith(".md") else df.to_csv(args.out, index=False)
