"""Aggregate one Hydra multirun over seeds: mean $\\pm$ std per metric.

Every job directory under the multirun root is read (metrics.json + .hydra).
Jobs that differ only in `seed` form one row; if the sweep also varied other
overrides, each distinct override set gets its own row.

    python scripts/seed_stats.py runs/multirun/2026-10-29_19-37-50
    python scripts/seed_stats.py runs/multirun/2026-10-29_19-37-50 --metrics acc i2t_r5 t2i_r5
    python scripts/seed_stats.py runs/multirun/2026-10-29_19-37-50 --metrics mean_r1 mean_r5 --out table.tex
    python scripts/seed_stats.py runs/multirun/2026-10-29_19-37-50 --pm "±" --out table.md

Metrics are reported in percent (x100) with 2 decimals by default; std is the
population std (ddof=0) across seeds, switch with --ddof 1.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import yaml

META_COLS = ["task", "protocol", "model", "prompts", "training", "overrides"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="multirun directory, e.g. runs/multirun/2026-10-29_19-37-50")
    ap.add_argument("--metrics", nargs="+", default=None,
                    help="metric keys from metrics.json summary (default: all found)")
    ap.add_argument("--out", default=None, help="output path; .csv / .md / .tex by extension")
    ap.add_argument("--pm", default="$\\pm$", help="separator between mean and std (default: $\\pm$)")
    ap.add_argument("--scale", type=float, default=100.0, help="multiply raw metric values (default 100 -> percent)")
    ap.add_argument("--decimals", type=int, default=2)
    ap.add_argument("--ddof", type=int, default=0, help="std degrees of freedom across seeds (0 population, 1 sample)")
    ap.add_argument("--per-seed", action="store_true", help="also print the raw per-seed table")
    return ap.parse_args()


def load_run(run_dir):
    metrics_path = os.path.join(run_dir, "metrics.json")
    overrides_path = os.path.join(run_dir, ".hydra", "overrides.yaml")
    config_path = os.path.join(run_dir, ".hydra", "config.yaml")
    if not (os.path.isfile(metrics_path) and os.path.isfile(config_path)):
        return None
    metrics = json.load(open(metrics_path))
    cfg = yaml.safe_load(open(config_path)) or {}
    overrides = [str(o) for o in (yaml.safe_load(open(overrides_path)) or [])] if os.path.isfile(overrides_path) else []
    seed = cfg.get("seed", 0)
    for item in overrides:
        if item.startswith("seed="):
            seed = int(item.split("=", 1)[1])
    row = {
        "run": os.path.basename(run_dir.rstrip("/")),
        "task": metrics.get("task"),
        "protocol": metrics.get("protocol"),
        "model": (cfg.get("model") or {}).get("name"),
        "prompts": (cfg.get("prompts") or {}).get("name"),
        "training": metrics.get("training") or "none",
        "overrides": " ".join(sorted(o for o in overrides if not o.startswith("seed="))),
        "seed": int(seed),
    }
    for key, value in metrics.get("summary", {}).items():
        row[key] = value["mean"] if isinstance(value, dict) else value
    return row


def main():
    args = parse_args()
    runs = [load_run(os.path.dirname(p))
            for p in sorted(glob.glob(os.path.join(args.root, "*", "metrics.json")))]
    runs = [r for r in runs if r is not None]
    if not runs:
        raise SystemExit(f"no finished runs (metrics.json) under {args.root}")
    df = pd.DataFrame(runs)

    available = [c for c in df.columns if c not in META_COLS + ["run", "seed"]]
    metrics = args.metrics or available
    missing = [m for m in metrics if m not in available]
    if missing:
        raise SystemExit(f"metrics {missing} not found; available: {available}")

    if args.per_seed:
        shown = df[["run", "seed"] + META_COLS + metrics].copy()
        shown[metrics] = shown[metrics] * args.scale
        print(shown.to_markdown(index=False, floatfmt=f".{args.decimals}f"))
        print()

    rows = []
    for group_values, part in df.groupby(META_COLS, sort=False, dropna=False):
        dup = part["seed"].duplicated()
        if dup.any():
            print(f"warning: duplicate seeds {sorted(part.loc[dup, 'seed'])} in group {dict(zip(META_COLS, group_values))}")
        out = dict(zip(META_COLS, group_values))
        out["n_seeds"] = len(part)
        out["seeds"] = ",".join(str(s) for s in sorted(part["seed"]))
        for key in metrics:
            vals = part[key].dropna().to_numpy(dtype=float) * args.scale
            if len(vals) == 0:
                out[f"{key}_mean"], out[f"{key}_std"], out[key] = np.nan, np.nan, ""
                continue
            mean = vals.mean()
            std = vals.std(ddof=args.ddof) if len(vals) > args.ddof else np.nan
            out[f"{key}_mean"], out[f"{key}_std"] = mean, std
            out[key] = (f"{mean:.{args.decimals}f} {args.pm} {std:.{args.decimals}f}"
                        if len(vals) > 1 else f"{mean:.{args.decimals}f}")
        rows.append(out)
    table = pd.DataFrame(rows)

    # drop metadata columns that are constant across rows so the table stays readable
    varying = [c for c in META_COLS if table[c].nunique(dropna=False) > 1]
    shown = table[varying + ["n_seeds"] + metrics]
    constant = {c: table[c].iloc[0] for c in META_COLS if c not in varying and str(table[c].iloc[0]) not in ("", "None", "nan")}
    if constant:
        print("common: " + ", ".join(f"{k}={v}" for k, v in constant.items()))
    print(shown.to_markdown(index=False))

    if args.out:
        ext = os.path.splitext(args.out)[1].lower()
        if ext == ".md":
            shown.to_markdown(args.out, index=False)
        elif ext == ".tex":
            cols = list(shown.columns)
            lines = ["\\begin{tabular}{" + "l" * len(cols) + "}", "\\toprule",
                     " & ".join(c.replace("_", "\\_") for c in cols) + " \\\\", "\\midrule"]
            for _, r in shown.iterrows():
                cells = [str(r[c]).replace("_", "\\_") if c not in metrics else str(r[c]) for c in cols]
                lines.append(" & ".join(cells) + " \\\\")
            lines += ["\\bottomrule", "\\end{tabular}"]
            with open(args.out, "w") as handle:
                handle.write("\n".join(lines) + "\n")
        else:
            table.to_csv(args.out, index=False)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
