"""Phase-4 selection and result tables (docs/bridge_protocol.md §5–6), all on B held-out folds.

    python scripts/bridge_results.py select1            # seed-0 ranking -> runs/p4/stage1.json (configs to re-check)
    python scripts/bridge_results.py final [--out runs/p4/results.md]   # needs torch + CLIP text tower for #6

Runs read: runs/p1/semi_s{s} (method 2), runs/p4/grid_s0/<cfg> (seed 0), runs/p4/recheck/<cfg>_s{s} (seeds 1, 2),
runs/p4/soft/<cfg>_s{s} (#5). The latest run directory containing val_logits_fold0.pt is used.
acc / per-class acc / R@k / mR1: mean over the 5 folds (as in metrics.json); TV and per-class accuracy:
pooled over all held-out B images. Seeds: mean ± std (population) over seeds 0–2.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=["select1", "final"])
ap.add_argument("--out", default="runs/p4/results.md")
args = ap.parse_args()
sys.path.insert(0, os.getcwd())

TOL = 0.5
NAME = re.compile(r"(caption|image)_l([\d.]+)_e([\d.]+)_k([\d.]+)(?:_(uniform|caption))?$")


def latest(pattern):
    ds = [d for d in sorted(glob.glob(pattern)) if os.path.exists(os.path.join(d, "val_logits_fold0.pt"))]
    return ds[-1] if ds else None


def fold_metrics(run):
    m = json.load(open(os.path.join(run, "metrics.json")))
    assert m["n_folds"] == 5 and m["eval_split"] == "val", run
    f = lambda key: float(np.mean([x["metrics"][key] for x in m["folds"]]) * 100)
    out = {k: f(k) for k in ["zs_acc", "zs_mean_per_class_acc", "i2t_R@1", "i2t_R@5", "i2t_R@10",
                             "t2i_R@1", "t2i_R@5", "t2i_R@10"]}
    out["mR1"] = (out["i2t_R@1"] + out["t2i_R@1"]) / 2
    return out


def parse(name):
    src, lam, eps, kap, prior = NAME.match(name).groups()
    return {"source": src, "lambda": float(lam), "eps_b": float(eps), "kappa": float(kap),
            "prior": prior or "(none)"}


def sort_key(row):
    return (-round(row["zs_acc"], 2), -row["mR1"], row["lambda"], -row["kappa"],
            row["prior"] != "uniform", -row["eps_b"])


GROUPS = {"#2 bridge kappa=0": lambda c: c["source"] == "caption" and c["kappa"] == 0,
          "#3 balanced bridge (main)": lambda c: c["source"] == "caption" and c["kappa"] > 0,
          "#4 balanced bridge, source=image": lambda c: c["source"] == "image" and c["kappa"] > 0}

m2 = {s: fold_metrics(latest(f"runs/p1/semi_s{s}/*")) for s in range(3)}
m2_mr1_s0 = m2[0]["mR1"]
m2_mr1_3 = np.mean([m2[s]["mR1"] for s in range(3)])

grid = []
for d in sorted(glob.glob("runs/p4/grid_s0/*")):
    run = latest(os.path.join(d, "*"))
    if run:
        grid.append({"name": os.path.basename(d), **parse(os.path.basename(d)), **fold_metrics(run)})
grid = pd.DataFrame(grid)
grid["eligible"] = grid["mR1"] >= m2_mr1_s0 - TOL


def fmt_grid(df):
    lines = ["| config | source | λ | ε_b | κ | prior | acc | per-class acc | mR1 | eligible |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {r['name']} | {r['source']} | {r['lambda']:g} | {r['eps_b']:g} | {r['kappa']:g} | {r['prior']} | "
                     f"{r['zs_acc']:.2f} | {r['zs_mean_per_class_acc']:.2f} | {r['mR1']:.2f} | {'yes' if r['eligible'] else 'no'} |")
    return "\n".join(lines)


def ranked(group):
    sub = grid[grid.apply(lambda r: GROUPS[group](r), axis=1)]
    rows = sorted(sub.to_dict("records"), key=sort_key)
    return [r for r in rows if r["eligible"]], rows


if args.stage == "select1":
    print(f"method 2 seed 0: acc {m2[0]['zs_acc']:.2f}, mR1 {m2_mr1_s0:.2f}; eligible if mR1 >= {m2_mr1_s0 - TOL:.2f}")
    print(f"grid runs found: {len(grid)}")
    stage1 = {}
    for g in GROUPS:
        elig, rows = ranked(g)
        print(f"\n## {g}: {len(rows)} configs, {len(elig)} eligible")
        print(fmt_grid(pd.DataFrame(rows)))
        stage1[g] = [r["name"] for r in elig[:3]]
        print("re-check (seeds 1, 2):", stage1[g])
    json.dump(stage1, open("runs/p4/stage1.json", "w"), indent=1)
    raise SystemExit

# ------------------------------------------------------------------------------------------------ final
import torch  # noqa: E402

from src.methods.semi import class_prior, caption_class_matrix, unbalanced_sinkhorn  # noqa: E402

b = pd.read_csv("data/splits/B_labeled.csv")
row_of = {p: i for i, p in enumerate(b.filepath)}
y = b.label.values
C = int(y.max()) + 1
true_share = np.bincount(y, minlength=C) / len(y)
names = None


def pooled(run, transform=None):
    """Pooled held-out predictions over the 5 folds -> (pred, per-fold acc list)."""
    pred = np.full(len(b), -1)
    accs = []
    for k in range(5):
        d = torch.load(os.path.join(run, f"val_logits_fold{k}.pt"), weights_only=False)
        p = transform(d) if transform else d["cos"].argmax(1)
        idx = np.array([row_of[q] for q in d["filepath"]])
        pred[idx] = p
        accs.append((p == y[idx]).mean())
    assert (pred >= 0).all()
    return pred, accs


def label_metrics(pred, accs):
    rec = np.array([(pred[y == c] == c).mean() for c in range(C)])
    share = np.bincount(pred, minlength=C) / len(pred)
    return {"acc_pooled": 100 * (pred == y).mean(), "acc_foldmean": 100 * np.mean(accs),
            "tv": 100 * 0.5 * np.abs(share - true_share).sum(), "recall": rec, "share": share}


def seeds_of(name, kind):
    """Run dirs for seeds 0-2 of a grid config ('recheck') or a soft variant ('soft')."""
    if kind == "recheck":
        return {0: latest(f"runs/p4/grid_s0/{name}/*"),
                **{s: latest(f"runs/p4/recheck/{name}_s{s}/*") for s in (1, 2)}}
    return {s: latest(f"runs/p4/soft/{name}_s{s}/*") for s in range(3)}


# ---- stage 2: re-check top 3 of each group with seeds 0-2
stage1 = json.load(open("runs/p4/stage1.json"))
lines = []
out = lines.append
out(f"Method 2 reference: seed 0 mR1 {m2_mr1_s0:.2f}; seeds 0–2 mR1 {m2_mr1_3:.2f} "
    f"(eligible ≥ {m2_mr1_3 - TOL:.2f}).\n")
chosen = {}
for g, cfgs in stage1.items():
    out(f"#### {g}: re-check of the coarse top 3 (seeds 0–2)\n")
    out("| config | acc | per-class acc | mR1 | eligible |")
    out("|---|---|---|---|---|")
    rows = []
    for name in cfgs:
        runs = seeds_of(name, "recheck")
        if any(v is None for v in runs.values()):
            out(f"| {name} | missing seeds {[s for s, v in runs.items() if v is None]} | | | |")
            continue
        ms = [fold_metrics(runs[s]) for s in range(3)]
        r = {"name": name, **parse(name), **{k: np.mean([m[k] for m in ms]) for k in ms[0]},
             "acc_std": np.std([m["zs_acc"] for m in ms]), "mR1_std": np.std([m["mR1"] for m in ms])}
        r["eligible"] = r["mR1"] >= m2_mr1_3 - TOL
        rows.append(r)
        out(f"| {name} | {r['zs_acc']:.2f} ± {r['acc_std']:.2f} | {r['zs_mean_per_class_acc']:.2f} | "
            f"{r['mR1']:.2f} ± {r['mR1_std']:.2f} | {'yes' if r['eligible'] else 'no'} |")
    elig = sorted([r for r in rows if r["eligible"]], key=sort_key)
    chosen[g] = elig[0]["name"] if elig else None
    out(f"\nSelected: **{chosen[g] or 'none eligible (protocol §5.4 applies)'}**\n")

# ---- #6 test-time balancing on method-2 predictions (transductive)
from src.data import b_folds, class_prompt_names, load_classes  # noqa: E402
from src.evaluate import encode_texts  # noqa: E402
from src.features import FeatureCache  # noqa: E402
from src.methods.semi import build_bank  # noqa: E402
import open_clip  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

mcfg = OmegaConf.load("configs/method/semi.yaml")
dcfg = OmegaConf.load("configs/data/kfold5.yaml")
names = class_prompt_names(load_classes(dcfg.manifest_dir), {})
cache = FeatureCache._index(torch.load("runs/cache/features_vit_b32_laion2b.pt", weights_only=True))
model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
tok = open_clip.get_tokenizer("ViT-B-32")
V = encode_texts(model.eval(), tok, [mcfg.bridge.template.format(n) for n in names], "cpu")
priors = {}
for k, (b_tr, _) in enumerate(b_folds(dcfg.manifest_dir, dcfg.b_split)):
    bank, bank_id, _ = build_bank(b_tr)
    A_cap = caption_class_matrix(FeatureCache.texts(cache, bank), V, mcfg.bridge.tau_text)
    priors[k] = {"uniform": class_prior("uniform", 0, A_cap, bank_id), "caption": class_prior("caption", 0, A_cap, bank_id)}


def tta(kappa, eps, prior):
    def f(d):
        r = torch.softmax(torch.tensor(d["scale"] * d["cos"], dtype=torch.float64), 1)
        return unbalanced_sinkhorn(r, eps, priors[d["fold"]][prior].double(), kappa).argmax(1).numpy()
    return f


TTA = [(k, e, p) for k in (0.5, 1.0) for e in (0.5, 1.0) for p in ("uniform", "caption")]
m2_runs = {s: latest(f"runs/p1/semi_s{s}/*") for s in range(3)}
coarse = sorted([{"cfg": c, "zs_acc": 100 * np.mean(pooled(m2_runs[0], tta(*c))[1]), "mR1": m2_mr1_s0,
                  "lambda": 0, "kappa": c[0], "prior": c[2], "eps_b": c[1]} for c in TTA], key=sort_key)
out("#### #6 test-time balancing (transductive): seed-0 screen of 8 settings, top 3 with seeds 0–2\n")
out("| κ | ε_b | prior | acc seed 0 | acc seeds 0–2 |")
out("|---|---|---|---|---|")
best6 = None
for i, r in enumerate(coarse):
    s3 = ""
    if i < 3:
        a = [100 * np.mean(pooled(m2_runs[s], tta(*r["cfg"]))[1]) for s in range(3)]
        s3 = f"{np.mean(a):.2f} ± {np.std(a):.2f}"
        if best6 is None or round(np.mean(a), 2) > round(best6[1], 2):
            best6 = (r["cfg"], np.mean(a))
    out(f"| {r['cfg'][0]:g} | {r['cfg'][1]:g} | {r['cfg'][2]} | {r['zs_acc']:.2f} | {s3} |")
out(f"\nSelected #6: κ={best6[0][0]:g}, ε_b={best6[0][1]:g}, prior={best6[0][2]}\n")

# ---- main table #1-#6
EXPS = [("#1 method 2", {s: m2_runs[s] for s in range(3)}, None)]
if chosen.get("#2 bridge kappa=0"):
    EXPS.append(("#2 bridge κ=0", seeds_of(chosen["#2 bridge kappa=0"], "recheck"), None))
main = chosen.get("#3 balanced bridge (main)")
if main:
    EXPS.append(("#3 balanced bridge (main)", seeds_of(main, "recheck"), None))
if chosen.get("#4 balanced bridge, source=image"):
    EXPS.append(("#4 balanced, source=image", seeds_of(chosen["#4 balanced bridge, source=image"], "recheck"), None))
if main and all(seeds_of(main, "soft").values()):
    EXPS.append(("#5 main, target=soft", seeds_of(main, "soft"), None))
EXPS.append(("#6 method 2 + test-time balancing † (transductive)", {s: m2_runs[s] for s in range(3)}, tta(*best6[0])))

out("### Main comparison on B held-out folds (seeds 0–2, mean ± std; %)\n")
out(f"Configurations: #2 {chosen.get('#2 bridge kappa=0')}, #3 {main}, "
    f"#4 {chosen.get('#4 balanced bridge, source=image')}, #5 = #3 with target=soft, "
    f"#6 κ={best6[0][0]:g} ε_b={best6[0][1]:g} prior={best6[0][2]}. † uses the unlabeled evaluation images.\n")
out("| experiment | acc | per-class acc | TV(pred, true) | I→T R@1 | I→T R@5 | I→T R@10 | T→I R@1 | T→I R@5 | T→I R@10 | mR1 |")
out("|---|---|---|---|---|---|---|---|---|---|---|")
percls = {}
for label, runs, tf in EXPS:
    vals = {k: [] for k in ["acc", "mpc", "tv", "i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10", "mR1"]}
    recs = []
    for s in range(3):
        fm = fold_metrics(runs[s])
        pred, accs = pooled(runs[s], tf)
        lm = label_metrics(pred, accs)
        vals["acc"].append(lm["acc_foldmean"])
        vals["mpc"].append(100 * lm["recall"].mean())
        vals["tv"].append(lm["tv"])
        for k in ["i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10", "mR1"]:
            vals[k].append(fm[k])
        recs.append(lm["recall"])
    percls[label] = np.mean(recs, 0)
    out(f"| {label} | " + " | ".join(f"{np.mean(v):.2f} ± {np.std(v):.2f}" for v in vals.values()) + " |")
out("\nper-class acc here is pooled over folds (mean of per-class recalls over all B), so it can differ by ≈0.1 "
    "from the fold-mean value used in the grid tables.\n")

out("### Per-class accuracy (seeds 0–2 mean, %)\n")
labels = list(percls)
out("| class | n | " + " | ".join(l.split(" ")[0] for l in labels) + " |")
out("|---|---|" + "---|" * len(labels))
for c in range(C):
    out(f"| {names[c]} | {(y == c).sum()} | " + " | ".join(f"{100 * percls[l][c]:.1f}" for l in labels) + " |")

text = "\n".join(lines)
print(text)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
open(args.out, "w").write(text + "\n")
json.dump({"chosen": chosen, "tta": list(best6[0])}, open("runs/p4/chosen.json", "w"), indent=1)
