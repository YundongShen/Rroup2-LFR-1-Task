"""Pseudo-label quality of the bridge on held-out B (docs/bridge_protocol.md §4; evaluation-only use of B labels).

For every fold of data=kfold5 the held-out B images are treated as unlabeled and go through the training
pseudo-label pipeline of src/methods/semi.py: teacher-adapted features -> top-3 captions from the fold's
B-train bank -> r (caption or image source) -> unbalanced Sinkhorn over the held-out images -> argmax.

    python scripts/pseudo_label_quality.py [--a_logs 'runs/p4/grid_s0/*k0*/*/fold*/train_log.json']
        --out runs/pl_quality

Teachers: frozen CLIP (identity adapter = first-epoch teacher) and the final EMA teacher of
runs/p1/semi_s{0,1,2} (latest run). Needs the feature cache and the CLIP text tower (run in the container).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, os.getcwd())
from src.data import b_folds, class_prompt_names, load_classes  # noqa: E402
from src.evaluate import encode_texts  # noqa: E402
from src.features import FeatureCache  # noqa: E402
from src.methods.semi import (Adapter, assign_pseudo_captions, bridge_r, build_bank,  # noqa: E402
                              caption_class_matrix, class_prior, unbalanced_sinkhorn)

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="runs/pl_quality")
ap.add_argument("--a_logs", default=None, help="glob of train_log.json of kappa=0 runs (A hard-label histograms)")
ap.add_argument("--eps_b", type=float, default=1.0)
ap.add_argument("--tau_text", type=float, default=0.01)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
dev = args.device

mcfg = OmegaConf.load("configs/method/semi.yaml")
dcfg = OmegaConf.load("configs/data/kfold5.yaml")
names = class_prompt_names(load_classes(dcfg.manifest_dir), {})
C = len(names)
folds = b_folds(dcfg.manifest_dir, dcfg.b_split)
cache = FeatureCache._index(torch.load("runs/cache/features_vit_b32_laion2b.pt", weights_only=True))

model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device=dev)
tok = open_clip.get_tokenizer("ViT-B-32")
model.eval()
V = encode_texts(model, tok, [mcfg.bridge.template.format(n) for n in names], dev).to(dev)


def teacher_of(seed, k):
    ad = Adapter(512, mcfg.adapter.hidden, mcfg.adapter.alpha).to(dev)
    if seed is not None:
        runs = [d for d in sorted(glob.glob(f"runs/p1/semi_s{seed}/*")) if os.path.exists(f"{d}/fold{k}/adapter.pt")]
        ad.load_state_dict(torch.load(f"{runs[-1]}/fold{k}/adapter.pt", weights_only=True)["teacher"])
    return ad.eval()


SETTINGS = [(src, prior, kappa) for src in ("caption", "image") for prior in ("uniform", "caption")
            for kappa in (0.0, 0.5, 1.0)]
TEACHERS = [("frozen CLIP", None), ("EMA teacher s0", 0), ("EMA teacher s1", 1), ("EMA teacher s2", 2)]
y_all = None
res = {}            # (teacher, setting) -> dict of arrays over all B
mass = {}           # teacher -> r column share (caption source)
for tname, seed in TEACHERS:
    preds = {s: [] for s in SETTINGS}
    ys, sel_flags, rmass = [], [], []
    for k, (b_tr, b_va) in enumerate(folds):
        ad = teacher_of(seed, k)
        with torch.no_grad():
            zV = F.normalize(ad(FeatureCache.images(cache, b_va.filepath).to(dev)), dim=-1)
            zB = F.normalize(ad(FeatureCache.images(cache, b_tr.filepath).to(dev)), dim=-1)
            bank, bank_id, _ = build_bank(b_tr)
            bank_id = bank_id.to(dev)
            tB = FeatureCache.texts(cache, bank).to(dev)
            cand, w, conf, keep = assign_pseudo_captions(zV, zB, tB, bank_id, mcfg.pcap.w_it, mcfg.pcap.topk,
                                                         mcfg.pcap.tau, mcfg.pcap.coverage)
            A_cap = caption_class_matrix(tB, V, args.tau_text)
            r_src = {"caption": bridge_r(cand, w, A_cap), "image": torch.softmax(zV @ V.T / args.tau_text, 1)}
            rmass.append(r_src["caption"].sum(0).cpu())
            for s in SETTINGS:
                src, prior, kappa = s
                pi = class_prior(prior, 0.0, A_cap, bank_id)
                Q = unbalanced_sinkhorn(r_src[src], args.eps_b, pi, kappa, mcfg.bridge.sinkhorn_iters)
                preds[s].append(Q.argmax(1).cpu().numpy())
        ys.append(b_va.label.values)
        flag = np.zeros(len(b_va), bool)
        flag[keep.cpu().numpy()] = True
        sel_flags.append(flag)
    y, sel = np.concatenate(ys), np.concatenate(sel_flags)
    y_all = y
    mass[tname] = (torch.stack(rmass).sum(0) / len(y)).numpy()
    for s in SETTINGS:
        res[(tname, s)] = {"pred": np.concatenate(preds[s]), "sel": sel}


def scores(pred, y):
    rec = np.array([(pred[y == c] == c).mean() for c in range(C)])
    return 100 * (pred == y).mean(), 100 * rec.mean(), rec


lines = []
out = lines.append
out("### Pseudo-label quality on held-out B (evaluation-only use of B labels; ε_b = 1, τ_text = 0.01)\n")
out("Pseudo-label accuracy / per-class mean accuracy (%) of ŷ = argmax Q against the B labels. "
    "EMA teacher = final teacher of method 2, mean ± std over seeds 0–2. "
    "'accepted' = coverage-0.5 subset by max R.\n")
out("| source | prior | κ | frozen CLIP: all | frozen CLIP: accepted | EMA teacher: all | EMA teacher: accepted |")
out("|---|---|---|---|---|---|---|")
summary = {}
for s in SETTINGS:
    cells = []
    for tset in (["frozen CLIP"], ["EMA teacher s0", "EMA teacher s1", "EMA teacher s2"]):
        for subset in ("all", "accepted"):
            acc, mpc = [], []
            for t in tset:
                r = res[(t, s)]
                m = np.ones_like(r["sel"]) if subset == "all" else r["sel"]
                a, p, _ = scores(r["pred"][m], y_all[m])
                acc.append(a)
                mpc.append(p)
            summary[f"{tset[0][:3]}|{subset}|{s}"] = {"acc": acc, "mpc": mpc}
            pm = lambda v: f"{np.mean(v):.1f}" + (f" ± {np.std(v):.1f}" if len(v) > 1 else "")
            cells.append(f"{pm(acc)} / {pm(mpc)}")
    out(f"| {s[0]} | {s[1] if s[2] > 0 else '(none)'} | {s[2]:g} | " + " | ".join(cells) + " |")
out("\nAt κ = 0 the prior has no effect (β = 1), so both prior rows coincide.\n")

# per-class pseudo-label accuracy, EMA teacher (mean over seeds), caption source, uniform prior
out("### Per-class pseudo-label accuracy (EMA teacher, mean over seeds 0–2, all held-out images, %)\n")
cols = [("caption", "uniform", 0.0), ("caption", "uniform", 1.0), ("image", "uniform", 0.0), ("image", "uniform", 1.0)]
out("| class | n | r share (caption, %) | " + " | ".join(f"{c[0]} κ={c[2]:g}" for c in cols) +
    " | hard labels caption κ=0 (mean count) |")
out("|---|---|---|" + "---|" * len(cols) + "---|")
ema = ["EMA teacher s0", "EMA teacher s1", "EMA teacher s2"]
recs = {c: np.mean([scores(res[(t, c)]["pred"], y_all)[2] for t in ema], 0) for c in cols}
counts0 = np.mean([np.bincount(res[(t, cols[0])]["pred"], minlength=C) for t in ema], 0)
share = np.mean([mass[t] for t in ema], 0)
for c in range(C):
    out(f"| {names[c]} | {(y_all == c).sum()} | {100 * share[c]:.2f} | " +
        " | ".join(f"{100 * recs[col][c]:.1f}" for col in cols) + f" | {counts0[c]:.1f} |")
zero_b = [names[c] for c in range(C) if counts0[c] == 0]
out(f"\n**Held-out B, caption source, κ = 0:** classes with zero hard pseudo-labels (all seeds): "
    f"{', '.join(zero_b) if zero_b else 'none'}; classes with r share < 0.5 %: "
    f"{', '.join(names[c] for c in range(C) if share[c] < 0.005) or 'none'}.\n")

if args.a_logs:
    files = sorted(glob.glob(args.a_logs, recursive=True))
    if files:
        first, last = [], []
        for f in files:
            ep = json.load(open(f))["epochs"]
            first.append(ep[0]["bridge_hist_all"])
            last.append(ep[-1]["bridge_hist_all"])
        first, last = np.array(first), np.array(last)
        out(f"**A, caption source, κ = 0** ({len(files)} training runs × folds; hard-label histogram over all "
            f"{int(first[0].sum())} A images, A labels never read): classes with zero hard labels in every run "
            f"at the first epoch: {', '.join(names[c] for c in range(C) if (first[:, c] == 0).all()) or 'none'}; "
            f"at the last epoch: {', '.join(names[c] for c in range(C) if (last[:, c] == 0).all()) or 'none'}. "
            f"Fewest labels (mean, last epoch): " +
            ", ".join(f"{names[c]} {last[:, c].mean():.0f}" for c in np.argsort(last.mean(0))[:6]) + ".\n")

text = "\n".join(lines)
print(text)
open(os.path.join(args.out, "report.md"), "w").write(text + "\n")
json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
