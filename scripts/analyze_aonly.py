"""Qualitative / failure analysis of an A-only run against zero-shot CLIP (ViT-B-32 laion2b, course prompt, all of B).

  - per-class zero-shot accuracy on B: baseline vs adapted, sorted by gain
  - prediction bias: share of B images predicted per class vs true share; zero-shot label histogram on A
  - most frequent confusions of the adapted model + example images
  - learning curves (zs_acc, mR, pseudo-label confidence) from history.json of the given runs

    python -m scripts.analyze_aonly --visual runs/main_rose_s0/<ts>/fold0/visual.pt \
        --curves main_rose_s0 main_vanilla_s0 p2_rose --out ../notes
"""
import argparse
import glob
import json
import os

import matplotlib
import open_clip
import torch
from omegaconf import OmegaConf
from PIL import Image

from src.data import b_folds, class_prompt_names, load_classes, load_unlabeled
from src.evaluate import class_embeddings, encode_images
from src.ops import encode_images_d4

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--visual", required=True)
ap.add_argument("--curves", nargs="*", default=[])
ap.add_argument("--out", default="../notes")
args = ap.parse_args()
os.makedirs(f"{args.out}/figs", exist_ok=True)
dev = "cuda"

model, _, tf = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device=dev)
tok = open_clip.get_tokenizer("ViT-B-32")
model.eval()
names = class_prompt_names(load_classes("data/splits"), {})
b = b_folds("data/splits", OmegaConf.create({"mode": "all"}))[0][1]
W = class_embeddings(model, tok, names, ["A satellite image of {}."], dev)
f0, y = encode_images(model, b, tf, dev)
fa0 = encode_images_d4(model, load_unlabeled("data/splits"), tf, dev, group=[0])[:, 0]
model.visual.load_state_dict(torch.load(args.visual, map_location=dev))
f1, _ = encode_images(model, b, tf, dev)
p0, p1 = (f0 @ W.T).argmax(1), (f1 @ W.T).argmax(1)

C = len(names)
rows = []
for c in range(C):
    m = y == c
    rows.append({"class": names[c], "n": int(m.sum()), "acc_zs": (p0[m] == c).float().mean().item(),
                 "acc_adapted": (p1[m] == c).float().mean().item(),
                 "pred_share_zs": (p0 == c).float().mean().item(), "pred_share_adapted": (p1 == c).float().mean().item(),
                 "true_share": m.float().mean().item(), "a_zs_label_share": (fa0 @ W.T).argmax(1).eq(c).float().mean().item()})
rows.sort(key=lambda r: r["acc_adapted"] - r["acc_zs"], reverse=True)
conf = torch.zeros(C, C, dtype=torch.long)
conf.index_put_((y, p1), torch.ones_like(y), accumulate=True)
off = conf.clone().fill_diagonal_(0)
top = [(int(i // C), int(i % C), int(off.view(-1)[i])) for i in off.view(-1).argsort(descending=True)[:6]]
json.dump({"per_class": rows, "top_confusions": [{"true": names[t], "pred": names[p], "count": n} for t, p, n in top]},
          open(f"{args.out}/analysis.json", "w"), indent=1)
print(f"{'class':22s} {'n':>3s} {'zs':>6s} {'adapt':>6s} {'Δ':>6s} {'pred%zs':>8s} {'pred%ad':>8s} {'true%':>6s} {'A-zs%':>6s}")
for r in rows:
    print(f"{r['class']:22s} {r['n']:3d} {100 * r['acc_zs']:6.1f} {100 * r['acc_adapted']:6.1f} "
          f"{100 * (r['acc_adapted'] - r['acc_zs']):+6.1f} {100 * r['pred_share_zs']:8.1f} {100 * r['pred_share_adapted']:8.1f} "
          f"{100 * r['true_share']:6.1f} {100 * r['a_zs_label_share']:6.1f}")
print("top confusions (true -> pred):", [(names[t], names[p], n) for t, p, n in top])

fig, axes = plt.subplots(len(top), 4, figsize=(8, 2.1 * len(top)))
for row, (t, p, n) in zip(axes, top):
    idx = ((y == t) & (p1 == p)).nonzero().flatten()[:4].tolist()
    for ax, i in zip(row, idx + [None] * (4 - len(idx))):
        ax.axis("off")
        if i is not None:
            ax.imshow(Image.open(b.filepath.iloc[i]).convert("RGB"))
    row[0].set_title(f"{names[t]} → {names[p]} ({n})", fontsize=8, loc="left")
fig.tight_layout()
fig.savefig(f"{args.out}/figs/confusions.png", dpi=120)

if args.curves:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for exp in args.curves:
        hist = json.load(open(sorted(glob.glob(f"runs/{exp}/*/fold0/history.json"))[-1]))
        ep = [h["epoch"] + 1 for h in hist]
        for ax, k in zip(axes, ["zs_acc", "mR", "pl_mean_conf"]):
            ax.plot(ep, [h[k] for h in hist], marker="o", label=exp)
            ax.set_title(k)
            ax.set_xlabel("epoch")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{args.out}/figs/curves.png", dpi=120)
