"""Qualitative analysis of one finished run: zero-shot vs. adapted on its evaluation split.

Writes into <run>/qualitative/:
  confusion_adapted.csv / .png     confusion matrix of the adapted model
  per_class_acc.csv / .png         per-class accuracy, zero-shot vs. adapted, sorted by change
  retrieval_failures.md            images whose 5 captions all miss the top-5 (i2t) and captions
                                   whose image misses the top-5 (t2i), with what was retrieved instead
  pseudo_worst_classes.csv         classes with the lowest pseudo-label precision on B-val (last epoch)

    python scripts/qualitative.py --run runs/classification_held-out_vit_b32_laion2b_single_semi/<timestamp>
    python scripts/qualitative.py --run <run> --fold 2 --n-failures 15
"""
import argparse
import json
import os
import sys

import matplotlib
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data import CAPTION_COLS, class_prompt_names, load_classes  # noqa: E402
from src.experiment.encode import retrieval_ranks  # noqa: E402
from src.experiment.features import load_feature_cache  # noqa: E402
from src.experiment.protocol import iter_splits  # noqa: E402
from src.experiment.residual import AdaptedCLIP, load_adapter_state  # noqa: E402
from src.experiment.semi import encode_prototypes  # noqa: E402
from src.models import build_model, get_device  # noqa: E402

OmegaConf.register_new_resolver("repo_root", lambda: ROOT, replace=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory with .hydra/config.yaml and checkpoints/")
    ap.add_argument("--fold", type=int, default=None, help="fold to analyse; default: the run's first fold")
    ap.add_argument("--n-failures", type=int, default=10)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = OmegaConf.load(os.path.join(args.run, ".hydra", "config.yaml"))
    metrics = json.load(open(os.path.join(args.run, "metrics.json")))
    device = get_device(args.device or cfg.device)
    out_dir = os.path.join(args.run, "qualitative")
    os.makedirs(out_dir, exist_ok=True)

    folds = [r["fold"] for r in metrics["folds"]]
    fold = folds[0] if args.fold is None else args.fold
    if fold not in folds:
        sys.exit(f"fold {fold} is not in this run, available: {folds}")
    tag = "full" if fold is None else f"fold{fold}"
    split = next(s for s in iter_splits(cfg.manifest_dir, cfg.protocol) if s["fold"] == fold)
    eval_df = split["eval"]

    classes = load_classes(cfg.manifest_dir)
    names = class_prompt_names(classes, cfg.prompts.class_names)
    clip, _, _, tokenizer = build_model(cfg.model, device)
    clip.eval()
    cache = load_feature_cache(cfg.get("feature_cache", None), cfg.model)
    if cache is None:
        sys.exit("this script reads the frozen feature cache; run scripts/cache_features.py first")
    P0 = encode_prototypes(clip, tokenizer, classes, cfg.prompts, device)

    paths = eval_df["filepath"].astype(str).tolist()
    captions = [str(v) for row in eval_df[CAPTION_COLS].itertuples(index=False) for v in row]
    img0 = cache.images(paths).to(device)
    txt0 = cache.texts(captions).to(device)
    labels = torch.tensor(eval_df["label"].to_numpy())

    ckpt_path = os.path.join(args.run, "checkpoints", f"{tag}.pt")
    adapted_available = os.path.isfile(ckpt_path)
    if adapted_available:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        t = state["metadata"]["training"]
        model = AdaptedCLIP(clip, int(t["hidden_dim"]), t["alpha"], t["beta"], str(t["adapters"]),
                            learn_logit_scale=False).to(device)
        load_adapter_state(model, state)
        model.eval()
        with torch.no_grad():
            img1 = model.adapt_image_features(img0)
            txt1 = model.adapt_text_features(txt0)
            P1 = model.adapt_text_features(P0)
    else:
        print(f"no checkpoint at {ckpt_path}; reporting the zero-shot model only")
        img1, txt1, P1 = img0, txt0, P0

    C = len(names)
    pred0 = (img0 @ P0.T).argmax(1).cpu()
    pred1 = (img1 @ P1.T).argmax(1).cpu()

    # ---- confusion matrix of the adapted model
    confusion = np.zeros((C, C), dtype=int)
    for y, p in zip(labels.tolist(), pred1.tolist()):
        confusion[y, p] += 1
    pd.DataFrame(confusion, index=names, columns=names).to_csv(os.path.join(out_dir, "confusion_adapted.csv"))
    fig, ax = plt.subplots(figsize=(11, 10))
    norm = confusion / np.maximum(confusion.sum(1, keepdims=True), 1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(C)), ax.set_yticks(range(C))
    ax.set_xticklabels(names, rotation=90, fontsize=7), ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("predicted"), ax.set_ylabel("true")
    ax.set_title(f"{tag}: adapted model, row-normalised confusion")
    for i in range(C):
        for j in range(C):
            if confusion[i, j] and i != j:
                ax.text(j, i, confusion[i, j], ha="center", va="center", fontsize=5)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_adapted.png"), dpi=150)
    plt.close(fig)

    # ---- per-class accuracy, zero-shot vs adapted
    rows = []
    for c in range(C):
        m = labels == c
        if not m.any():
            continue
        a0, a1 = (pred0[m] == c).float().mean().item(), (pred1[m] == c).float().mean().item()
        rows.append({"class": names[c], "n": int(m.sum()), "zero_shot": 100 * a0, "adapted": 100 * a1,
                     "change": 100 * (a1 - a0)})
    per_class = pd.DataFrame(rows).sort_values("change")
    per_class.to_csv(os.path.join(out_dir, "per_class_acc.csv"), index=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(per_class))
    ax.bar(x - 0.2, per_class["zero_shot"], width=0.4, label="zero-shot")
    ax.bar(x + 0.2, per_class["adapted"], width=0.4, label="adapted")
    ax.set_xticks(x), ax.set_xticklabels(per_class["class"], rotation=90, fontsize=7)
    ax.set_ylabel("per-class acc (%)"), ax.legend(), ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "per_class_acc.png"), dpi=150)
    plt.close(fig)

    # ---- retrieval failures
    sim = (img1 @ txt1.T).cpu()
    i2t_rank, t2i_rank = retrieval_ranks(img1.cpu(), txt1.cpu())
    lines = [f"# Retrieval failures ({tag}, adapted model)", ""]
    lines.append(f"## Image -> text: images whose captions are all outside the top-5 ({int((i2t_rank >= 5).sum())} images)")
    for i in torch.nonzero(i2t_rank >= 5).flatten().tolist()[:args.n_failures]:
        top = sim[i].topk(5).indices.tolist()
        lines.append(f"\n**{paths[i]}** (class {names[labels[i]]}, first correct caption at rank {int(i2t_rank[i]) + 1})")
        lines.append(f"- ground truth: {captions[5 * i]}")
        for j in top:
            lines.append(f"- retrieved: {captions[j]}  _(image {paths[j // 5]})_")
    lines.append(f"\n## Text -> image: captions whose image is outside the top-5 ({int((t2i_rank >= 5).sum())} captions)")
    for j in torch.nonzero(t2i_rank >= 5).flatten().tolist()[:args.n_failures]:
        top = sim[:, j].topk(5).indices.tolist()
        lines.append(f"\n**{captions[j]}** (image {paths[j // 5]}, class {names[labels[j // 5]]}, correct image at rank {int(t2i_rank[j]) + 1})")
        for i in top:
            lines.append(f"- retrieved: {paths[i]} (class {names[labels[i]]})")
    with open(os.path.join(out_dir, "retrieval_failures.md"), "w") as handle:
        handle.write("\n".join(lines) + "\n")

    # ---- pseudo-label precision per class (B-val, last epoch)
    pseudo_path = os.path.join(args.run, "logs", f"{tag}_pseudo.json")
    if os.path.isfile(pseudo_path):
        log = json.load(open(pseudo_path))
        last = log[-1]
        if "val" in last:
            table = pd.DataFrame({
                "class": names,
                "val_adopted": last["val"]["per_class_count"],
                "val_correct": last["val"]["per_class_correct"],
                "val_precision": last["val"]["per_class_precision"],
                "unlabeled_adopted": last["unlabeled"]["per_class_count"],
                "threshold": last["unlabeled"]["thresholds"],
            }).sort_values("val_precision")
            table.to_csv(os.path.join(out_dir, "pseudo_worst_classes.csv"), index=False)
            print("lowest pseudo-label precision on B-val (last epoch):")
            print(table.head(5).to_string(index=False))

    print(f"zero-shot acc {100 * (pred0 == labels).float().mean():.2f}  adapted acc {100 * (pred1 == labels).float().mean():.2f}")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
