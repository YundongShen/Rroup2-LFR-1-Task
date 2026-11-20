"""Pseudo-labels for the unlabeled Split A images.

The class distribution comes from the frozen CLIP or from the EMA teacher, never
from the student's own current step. `assign` turns a [n, C] probability matrix
into (labels, confidence, mask, per-class thresholds); `apply_thresholds` reuses
those thresholds on another set, which is how the B-val precision of the same
rule is measured.

Strategies
  global              adopt when max prob >= threshold
  per_class_adaptive  FreeMatch-style: threshold_c = threshold * mean_c / max_k mean_k,
                      where mean_c is the average probability the source gives class c
                      over all unlabeled images. Rare or hard classes get a lower bar.
  class_balanced      global threshold, then every class keeps at most `quota` of its
                      most confident images, quota = median of the per-class counts that
                      passed the threshold. Caps the easy classes that dominate RSICD.
"""
import torch

STRATEGIES = ("global", "per_class_adaptive", "class_balanced")


@torch.no_grad()
def assign(probs, strategy="per_class_adaptive", threshold=0.9, min_per_class=0):
    """Returns dict(labels [n], conf [n], mask [n] bool, thresholds [C])."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown pseudo-label strategy {strategy!r}, use one of {STRATEGIES}")
    probs = probs.float()
    conf, labels = probs.max(dim=1)
    num_classes = probs.shape[1]
    device = probs.device

    if strategy == "global":
        thresholds = torch.full((num_classes,), float(threshold), device=device)
    elif strategy == "per_class_adaptive":
        class_mean = probs.mean(dim=0)
        thresholds = float(threshold) * class_mean / class_mean.max().clamp_min(1e-12)
    else:  # class_balanced
        thresholds = torch.full((num_classes,), float(threshold), device=device)
        passed = conf >= threshold
        counts = torch.bincount(labels[passed], minlength=num_classes)
        nonzero = counts[counts > 0]
        quota = int(nonzero.float().median().item()) if len(nonzero) else 0
        mask = passed.clone()
        for c in range(num_classes):
            if quota > 0 and counts[c] > quota:
                rows = ((labels == c) & passed).nonzero(as_tuple=True)[0]
                keep = rows[conf[rows].topk(quota).indices]
                mask[rows] = False
                mask[keep] = True
                thresholds[c] = conf[keep].min()  # reported bar: the quota-th most confident image
    if strategy != "class_balanced":
        mask = conf >= thresholds[labels]
    if min_per_class > 0:
        for c in range(num_classes):
            rows = (labels == c).nonzero(as_tuple=True)[0]
            if len(rows) and int(mask[rows].sum()) < min_per_class:
                keep = rows[conf[rows].topk(min(min_per_class, len(rows))).indices]
                mask[keep] = True
    return {"labels": labels, "conf": conf, "mask": mask, "thresholds": thresholds}


@torch.no_grad()
def apply_thresholds(probs, thresholds):
    conf, labels = probs.float().max(dim=1)
    return {"labels": labels, "conf": conf, "mask": conf >= thresholds[labels], "thresholds": thresholds}


@torch.no_grad()
def summarize(assignment, true_labels=None, num_classes=None):
    """Counts (and, with true labels, precision) of the adopted pseudo-labels, overall and per class."""
    labels, mask = assignment["labels"], assignment["mask"]
    num_classes = int(num_classes or assignment["thresholds"].shape[0])
    counts = torch.bincount(labels[mask], minlength=num_classes)
    out = {
        "n_total": int(labels.shape[0]),
        "n_adopted": int(mask.sum()),
        "adopt_rate": float(mask.float().mean()),
        "per_class_count": counts.tolist(),
        "thresholds": [round(float(t), 4) for t in assignment["thresholds"]],
    }
    if true_labels is not None:
        true_labels = true_labels.to(labels.device)
        correct = (labels == true_labels) & mask
        per_class_correct = torch.bincount(labels[correct], minlength=num_classes)
        out["precision"] = float(correct.sum() / mask.sum().clamp_min(1)) if mask.any() else float("nan")
        out["per_class_correct"] = per_class_correct.tolist()
        out["per_class_precision"] = [
            float(c / n) if n > 0 else float("nan") for c, n in zip(per_class_correct.tolist(), counts.tolist())]
    return out
