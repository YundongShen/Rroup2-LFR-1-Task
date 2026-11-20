"""Classification and bidirectional retrieval on feature tensors.

Everything the semi trainer logs per epoch, and everything the final tables
report, comes through evaluate_features. The inputs are the adapted image
features, the caption features (5 per image, in row order), the 30 class
prototypes and the labels. All similarities are cosine similarities of
L2-normalized vectors; classification and retrieval share the same image
features.

Image-to-text: an image has 5 correct captions, a hit at K is any of them in
the top K. Text-to-image: a caption has 1 correct image.
"""
import numpy as np
import torch
import torch.nn.functional as F

from src.experiment.encode import recall_at, retrieval_ranks

RETRIEVAL_KS = (1, 5, 10)


def _captions_flat(txt):
    """[n, 5, d] or [5n, d] -> [5n, d]."""
    if txt.dim() == 3:
        return txt.flatten(0, 1)
    return txt


@torch.no_grad()
def classification_metrics(img, prototypes, labels, num_classes=None):
    img = F.normalize(img.float(), dim=-1)
    prototypes = F.normalize(prototypes.float(), dim=-1)
    pred = (img @ prototypes.T).argmax(1).cpu()
    labels = labels.cpu()
    num_classes = int(num_classes or prototypes.shape[0])
    per_class = torch.full((num_classes,), float("nan"))
    for k in labels.unique().tolist():
        per_class[k] = (pred[labels == k] == k).float().mean()
    present = ~torch.isnan(per_class)
    return {
        "cls_acc": (pred == labels).float().mean().item(),
        "mean_per_class_acc": per_class[present].mean().item(),
        "per_class_acc": per_class.tolist(),
        "pred": pred,
    }


@torch.no_grad()
def retrieval_metrics(img, txt, ks=RETRIEVAL_KS):
    img = F.normalize(img.float(), dim=-1).cpu()
    txt = F.normalize(_captions_flat(txt).float(), dim=-1).cpu()
    i2t, t2i = retrieval_ranks(img, txt)
    out = {}
    out.update({f"i2t_r{k}": v for k, v in zip(ks, recall_at(i2t, ks).values())})
    out.update({f"t2i_r{k}": v for k, v in zip(ks, recall_at(t2i, ks).values())})
    # label-free summaries, usable for model selection when class labels are off limits
    out["mean_r1"] = 0.5 * (out[f"i2t_r{ks[0]}"] + out[f"t2i_r{ks[0]}"])
    out["mean_recall"] = sum(out[f"{d}_r{k}"] for d in ("i2t", "t2i") for k in ks) / (2 * len(ks))
    return out


@torch.no_grad()
def evaluate_features(img, txt, prototypes, labels, ks=RETRIEVAL_KS):
    """dict with cls_acc, mean_per_class_acc, i2t_r{k}, t2i_r{k}, per_class_acc (list of 30)."""
    out = classification_metrics(img, prototypes, labels)
    out.pop("pred")
    if txt is not None:
        out.update(retrieval_metrics(img, txt, ks))
    return out


def scalar_metrics(metrics):
    """Drop list-valued entries so the dict can go into a csv row."""
    return {k: v for k, v in metrics.items() if not isinstance(v, (list, tuple))}


@torch.no_grad()
def bootstrap_std(img, txt, prototypes, labels, n_boot=1000, seed=0, ks=RETRIEVAL_KS):
    """Standard deviation of every scalar metric under image-level bootstrap resampling.

    This is the ± source recommended for the deterministic zero-shot row. Use the
    same n_boot and seed for every method row that reports it.
    """
    rng = np.random.default_rng(seed)
    n = img.shape[0]
    labels = labels.cpu()
    # Per-query outcomes are computed once against the full gallery; the bootstrap
    # resamples images (each with its 5 captions) and averages those outcomes.
    cls = classification_metrics(img, prototypes, labels)
    correct = (cls["pred"] == labels).float()
    num_classes = prototypes.shape[0]
    i2t = t2i = None
    if txt is not None:
        i2t, t2i = retrieval_ranks(
            F.normalize(img.float(), dim=-1).cpu(), F.normalize(_captions_flat(txt).float(), dim=-1).cpu())
        t2i = t2i.view(n, -1)
    samples = {}
    for _ in range(n_boot):
        index = torch.as_tensor(rng.integers(0, n, n))
        row = {"cls_acc": correct[index].mean().item()}
        per_class = [correct[index][labels[index] == k].mean().item()
                     for k in range(num_classes) if (labels[index] == k).any()]
        row["mean_per_class_acc"] = float(np.mean(per_class))
        if txt is not None:
            row.update({f"i2t_r{k}": (i2t[index] < k).float().mean().item() for k in ks})
            row.update({f"t2i_r{k}": (t2i[index] < k).float().mean().item() for k in ks})
        for key, value in row.items():
            samples.setdefault(key, []).append(value)
    return {key: float(np.std(values)) for key, values in samples.items()}
