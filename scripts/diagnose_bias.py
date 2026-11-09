"""Why does the pseudo-caption (semi-supervised) adapter gain so little classification accuracy?

Three hypotheses, measured on the held-out B folds. Every B image is predicted once, by the model of the
fold it was held out from. B class labels are used here for evaluation only, never for training or selection.

  H1 class bias     the predicted class distribution is skewed (classes never / rarely predicted, others
                    over-predicted); re-balancing the predictions alone recovers much of the error.
  H2 conversion     the captions carry class evidence that the image embedding does not use: caption-derived
                    class distributions are right where the image prediction is wrong.
  H3 near-synonyms  errors go to classes whose prompts are almost identical in CLIP text space.

Step 1, during training. After fitting each fold, store the held-out scores of every method to compare:

    from scripts.diagnose_bias import class_text_embeddings, save_val_logits
    W = class_text_embeddings(model, tokenizer, names, templates, device)      # frozen text tower
    feats = <normalized image features of b_val exactly as used at evaluation (adapter included)>
    save_val_logits(f"{out_dir}/val_logits_fold{k}.pt", b_val.filepath.tolist(), feats, W, fold=k,
                    scale=model.logit_scale.exp().item())

Step 2, from the repository root:

    python -m scripts.diagnose_bias \
        --run supervised='runs/sup*/**/val_logits_fold*.pt' \
        --run semi='runs/semi*/**/val_logits_fold*.pt' --out notes/diagnosis

Frozen zero-shot CLIP is always added as a reference. Use one seed per method for the main table and repeat
over seeds before drawing conclusions (about 35 B images per class, so per-class numbers are noisy).
"""
import argparse
import glob
import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

TEMPLATES = ["A satellite image of {}."]


# ----------------------------------------------------------------------------- helpers used during training
def class_text_embeddings(model, tokenizer, names, templates, dev):
    import torch
    import torch.nn.functional as F
    from src.evaluate import encode_texts
    with torch.no_grad():
        W = [encode_texts(model, tokenizer, [t.format(n) for t in templates], dev).mean(0) for n in names]
    return F.normalize(torch.stack(W), dim=-1)


def save_val_logits(path, filepaths, img_feats, W, fold, scale):
    """img_feats [N,d] and W [C,d] L2-normalized; stores cosine scores of the held-out images."""
    import torch
    torch.save({"filepath": list(filepaths), "cos": (img_feats.float().cpu() @ W.float().cpu().T).numpy(),
                "fold": int(fold), "scale": float(scale)}, path)


# ----------------------------------------------------------------------------- pure analysis (numpy only)
def softmax(x, axis=-1):
    x = x - x.max(axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis, keepdims=True)


def log_softmax(x, axis=-1):
    x = x - x.max(axis, keepdims=True)
    return x - np.log(np.exp(x).sum(axis, keepdims=True))


def sinkhorn(S, eps, prior=None, n_iter=50):
    """Balanced assignment of N rows to C classes; class masses follow `prior` (uniform if None)."""
    N, C = S.shape
    prior = np.full(C, 1.0 / C) if prior is None else np.asarray(prior, float) / np.sum(prior)
    Q = np.exp((S - S.max()) / eps).T.astype(np.float64)       # [C, N]
    Q /= Q.sum()
    for _ in range(n_iter):
        Q *= (prior / np.clip(Q.sum(1), 1e-300, None))[:, None]
        Q *= ((1.0 / N) / np.clip(Q.sum(0), 1e-300, None))[None, :]
    return (Q * N).T


def per_class_recall(pred, y, C):
    return np.array([(pred[y == c] == c).mean() if (y == c).any() else np.nan for c in range(C)])


def bias_report(pred, y, C):
    true = np.bincount(y, minlength=C) / len(y)
    share = np.bincount(pred, minlength=C) / len(pred)
    present = true > 0
    over = present & (share > 2 * true)
    wrong = pred != y
    recall = per_class_recall(pred, y, C)
    rep = {"acc": float((~wrong).mean()),
           "mean_per_class_acc": float(np.nanmean(recall)),
           "tv_pred_vs_true": float(0.5 * np.abs(share - true).sum()),
           "n_never_predicted": int((present & (share == 0)).sum()),
           "n_under_predicted(<0.5x)": int((present & (share < 0.5 * true)).sum()),
           "n_over_predicted(>2x)": int(over.sum()),
           "err_share_into_over_predicted": float(over[pred[wrong]].mean()) if wrong.any() else 0.0}
    return rep, share, recall


def spearman(a, b):
    ra, rb = pd.Series(a).rank().values, pd.Series(b).rank().values
    return float(np.corrcoef(ra, rb)[0, 1])


def synonym_report(pred, y, K, k=3):
    """K [C,C] cosine similarity of the class prompts in text space."""
    C = K.shape[0]
    present = np.bincount(y, minlength=C) > 0
    Ko = K.copy()
    np.fill_diagonal(Ko, -np.inf)
    Ko[:, ~present] = -np.inf
    near = np.argsort(-Ko, 1)[:, :k]
    wrong = pred != y
    into_near = np.array([p in near[t] for t, p in zip(y[wrong], pred[wrong])])
    conf = np.zeros((C, C))
    np.add.at(conf, (y, pred), 1)
    rate = conf / np.clip(conf.sum(1, keepdims=True), 1, None)
    sym = (rate + rate.T) / 2
    iu = np.triu_indices(C, 1)
    keep = present[iu[0]] & present[iu[1]]
    return {f"err_share_into_{k}_text_nearest": float(into_near.mean()) if wrong.any() else 0.0,
            "chance_level": k / max(int(present.sum()) - 1, 1),
            "spearman(text_sim, confusion)": spearman(K[iu][keep], sym[iu][keep])}


def zscore(x):
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-8)


def pseudo_caption_r(z_val, z_tr, cap_ids_tr, T_cap, A_cap, lam=0.2, k=3, tau=0.1):
    """Method-2 style caption assignment for held-out images (frozen features stand in for the EMA teacher).

    z_val [Nv,d], z_tr [Nt,d] image features; cap_ids_tr [Nt,5] ids of each training image's captions
    into T_cap [M,d] / A_cap [M,C] (caption features / caption->class distributions). Returns r [Nv,C].
    """
    bank = np.unique(cap_ids_tr)
    S_it = z_val @ T_cap[bank].T
    S_img = z_val @ z_tr.T
    S_ii = np.full((len(z_val), len(bank)), -np.inf)
    pos = {c: i for i, c in enumerate(bank)}
    for img, ids in enumerate(cap_ids_tr):
        for c in set(ids.tolist()):
            j = pos[c]
            S_ii[:, j] = np.maximum(S_ii[:, j], S_img[:, img])
    R = lam * zscore(S_it) + (1 - lam) * zscore(S_ii)
    top = np.argpartition(-R, k, axis=1)[:, :k]
    w = softmax(np.take_along_axis(R, top, 1) / tau, 1)
    return np.einsum("nk,nkc->nc", w, A_cap[bank][top])


def caption_report(pred, y, logp, r, beta=1.0):
    rc = r.argmax(1)
    wrong = pred != y
    return {"acc_caption_only": float((rc == y).mean()),
            "caption_right_when_image_wrong": float((rc[wrong] == y[wrong]).mean()) if wrong.any() else 0.0,
            "acc_image_x_caption": float(((logp + beta * np.log(r + 1e-6)).argmax(1) == y).mean())}


# ----------------------------------------------------------------------------- main
def main():
    import open_clip
    import torch
    from src.data import CAPTION_COLS, b_folds, class_prompt_names, load_classes
    from src.evaluate import encode_images, encode_texts

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[], help="name=glob of val_logits_fold*.pt files")
    ap.add_argument("--manifest_dir", default="data/splits")
    ap.add_argument("--arch", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--eps", type=float, default=0.02, help="Sinkhorn temperature on cosine scores")
    ap.add_argument("--tau_text", type=float, default=0.01, help="caption->class softmax temperature")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="notes/diagnosis")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    b = pd.read_csv(os.path.join(args.manifest_dir, "B_labeled.csv"))
    y = b.label.values.astype(int)
    names = class_prompt_names(load_classes(args.manifest_dir), {})
    C = len(names)
    row_of = {p: i for i, p in enumerate(b.filepath)}

    # held-out fold of every B image (same split as data=kfold5)
    split = SimpleNamespace(mode="kfold", k=5, stratify=True, seed=0)
    fold = np.full(len(b), -1)
    for f, (_, va) in enumerate(b_folds(args.manifest_dir, split)):
        fold[[row_of[p] for p in va.filepath]] = f

    model, _, tf = open_clip.create_model_and_transforms(args.arch, pretrained=args.pretrained, device=args.device)
    tok = open_clip.get_tokenizer(args.arch)
    model.eval()
    W = class_text_embeddings(model, tok, names, TEMPLATES, args.device).cpu()
    z0 = encode_images(model, b, tf, args.device)[0].numpy()
    caps = b[CAPTION_COLS].values
    uniq, cap_ids = np.unique(caps.reshape(-1), return_inverse=True)
    cap_ids = cap_ids.reshape(len(b), 5)
    T_cap = encode_texts(model, tok, uniq.tolist(), args.device).numpy()
    Wn = W.numpy()
    A_cap = softmax(T_cap @ Wn.T / args.tau_text, 1)            # caption -> class distribution
    K = Wn @ Wn.T

    # caption evidence: own human captions (r_gt) and method-2 style pseudo-captions from the other folds
    r_gt = A_cap[cap_ids].mean(1)
    r_ps = np.zeros((len(b), C))
    for f in range(5):
        va, tr = np.where(fold == f)[0], np.where(fold != f)[0]
        r_ps[va] = pseudo_caption_r(z0[va], z0[tr], cap_ids[tr], T_cap, A_cap)

    true_share = np.bincount(y, minlength=C) / len(y)
    prior_rep = {"tv_uniform_vs_true": float(0.5 * np.abs(1.0 / C - true_share).sum()),
                 "tv_caption_prior_vs_true": float(0.5 * np.abs(r_gt.mean(0) - true_share).sum())}

    methods = {"zeroshot": (z0 @ Wn.T, model.logit_scale.exp().item())}
    for spec in args.run:
        name, pattern = spec.split("=", 1)
        files = sorted(glob.glob(pattern, recursive=True))
        assert files, f"no files for {spec}"
        cos, scale, seen = np.full((len(b), C), np.nan), None, np.zeros(len(b), bool)
        for fp in files:
            d = torch.load(fp, weights_only=False)
            idx = np.array([row_of[p] for p in d["filepath"]])
            if not (fold[idx] == d["fold"]).all():
                print(f"[warn] {fp}: stored fold differs from data=kfold5 split")
            cos[idx], seen[idx], scale = d["cos"], True, d["scale"]
        assert seen.all(), f"{name}: {int((~seen).sum())} B images have no held-out prediction"
        methods[name] = (cos, scale)

    summary, per_class = {"class_prior": prior_rep}, {"class": names, "n": np.bincount(y, minlength=C),
                                                      "true_share": true_share}
    for name, (cos, scale) in methods.items():
        pred = cos.argmax(1)
        logp = log_softmax(scale * cos, 1)
        rep, share, recall = bias_report(pred, y, C)
        for tag, prior in [("uniform", None), ("oracle_prior", true_share)]:
            pb = sinkhorn(cos, args.eps, prior).argmax(1)
            rep[f"acc_after_balancing_{tag}"] = float((pb == y).mean())
            rep[f"mpc_after_balancing_{tag}"] = float(np.nanmean(per_class_recall(pb, y, C)))
        rep.update(synonym_report(pred, y, K))
        rep.update({f"gt_captions/{k}": v for k, v in caption_report(pred, y, logp, r_gt).items()})
        rep.update({f"pseudo_captions/{k}": v for k, v in caption_report(pred, y, logp, r_ps).items()})
        summary[name] = rep
        per_class[f"pred_share_{name}"], per_class[f"recall_{name}"] = share, recall
    per_class["recall_gt_captions_only"] = per_class_recall(r_gt.argmax(1), y, C)
    per_class["recall_pseudo_captions_only"] = per_class_recall(r_ps.argmax(1), y, C)

    pd.DataFrame(per_class).to_csv(os.path.join(args.out, "per_class.csv"), index=False, float_format="%.3f")
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    table = pd.DataFrame({k: v for k, v in summary.items() if k != "class_prior"})
    with pd.option_context("display.max_rows", None, "display.width", 200, "display.float_format", "{:.3f}".format):
        print(table)
    print("class prior:", prior_rep)


if __name__ == "__main__":
    main()