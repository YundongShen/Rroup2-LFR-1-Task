"""Soft caption pseudo-supervision for Split A: dual-path retrieval over the B-train caption bank.

For an unlabeled image u and every unique B-train caption j:

    S_IT(u, j) = cos(z_u, t_j)                      image -> text
    S_II(u, j) = max_{i : j is a caption of i} cos(z_u, z_i)   image -> image, then the image's captions
    R(u, j)    = lam * z(S_IT) + (1 - lam) * z(S_II)          z-scored per image over the whole bank

The top-k captions by R are the candidates N_u, weighted by softmax(R / tau_p). An image
is accepted when its R_max is in the top `coverage` fraction of Split A this epoch. The
student then minimizes the weighted cross-entropy of its image-text softmax over the whole
bank, i.e. -sum_j w_uj log p(j | x_u), on the accepted rows.

The scores come from the EMA teacher (or the frozen CLIP), never from the student.
scripts/caption_bank_eval.py validates the retrieval offline on B with held-out captions.
"""
import math

import torch
import torch.nn.functional as F


def build_caption_bank(frame, caption_cols):
    """Unique caption strings of a labeled frame, and for each one the row indices it belongs to."""
    captions, sources, index = [], [], {}
    for i, row in enumerate(frame[caption_cols].itertuples(index=False)):
        for value in row:
            text = str(value)
            if text not in index:
                index[text] = len(captions)
                captions.append(text)
                sources.append([])
            if i not in sources[index[text]]:
                sources[index[text]].append(i)
    return captions, sources


def caption_image_scores(s_img, sources):
    """[nq, n_images] image-image cosine -> [nq, n_captions], max over each caption's source images."""
    cap_idx = torch.tensor([j for j, src in enumerate(sources) for _ in src], device=s_img.device)
    img_idx = torch.tensor([i for src in sources for i in src], device=s_img.device)
    out = torch.full((s_img.shape[0], len(sources)), float("-inf"), device=s_img.device, dtype=s_img.dtype)
    return out.scatter_reduce(1, cap_idx.expand(s_img.shape[0], -1), s_img[:, img_idx], reduce="amax")


def zscore(s):
    return (s - s.mean(1, keepdim=True)) / s.std(1, keepdim=True).clamp_min(1e-8)


@torch.no_grad()
def fused_scores(q_img, bank_img, bank_txt, sources, lam):
    """R = lam * z(S_IT) + (1 - lam) * z(S_II). All features L2-normalized. Returns (R, S_IT, S_II)."""
    s_it = q_img @ bank_txt.T
    s_ii = caption_image_scores(q_img @ bank_img.T, sources)
    return lam * zscore(s_it) + (1 - lam) * zscore(s_ii), s_it, s_ii


@torch.no_grad()
def assign_captions(R, top_k=5, tau_p=0.3, coverage=0.2):
    """Candidates, soft weights and the acceptance mask for every row of R.

    tau_p = inf gives uniform weights. coverage is the fraction of rows accepted, the ones
    with the highest R_max; the implied R threshold is reported in the stats.
    """
    n = R.shape[0]
    scores, cand = R.topk(top_k, dim=1)                                        # [n, k]
    if math.isinf(tau_p):
        weights = torch.full_like(scores, 1.0 / top_k)
    else:
        weights = F.softmax(scores / float(tau_p), dim=1)
    r_max = scores[:, 0]
    n_accept = int(round(float(coverage) * n))
    mask = torch.zeros(n, dtype=torch.bool, device=R.device)
    threshold = float("inf")
    if n_accept > 0:
        order = r_max.argsort(descending=True)
        mask[order[:n_accept]] = True
        threshold = r_max[order[n_accept - 1]].item()
    entropy = -(weights * weights.clamp_min(1e-12).log()).sum(1)
    quantiles = torch.quantile(r_max.float(), torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=R.device))
    stats = {
        "n_total": n,
        "n_accepted": int(mask.sum()),
        "r_threshold": threshold,
        "r_max_quantiles": [round(v, 4) for v in quantiles.tolist()],
        "weight_entropy": float(entropy.mean()),
        "weight_entropy_max": math.log(top_k),
        "top1_weight": float(weights[:, 0].mean()),
    }
    return {"candidates": cand, "weights": weights, "mask": mask, "r_max": r_max}, stats


def pseudo_caption_loss(img, bank_txt, candidates, weights, mask, logit_scale):
    """Weighted soft cross-entropy of the image-text softmax over the whole caption bank.

    img [b, d] normalized student features; bank_txt [nt, d] normalized; candidates / weights [b, k];
    mask [b] rows that take part. Returns the mean over the masked rows, or 0 with a graph when none.
    """
    if not bool(mask.any()):
        return img.sum() * 0.0
    logits = logit_scale * (img[mask] @ bank_txt.T)                           # [m, nt]
    log_p = F.log_softmax(logits, dim=-1)
    picked = log_p.gather(1, candidates[mask])                                 # [m, k]
    return -(weights[mask] * picked).sum(1).mean()
