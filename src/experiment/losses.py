"""The four loss terms of the semi-supervised adapter.

    L = L_con + lambda_cls L_cls + lambda_cons L_cons + lambda_anc L_anc

Every function takes features that are already L2-normalized unless it says
otherwise. `logit_scale` is the scalar temperature (CLIP's logit_scale.exp(),
about 100), not its log.
"""
import torch
import torch.nn.functional as F


def clip_loss(img, txt, logit_scale, img_ids=None):
    """Symmetric InfoNCE between one image and one caption per row.

    img, txt: [b, d] normalized. img_ids: [b] ids of the source image; two rows that
    share an id are captions of the same image, and are not treated as negatives of
    each other.
    """
    logits = logit_scale * (img @ txt.T)
    if img_ids is not None:
        same = img_ids[:, None] == img_ids[None, :]
        same.fill_diagonal_(False)
        logits = logits.masked_fill(same, float("-inf"))
    target = torch.arange(logits.shape[0], device=logits.device)
    loss_i2t = F.cross_entropy(logits, target)
    loss_t2i = F.cross_entropy(logits.T, target)
    return 0.5 * (loss_i2t + loss_t2i), {"loss_i2t": loss_i2t.detach(), "loss_t2i": loss_t2i.detach()}


def cls_logits(img, prototypes, logit_scale):
    """logits = s * cos(z, P_c), the same quantity the zero-shot protocol argmaxes."""
    return logit_scale * (F.normalize(img, dim=-1) @ F.normalize(prototypes, dim=-1).T)


def cls_logits_per_image_template(img, bank, indices, logit_scale):
    """Same as cls_logits, but each image uses one template for every class.

    bank: [C, K, d] prompt embeddings. indices: [B] template ids in 0..K-1.
    Image b is scored against bank[:, indices[b]], so the softmax still has one logit per class.
    """
    img = F.normalize(img, dim=-1)
    picked = F.normalize(bank[:, indices].detach(), dim=-1)
    return logit_scale * torch.einsum("bd,cbd->bc", img, picked)


def cls_loss(img, prototypes, labels, mask, logit_scale):
    """Cross-entropy over the class prototypes on the rows where mask is True."""
    if mask is None:
        mask = torch.ones(img.shape[0], dtype=torch.bool, device=img.device)
    if not bool(mask.any()):
        return img.sum() * 0.0
    logits = cls_logits(img[mask], prototypes, logit_scale)
    return F.cross_entropy(logits, labels[mask])


def consistency_loss(student_logits, teacher_probs, kind="kl", temperature=1.0):
    """Student class distribution on the perturbed view vs. the (detached) teacher distribution.

    kind='kl': KL(p_teacher || p_student). kind='mse': squared error between the two
    probability vectors, summed over classes.
    """
    teacher_probs = teacher_probs.detach()
    log_student = F.log_softmax(student_logits / temperature, dim=-1)
    if kind == "kl":
        return F.kl_div(log_student, teacher_probs, reduction="batchmean")
    if kind == "mse":
        return (log_student.exp() - teacher_probs).pow(2).sum(-1).mean()
    raise ValueError(f"unknown consistency kind {kind!r}, use kl | mse")


def anchor_loss(adapted_raw, frozen, kind="l2", prototypes=None, logit_scale=None, frozen_probs=None):
    """Keep the adapted features near the frozen CLIP features.

    kind='l2': mean squared distance between the unnormalized residual output and the
    frozen (normalized) input. kind='kl': KL(p_clip || p_student) over the class
    prototypes; frozen_probs may be passed precomputed, otherwise it is computed from
    `frozen`.
    """
    if kind == "l2":
        return (adapted_raw - frozen).pow(2).sum(-1).mean()
    if kind == "kl":
        if prototypes is None or logit_scale is None:
            raise ValueError("kl anchor needs prototypes and logit_scale")
        if frozen_probs is None:
            frozen_probs = F.softmax(cls_logits(frozen, prototypes, logit_scale), dim=-1)
        log_student = F.log_softmax(cls_logits(adapted_raw, prototypes, logit_scale), dim=-1)
        return F.kl_div(log_student, frozen_probs.detach(), reduction="batchmean")
    raise ValueError(f"unknown anchor kind {kind!r}, use l2 | kl")


def ramp_up(step, total_steps, fraction):
    """Linear 0 -> 1 over the first `fraction` of all steps, then 1."""
    if fraction <= 0:
        return 1.0
    warm = max(1, int(round(fraction * total_steps)))
    return min(1.0, (step + 1) / warm)


def image_nce_loss(view1, view2, logit_scale):
    """Symmetric InfoNCE between two augmented views of the same batch of images.

    view1, view2: [b, d] L2-normalized. Diagonal entries are positive pairs; all
    off-diagonal entries are negatives (other images in the batch).

    Gradient flows through both views with no stop-gradient — symmetric pull.
    Used for the Split A image-image contrastive term (L_cons rewrite).
    """
    logits = logit_scale * (view1 @ view2.T)   # [b, b]
    target = torch.arange(logits.shape[0], device=logits.device)
    loss_12 = F.cross_entropy(logits, target)
    loss_21 = F.cross_entropy(logits.T, target)
    return 0.5 * (loss_12 + loss_21)
