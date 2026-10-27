"""ROSE: Rotation-Orbit Self-training with Equipartitioned pseudo-labels. Uses split A images only.

Every epoch the current model relabels A: image features are averaged over the 8 D4 views (overhead
imagery has no canonical orientation) and scored against the class-prompt embeddings W; Sinkhorn then
equipartitions the assignment over classes, countering zero-shot CLIP's bias towards a few scene types,
and its argmax is the (hard) pseudo-label q.
The image tower is trained on strongly augmented, randomly rotated/flipped views x~:
    L = CE(q, softmax(s * f(x~) W^T)) + lambda * KL(P0 || P)
with P0 / P the row-softmaxed image-image similarities of the batch under the frozen / current image tower
(relational anchor: keeps instance-level neighbourhoods, which caption retrieval relies on).
The text tower is frozen, so class prompts and captions stay in the pretrained text space.
"""
import copy
import json
import math
import os

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data import class_prompt_names, load_classes
from src.evaluate import class_embeddings, evaluate
from src.ops import d4, encode_images_d4, sinkhorn


class AugImages(Dataset):
    def __init__(self, df, transform, rotate):
        self.paths = df.filepath.tolist()
        self.transform = transform
        self.rotate = rotate

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        x = self.transform(Image.open(self.paths[i]).convert("RGB"))
        if self.rotate:
            x = d4(x[None], int(torch.randint(8, ()))).squeeze(0)
        return i, x


def strong_transform(eval_tf, size):
    return T.Compose([T.RandomResizedCrop(size, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC),
                      T.RandomApply([T.ColorJitter(0.4, 0.4, 0.3, 0.05)], p=0.8), T.RandomGrayscale(0.1),
                      T.ToTensor(), eval_tf.transforms[-1]])


def set_trainable(model, mode):
    """visual | ln (LayerNorm affine only) | lastN (last N blocks + ln_post + proj). Text tower stays frozen."""
    model.requires_grad_(False)
    v = model.visual
    if mode == "visual":
        params = list(v.parameters())
    elif mode == "ln":
        params = [p for m in v.modules() if isinstance(m, torch.nn.LayerNorm) for p in m.parameters()]
    elif mode.startswith("last"):
        params = [p for blk in v.transformer.resblocks[-int(mode[4:]):] for p in blk.parameters()]
        params += list(v.ln_post.parameters()) + [v.proj]
    else:
        raise ValueError(f"unknown trainable mode: {mode}")
    for p in params:
        p.requires_grad_(True)
    return params


@torch.no_grad()
def pseudo_labels(model, df, eval_tf, W, m, dev, num_workers, scale):
    """Soft label distribution q [N, C] over classes for every image of df."""
    f = encode_images_d4(model, df, eval_tf, dev, num_workers=num_workers, group=range(8) if m.orbit else [0])
    s = F.normalize(f.mean(1), dim=-1) @ W.T.cpu()
    if m.labels == "sinkhorn":
        return sinkhorn(s, m.eps)
    if m.labels == "argmax":
        return (scale * s).softmax(1)
    raise ValueError(f"unknown labels: {m.labels}")


def relational_kl(z0, z, temp):
    """KL(P0 || P) between row-softmaxed in-batch similarity matrices (self-similarity masked out)."""
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    s0 = (z0 @ z0.T / temp).masked_fill(eye, -1e4)
    s = (z @ z.T / temp).masked_fill(eye, -1e4)
    return F.kl_div(F.log_softmax(s, 1), F.log_softmax(s0, 1), log_target=True, reduction="batchmean")


class ROSE:
    trains = True

    def __init__(self, cfg):
        self.cfg = cfg

    def fit(self, model, tokenizer, train_tf, eval_tf, b_train, b_val, a_unlabeled, device, out_dir):
        cfg, m, t = self.cfg, self.cfg.method, self.cfg.train
        names = class_prompt_names(load_classes(cfg.data.manifest_dir), cfg.prompts.class_names)
        templates = list(cfg.prompts.templates)
        W = class_embeddings(model, tokenizer, names, templates, device).to(device)
        a = a_unlabeled if not t.max_images else a_unlabeled.sample(n=t.max_images, random_state=cfg.seed)
        anchor = copy.deepcopy(model.visual).eval().requires_grad_(False) if m.anchor_weight > 0 else None
        params = set_trainable(model, t.trainable)
        opt = torch.optim.AdamW(params, lr=t.lr, weight_decay=t.weight_decay)
        loader = DataLoader(AugImages(a, strong_transform(eval_tf, model.visual.image_size), m.orbit),
                            batch_size=t.batch_size, shuffle=True, drop_last=True, num_workers=t.num_workers,
                            pin_memory=True, persistent_workers=True)
        total = t.epochs * len(loader)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0, (k + 1) / t.warmup_steps)
                                                  * 0.5 * (1 + math.cos(math.pi * min(k, total) / total)))
        scale = model.logit_scale.exp().item()
        self.history, prev = [], None
        for ep in range(t.epochs):
            model.eval()
            if ep == 0 or m.refresh:
                q = pseudo_labels(model, a, eval_tf, W, m, device, t.num_workers, scale)
                conf, hard = q.max(1)
                keep = (conf >= m.min_conf).float()
                # soft targets + per-epoch relabelling feed back into ever flatter labels (observed collapse)
                target = q if m.soft_labels else F.one_hot(hard, len(names)).float()
                p = torch.bincount(hard, minlength=len(names)).float() / len(hard)
                pl_stats = {"pl_mean_conf": conf.mean().item(), "pl_kept": keep.mean().item(),
                            "pl_class_entropy": -(p * (p + 1e-12).log()).sum().item() / math.log(len(names)),
                            "pl_changed": float("nan") if prev is None else (hard != prev).float().mean().item()}
                prev = hard
            model.train()
            sums = {"loss_pl": 0.0, "loss_anchor": 0.0}
            for idx, x in loader:
                x = x.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    z = F.normalize(model.encode_image(x).float(), dim=-1)
                    z0 = F.normalize(anchor(x).float(), dim=-1) if anchor is not None else None
                w = keep[idx].to(device)
                ce = -(target[idx].to(device) * F.log_softmax(scale * z @ W.T, 1)).sum(1)
                loss_pl = (w * ce).sum() / w.sum().clamp(min=1)
                loss_anchor = relational_kl(z0, z, m.anchor_temp) if anchor is not None else torch.zeros((), device=device)
                loss = loss_pl + m.anchor_weight * loss_anchor
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                sums["loss_pl"] += loss_pl.item() / len(loader)
                sums["loss_anchor"] += loss_anchor.item() / len(loader)
            rec = {"epoch": ep, **pl_stats, **sums}
            if m.track_val:  # learning curves on B only; no checkpoint selection
                rec.update(evaluate(model, tokenizer, eval_tf, b_val, device, cfg.eval, names, templates))
            self.history.append(rec)
            print("[rose] " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
        os.makedirs(out_dir, exist_ok=True)
        json.dump(self.history, open(os.path.join(out_dir, "history.json"), "w"), indent=1)
        if t.save_model:
            torch.save(model.visual.state_dict(), os.path.join(out_dir, "visual.pt"))
        model.eval()
        return model
