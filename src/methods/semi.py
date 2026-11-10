"""Method 2 (semi-supervised) and method 1 (`pcap.enabled=false`), see README "Method".

CLIP stays frozen; the only trainable module is an image-side residual adapter
    z~ = z + alpha * h(z),   z = normalize(CLIP image feature).
Training runs on cached frozen features (src/features.py). After training the adapter is wrapped
into `model.encode_image`, so the repository's standard `evaluate` sees the adapted model.

Losses
  L_con  : symmetric InfoNCE on B-train pairs (1 random caption per image per step); a caption in
           the batch whose text also belongs to image i is not a negative for i.
  L_pcap : soft cross-entropy over the caption bank (unique B-train caption texts of this fold) for
           the A images accepted this epoch; candidates/weights come from the EMA teacher.
  L = L_con + lambda_pcap(t) * L_pcap, lambda_pcap ramps 0 -> lambda over the first `ramp` of steps.
"""
import copy
import json
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data import CAPTION_COLS, class_prompt_names, load_classes
from src.evaluate import retrieval, zero_shot
from src.features import FeatureCache


class Adapter(nn.Module):
    def __init__(self, dim, hidden, alpha):
        super().__init__()
        self.alpha = alpha
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        nn.init.zeros_(self.fc2.weight)       # adapter == identity at initialisation
        nn.init.zeros_(self.fc2.bias)

    def forward(self, z):
        return z + self.alpha * self.fc2(self.act(self.fc1(z)))


def wrap_encode_image(model, adapter):
    """Make model.encode_image return adapter(normalize(original encode_image(x)))."""
    orig = model.encode_image
    model.adapter = adapter                    # register as submodule (moves with .to / .eval)

    def encode_image(image, normalize=False):
        out = adapter(F.normalize(orig(image).float(), dim=-1))
        return F.normalize(out, dim=-1) if normalize else out

    model.encode_image = encode_image
    return model


def zscore_rows(x):
    return (x - x.mean(1, keepdim=True)) / x.std(1, keepdim=True, unbiased=False).clamp_min(1e-8)


def build_bank(b_train):
    """Unique B-train caption texts (sorted), bank id per (image, slot), owner matrix [NB, NJ]."""
    caps = b_train[CAPTION_COLS].values                        # [NB, 5]
    bank = sorted(set(caps.reshape(-1).tolist()))
    pos = {t: j for j, t in enumerate(bank)}
    bank_id = torch.tensor([[pos[t] for t in row] for row in caps])   # [NB, 5]
    owns = torch.zeros(len(caps), len(bank), dtype=torch.bool)
    owns[torch.arange(len(caps))[:, None], bank_id] = True
    return bank, bank_id, owns


@torch.no_grad()
def assign_pseudo_captions(zA_t, zB_t, tB, bank_id, w_it, topk, tau, coverage):
    """Teacher-side pseudo-caption assignment for all A images.

    zA_t [NA,d], zB_t [NB,d]: teacher-adapted, L2-normalised image features; tB [NJ,d] bank text
    features; bank_id [NB,5]. Returns candidates [NA,k], weights [NA,k], confidence [NA], kept idx.
    """
    NA, NJ = zA_t.shape[0], tB.shape[0]
    S_it = zA_t @ tB.T                                          # [NA, NJ]
    S_img = zA_t @ zB_t.T                                       # [NA, NB]
    S_ii = torch.full((NA, NJ), -float("inf"), device=zA_t.device, dtype=zA_t.dtype)
    idx = bank_id.reshape(1, -1).expand(NA, -1)                 # column i*5+k -> bank id of image i slot k
    S_ii.scatter_reduce_(1, idx, S_img.repeat_interleave(bank_id.shape[1], dim=1), reduce="amax")
    R = w_it * zscore_rows(S_it) + (1 - w_it) * zscore_rows(S_ii)
    top_r, cand = R.topk(topk, dim=1)
    w = torch.softmax(top_r / tau, dim=1)
    conf = top_r[:, 0]
    n_keep = int(round(coverage * NA))
    keep = conf.topk(n_keep).indices.sort().values
    return cand, w, conf, keep


def contrastive_loss(zi, tj, owns_batch, scale):
    """Symmetric InfoNCE; owns_batch[i,k]=True if caption k's text belongs to image i."""
    n = zi.shape[0]
    logits = scale * zi @ tj.T
    eye = torch.eye(n, dtype=torch.bool, device=zi.device)
    logits = logits.masked_fill(owns_batch & ~eye, -float("inf"))
    y = torch.arange(n, device=zi.device)
    return 0.5 * (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y))


def pcap_loss(zu, tB, cand, w, scale):
    logp = torch.log_softmax(scale * zu @ tB.T, dim=1)          # over the whole bank
    return -(w * logp.gather(1, cand)).sum(1).mean()


class _IndexStream:
    """Endless stream of shuffled indices in chunks of `bs` (reshuffled every pass)."""

    def __init__(self, n, bs, gen):
        self.n, self.bs, self.gen, self.buf = n, bs, gen, torch.empty(0, dtype=torch.long)

    def next(self):
        while len(self.buf) < self.bs:
            self.buf = torch.cat([self.buf, torch.randperm(self.n, generator=self.gen)])
        out, self.buf = self.buf[:self.bs], self.buf[self.bs:]
        return out


class Semi:
    trains = True

    def __init__(self, cfg):
        self.cfg = cfg
        self.m = cfg.method
        self.n_fit = 0
        classes = load_classes(cfg.data.manifest_dir)
        self.class_names = class_prompt_names(classes, cfg.prompts.class_names)
        self.templates = list(cfg.prompts.templates)
        self.last_cached_metrics = None

    def fit(self, model, tokenizer, train_tf, eval_tf, b_train, b_val, a_unlabeled, dev, out_dir):
        m = self.m
        base = os.path.basename(os.path.normpath(out_dir))      # run.py passes <run_dir>/fold{i}
        fold = int(base[4:]) if base.startswith("fold") and base[4:].isdigit() else self.n_fit
        self.n_fit += 1
        os.makedirs(out_dir, exist_ok=True)
        seed = int(self.cfg.seed) * 1000 + fold
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed)

        # ---- frozen features (A and all of B are cached once per model; bank = B-train only)
        cache = FeatureCache(m.cache_dir, self.cfg.model.name).load_or_build(
            model, tokenizer, eval_tf, a_unlabeled, pd.concat([b_train, b_val]), dev)
        zA = FeatureCache.images(cache, a_unlabeled["filepath"]).to(dev)
        zB = FeatureCache.images(cache, b_train["filepath"]).to(dev)
        bank, bank_id, owns = build_bank(b_train)
        tB = FeatureCache.texts(cache, bank).to(dev)
        bank_id, owns = bank_id.to(dev), owns.to(dev)
        zV = FeatureCache.images(cache, b_val["filepath"]).to(dev)
        tV = FeatureCache.texts(cache, b_val[CAPTION_COLS].values.reshape(-1).tolist()).to(dev)
        scale = model.logit_scale.exp().item()
        NA, NB = zA.shape[0], zB.shape[0]
        print(f"[fold {fold}] NA={NA} NB={NB} bank={len(bank)} scale={scale:.2f} pcap={m.pcap.enabled}")

        # ---- adapter, EMA teacher, optimiser
        student = Adapter(zA.shape[1], m.adapter.hidden, m.adapter.alpha).to(dev)
        teacher = copy.deepcopy(student).requires_grad_(False)
        opt = torch.optim.AdamW(student.parameters(), lr=m.lr, weight_decay=m.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=m.steps)
        b_stream = _IndexStream(NB, m.batch_b, gen)
        ramp_steps = max(1, int(m.pcap.ramp * m.steps))

        def val_mR1(adapter):
            with torch.no_grad():
                r = retrieval(F.normalize(adapter(zV), dim=-1).cpu(), tV.cpu(), (1,))
            return 50 * (r["i2t_R@1"] + r["t2i_R@1"])

        curve, epochs, log = [], [], []
        a_queue = torch.empty(0, dtype=torch.long)
        cand = w = keep = None
        prev_top1 = None
        for step in range(m.steps):
            if m.monitor_every and step % m.monitor_every == 0:
                curve.append({"step": step, "student_mR1": val_mR1(student), "teacher_mR1": val_mR1(teacher)})

            # epoch start: teacher re-assigns pseudo-captions, A queue = shuffled accepted images
            if m.pcap.enabled and len(a_queue) == 0:
                with torch.no_grad():
                    zA_t = F.normalize(teacher(zA), dim=-1)
                    zB_t = F.normalize(teacher(zB), dim=-1)
                cand, w, conf, keep = assign_pseudo_captions(
                    zA_t, zB_t, tB, bank_id, m.pcap.w_it, m.pcap.topk, m.pcap.tau, m.pcap.coverage)
                top1 = cand[:, 0]
                epochs.append({"step": step, "n_keep": len(keep),
                               "conf_mean_kept": conf[keep].mean().item(),
                               "conf_threshold": conf[keep].min().item(),
                               "w_top1_mean_kept": w[keep, 0].mean().item(),
                               "top1_changed": None if prev_top1 is None
                               else (top1 != prev_top1).float().mean().item()})
                prev_top1 = top1
                a_queue = keep[torch.randperm(len(keep), generator=gen).to(dev)]

            # B contrastive batch
            ib = b_stream.next().to(dev)
            slot = torch.randint(5, (len(ib),), generator=gen).to(dev)
            cj = bank_id[ib, slot]
            zi = F.normalize(student(zB[ib]), dim=-1)
            l_con = contrastive_loss(zi, tB[cj], owns[ib][:, cj], scale)
            loss, rec = l_con, {"l_con": l_con.item()}

            if m.pcap.enabled:
                u, a_queue = a_queue[:m.batch_a], a_queue[m.batch_a:]
                zu = F.normalize(student(zA[u]), dim=-1)
                l_p = pcap_loss(zu, tB, cand[u], w[u], scale)
                lam = m.pcap["lambda"] * min(1.0, step / ramp_steps)
                loss = loss + lam * l_p
                rec.update(l_pcap=l_p.item(), lam_pcap=lam)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if m.pcap.enabled:
                with torch.no_grad():
                    for pt, ps in zip(teacher.parameters(), student.parameters()):
                        pt.mul_(m.ema).add_(ps, alpha=1 - m.ema)
            if step % 100 == 0 or step == m.steps - 1:
                rec["step"] = step
                log.append(rec)
                print(f"[fold {fold}] step {step} " + " ".join(f"{k}={v:.4f}" for k, v in rec.items() if k != "step"))

        if m.monitor_every:
            curve.append({"step": m.steps, "student_mR1": val_mR1(student), "teacher_mR1": val_mR1(teacher)})
        student.eval()

        # cached-feature evaluation on B-val, compared later with the standard evaluate() path
        with torch.no_grad():
            zv = F.normalize(student(zV), dim=-1).cpu()
        labels = torch.tensor(b_val["label"].values)
        cm = zero_shot(model, tokenizer, zv, labels, self.class_names, self.templates, dev)
        cm.update(retrieval(zv, tV.cpu(), tuple(self.cfg.eval.tasks.retrieval.ks)))
        self.last_cached_metrics = cm

        torch.save({"student": student.state_dict(), "teacher": teacher.state_dict()},
                   os.path.join(out_dir, "adapter.pt"))
        json.dump({"curve": curve, "epochs": epochs, "log": log, "bank_size": len(bank)},
                  open(os.path.join(out_dir, "train_log.json"), "w"), indent=1)
        return wrap_encode_image(model, student)
