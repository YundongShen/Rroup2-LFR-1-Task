"""Contrastive caption fine-tuning with real and/or VLM-generated captions.

    L = L_con(B-train, b_source captions) + a_weight * L_con(A, VLM captions)

b_source:
  real   : sample one of the 5 human captions per step (standard)
  real1  : one fixed random human caption per image (count-matched to vlm)
  vlm    : the VLM caption for the B image (pilot: how good are synthetic captions as supervision?)
a_source:
  none | vlm : add split-A images paired with their VLM captions (A's human captions are never read)
"""
import json
import math
import os
import random

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data import CAPTION_COLS


def load_vlm(path):
    return {json.loads(l)["filename"]: json.loads(l)["caps"] for l in open(path)} if path else {}


class Pairs(Dataset):
    """(image, caption-list) pairs; one caption is sampled per call."""

    def __init__(self, paths, caps, transform, tokenizer):
        self.paths, self.caps, self.tf, self.tok = paths, caps, transform, tokenizer

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = self.tf(Image.open(self.paths[i]).convert("RGB"))
        return img, self.tok([random.choice(self.caps[i])])[0]


def clip_loss(model, img, txt):
    zi = F.normalize(model.encode_image(img), dim=-1)
    zt = F.normalize(model.encode_text(txt), dim=-1)
    logits = model.logit_scale.exp() * zi @ zt.T
    y = torch.arange(len(img), device=img.device)
    return (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y)) / 2


def cycle(loader):
    while True:
        yield from loader


class CapFT:
    trains = True

    def __init__(self, cfg):
        self.cfg = cfg

    def _b_pairs(self, b_train, vlm, rng):
        m = self.cfg.method
        paths = b_train.filepath.tolist()
        if m.b_source == "real":
            caps = b_train[CAPTION_COLS].values.tolist()
        elif m.b_source == "real1":
            caps = [[rng.choice(r)] for r in b_train[CAPTION_COLS].values.tolist()]
        elif m.b_source == "vlm":
            caps = [vlm[os.path.basename(p)] for p in paths]
        else:
            raise ValueError(m.b_source)
        return paths, caps

    def fit(self, model, tokenizer, train_tf, eval_tf, b_train, b_val, a_unlabeled, device, out_dir):
        m, t = self.cfg.method, self.cfg.train
        rng = random.Random(self.cfg.seed)
        paths, caps = self._b_pairs(b_train, load_vlm(m.b_captions) if m.b_source == "vlm" else {}, rng)
        kw = dict(batch_size=t.batch_size, shuffle=True, drop_last=True, num_workers=t.num_workers,
                  persistent_workers=t.num_workers > 0)
        b_loader = DataLoader(Pairs(paths, caps, train_tf, tokenizer), **kw)

        a_iter = None
        if m.a_source == "vlm":
            vlm_a = load_vlm(m.a_captions)
            a_paths = [p for p in a_unlabeled.filepath if os.path.basename(p) in vlm_a]
            a_caps = [vlm_a[os.path.basename(p)] for p in a_paths]
            print(f"A: {len(a_paths)} images with VLM captions")
            a_iter = cycle(DataLoader(Pairs(a_paths, a_caps, train_tf, tokenizer), **kw))

        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=t.lr, weight_decay=t.weight_decay)
        steps = t.epochs * len(b_loader)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1, (s + 1) / t.warmup_steps) * 0.5 * (1 + math.cos(math.pi * s / steps)))

        model.train()
        step = 0
        for ep in range(t.epochs):
            tot = 0.0
            for img, txt in b_loader:
                loss = clip_loss(model, img.to(device), txt.to(device))
                if a_iter is not None:
                    ia, ta = next(a_iter)
                    loss = loss + m.a_weight * clip_loss(model, ia.to(device), ta.to(device))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step(); sched.step(); step += 1
                with torch.no_grad():
                    model.logit_scale.clamp_(0, math.log(100))
                tot += loss.item()
            print(f"  epoch {ep + 1}/{t.epochs} loss {tot / len(b_loader):.4f}")
        return model
