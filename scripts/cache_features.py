"""Extract frozen CLIP features once, for later adapter runs.

Two files per model, both under data/features/:

  <model>.pt       eval-transform features of every image in A, B and C and of every
                   caption in B and C. Read by evaluation, and by training with
                   training.use_train_transform=false (residual) or by the semi
                   trainer, which always trains on cached features.
  <model>_aug.pt   K random pixel-level views (src/augment.py) of every Split A image,
                   shape [n_A, K, d]. Read by the semi trainer for the consistency and
                   pseudo-label losses (training.unlabeled.view=pixel). Skipped with
                   --aug-views 0.

The Split A manifest is read through load_unlabeled, which only exposes file paths,
so no caption or label of A is ever touched here.

    python scripts/cache_features.py
    python scripts/cache_features.py --model configs/model/vit_l14_laion2b.yaml --device mps
    python scripts/cache_features.py --aug-views 8 --aug-splits A,B
    python scripts/cache_features.py --check     # compare cached vs. online zero-shot accuracy on B
"""
import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.augment import build_augment_transform, build_weak_augment_transform  # noqa: E402
from src.data import CAPTION_COLS, class_prompt_names, load_classes, load_test, load_unlabeled  # noqa: E402
from src.models import build_model, get_device  # noqa: E402


class _Paths(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("RGB"))


class _Views(Dataset):
    """K independent augmented views of one image, stacked."""

    def __init__(self, paths, transform, views):
        self.paths = paths
        self.transform = transform
        self.views = views

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image = Image.open(self.paths[i]).convert("RGB")
        return torch.stack([self.transform(image) for _ in range(self.views)])


def _unique(values):
    seen = []
    known = set()
    for value in values:
        if value not in known:
            known.add(value)
            seen.append(value)
    return seen


def _seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


@torch.no_grad()
def _encode_images(model, paths, transform, device, batch_size, num_workers):
    loader = DataLoader(_Paths(paths, transform), batch_size=batch_size, num_workers=num_workers)
    features = []
    for images in tqdm(loader, desc="images", unit="batch"):
        raw = model.encode_image(images.to(device)).float()
        features.append(F.normalize(raw, dim=-1).cpu())
    return torch.cat(features)


@torch.no_grad()
def _encode_views(model, paths, transform, views, device, batch_size, num_workers, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        _Views(paths, transform, views), batch_size=batch_size, num_workers=num_workers,
        worker_init_fn=_seed_worker, generator=generator)
    features = []
    for batch in tqdm(loader, desc=f"{views} views", unit="batch"):
        n, k = batch.shape[:2]
        raw = model.encode_image(batch.flatten(0, 1).to(device)).float()
        features.append(F.normalize(raw, dim=-1).view(n, k, -1).cpu())
    return torch.cat(features)


@torch.no_grad()
def _encode_texts(model, tokenizer, texts, device, batch_size):
    features = []
    steps = range(0, len(texts), batch_size)
    for start in tqdm(steps, desc="captions", unit="batch", total=(len(texts) + batch_size - 1) // batch_size):
        tokens = tokenizer(texts[start:start + batch_size]).to(device)
        raw = model.encode_text(tokens).float()
        features.append(F.normalize(raw, dim=-1).cpu())
    return torch.cat(features)


def _zero_shot_check(model, tokenizer, labeled, cached_paths, cached_feats, eval_transform, device, batch_size,
                     num_workers, manifest):
    """Cached vs. online zero-shot accuracy on B. The two must agree to within 0.1 point."""
    classes = load_classes(manifest)
    names = class_prompt_names(classes, {})
    prototypes = _encode_texts(model, tokenizer, [f"A satellite image of {n}." for n in names], device, batch_size)
    index = {p: i for i, p in enumerate(cached_paths)}
    paths = labeled["filepath"].astype(str).tolist()
    labels = torch.tensor(labeled["label"].to_numpy())
    cached = cached_feats[torch.tensor([index[p] for p in paths])]
    online = _encode_images(model, paths, eval_transform, device, batch_size, num_workers)
    acc_cached = ((cached @ prototypes.T).argmax(1) == labels).float().mean().item()
    acc_online = ((online @ prototypes.T).argmax(1) == labels).float().mean().item()
    diff = abs(acc_cached - acc_online) * 100
    print(f"zero-shot on B: cached {100 * acc_cached:.2f}  online {100 * acc_online:.2f}  |diff| {diff:.3f} pt")
    if diff >= 0.1:
        raise SystemExit("cached features disagree with online encoding; do not use this cache")


def main():
    parser = argparse.ArgumentParser(description="Cache frozen CLIP features")
    parser.add_argument("--model", default=os.path.join(ROOT, "configs/model/vit_b32_laion2b.yaml"))
    parser.add_argument("--manifest", default=os.path.join(ROOT, "data/splits"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out", default=None, help="eval cache path; default data/features/<model>.pt")
    parser.add_argument("--aug-views", type=int, default=4, help="K views per image for the augmented cache; 0 skips it")
    parser.add_argument("--aug-splits", default="A", help="comma list of A,B: which images get augmented views")
    parser.add_argument("--aug-seed", type=int, default=0)
    parser.add_argument("--weak-views", type=int, default=0, help="K views per image for the *weak*-aug cache (scale 0.9-1.0, hflip only); 0 skips it")
    parser.add_argument("--weak-splits", default="A", help="comma list of A,B: which images get weak views")
    parser.add_argument("--skip-eval-cache", action="store_true", help="only write the augmented cache")
    parser.add_argument("--check", action="store_true", help="compare cached and online zero-shot accuracy on B")
    args = parser.parse_args()

    model_cfg = OmegaConf.load(args.model)
    out = args.out or os.path.join(ROOT, "data/features", f"{model_cfg.name}.pt")
    aug_out = os.path.splitext(out)[0] + "_aug.pt"
    weak_out = os.path.splitext(out)[0] + "_weak_aug.pt"
    device = get_device(args.device)
    model, _, eval_transform, tokenizer = build_model(model_cfg, device)
    model.eval()

    unlabeled = load_unlabeled(args.manifest)
    labeled = pd.read_csv(os.path.join(args.manifest, "B_labeled.csv"))
    test = load_test(args.manifest)
    a_paths = unlabeled["filepath"].astype(str).tolist()
    b_paths = labeled["filepath"].astype(str).tolist()
    os.makedirs(os.path.dirname(out), exist_ok=True)

    if not args.skip_eval_cache:
        paths = _unique(a_paths + b_paths + test["filepath"].astype(str).tolist())
        texts = _unique(
            [str(value) for frame in (labeled, test) for column in CAPTION_COLS for value in frame[column].tolist()])
        print(f"encoding {len(paths)} images and {len(texts)} captions on {device}")
        image_features = _encode_images(model, paths, eval_transform, device, args.batch_size, args.num_workers)
        blob = {
            "arch": model_cfg.arch,
            "pretrained": model_cfg.pretrained,
            "precision": model_cfg.precision,
            "transform": "eval",
            "image_paths": paths,
            "image_features": image_features,
            "texts": texts,
            "text_features": _encode_texts(model, tokenizer, texts, device, args.batch_size),
        }
        torch.save(blob, out)
        print(f"saved {out}  images {tuple(blob['image_features'].shape)}  texts {tuple(blob['text_features'].shape)}")
        if args.check:
            _zero_shot_check(model, tokenizer, labeled, paths, image_features, eval_transform, device,
                             args.batch_size, args.num_workers, args.manifest)

    if args.aug_views > 0:
        splits = [s.strip().upper() for s in args.aug_splits.split(",") if s.strip()]
        unknown = [s for s in splits if s not in ("A", "B")]
        if unknown:
            raise SystemExit(f"--aug-splits only accepts A and B, got {unknown}; C is never augmented")
        aug_paths = _unique((a_paths if "A" in splits else []) + (b_paths if "B" in splits else []))
        transform = build_augment_transform(eval_transform)
        print(f"encoding {args.aug_views} views of {len(aug_paths)} images ({'+'.join(splits)}) on {device}")
        print(transform)
        features = _encode_views(
            model, aug_paths, transform, args.aug_views, device, max(1, args.batch_size // args.aug_views),
            args.num_workers, args.aug_seed)
        blob = {
            "arch": model_cfg.arch,
            "pretrained": model_cfg.pretrained,
            "precision": model_cfg.precision,
            "transform": "aug",
            "views": args.aug_views,
            "splits": splits,
            "seed": args.aug_seed,
            "augment": str(transform),
            "image_paths": aug_paths,
            "image_features": features.half(),
        }
        torch.save(blob, aug_out)
        print(f"saved {aug_out}  views {tuple(features.shape)}")

    if args.weak_views > 0:
        w_splits = [s.strip().upper() for s in args.weak_splits.split(",") if s.strip()]
        unknown = [s for s in w_splits if s not in ("A", "B")]
        if unknown:
            raise SystemExit(f"--weak-splits only accepts A and B, got {unknown}")
        weak_paths = _unique((a_paths if "A" in w_splits else []) + (b_paths if "B" in w_splits else []))
        weak_transform = build_weak_augment_transform(eval_transform)
        print(f"encoding {args.weak_views} weak views of {len(weak_paths)} images ({'+'.join(w_splits)}) on {device}")
        print(weak_transform)
        weak_features = _encode_views(
            model, weak_paths, weak_transform, args.weak_views, device,
            max(1, args.batch_size // args.weak_views), args.num_workers, args.aug_seed)
        weak_blob = {
            "arch": model_cfg.arch,
            "pretrained": model_cfg.pretrained,
            "precision": model_cfg.precision,
            "transform": "weak_aug",
            "views": args.weak_views,
            "splits": w_splits,
            "seed": args.aug_seed,
            "augment": str(weak_transform),
            "image_paths": weak_paths,
            "image_features": weak_features.half(),
        }
        torch.save(weak_blob, weak_out)
        print(f"saved {weak_out}  views {tuple(weak_features.shape)}")



if __name__ == "__main__":
    main()
