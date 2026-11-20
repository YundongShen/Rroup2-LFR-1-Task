"""Pixel-level augmentation for the cached unlabeled views.

The frozen image encoder sees K random views of every Split A image once, in
scripts/cache_features.py. Training then reads those K features instead of
running the encoder. The transform ends with the CLIP Normalize of the model's
own eval transform, so the only difference from the eval view is the
augmentation itself.
"""
import random

import torch
from torchvision import transforms as T


class RandomRot90:
    """Rotate by a random multiple of 90 degrees. Aerial images have no canonical 'up'."""

    def __call__(self, image):
        return image.rotate(90 * random.randint(0, 3))

    def __repr__(self):
        return "RandomRot90()"


def normalize_stats(transform):
    """(mean, std) of the Normalize step inside an OpenCLIP transform."""
    for child in _iter_transforms(transform):
        if child.__class__.__name__ == "Normalize":
            return tuple(float(v) for v in child.mean), tuple(float(v) for v in child.std)
    raise ValueError("eval transform has no Normalize step")


def image_size(transform):
    for child in _iter_transforms(transform):
        if child.__class__.__name__ in ("CenterCrop", "Resize", "RandomResizedCrop"):
            size = child.size
            return int(size[0]) if isinstance(size, (tuple, list)) else int(size)
    raise ValueError("eval transform has no crop or resize step")


def _iter_transforms(transform):
    children = getattr(transform, "transforms", None)
    if children is None:
        yield transform
        return
    for child in children:
        yield from _iter_transforms(child)


def build_augment_transform(eval_transform, scale=(0.5, 1.0), jitter=0.4, hue=0.1):
    """RandomResizedCrop, flips, rot90, ColorJitter, then CLIP's own Normalize."""
    mean, std = normalize_stats(eval_transform)
    size = image_size(eval_transform)
    return T.Compose([
        T.RandomResizedCrop(size, scale=scale, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        RandomRot90(),
        T.ColorJitter(jitter, jitter, jitter, hue),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def feature_perturbation(features, kind, dropout_p=0.1, noise_sigma=0.02, generator=None):
    """Feature-level view of already normalized features. Returns an unnormalized tensor."""
    if kind == "none":
        return features
    if kind == "dropout":
        keep = torch.rand(features.shape, device=features.device, generator=generator) >= dropout_p
        return features * keep / (1.0 - dropout_p)
    if kind == "noise":
        noise = torch.randn(features.shape, device=features.device, generator=generator) * noise_sigma
        return features + noise
    raise ValueError(f"unknown feature perturbation {kind!r}, use none | dropout | noise")


def build_weak_augment_transform(eval_transform, scale=(0.9, 1.0)):
    """Weak augmentation for Split A image-image contrastive learning.

    Only a tight random crop (scale 0.9–1.0) and horizontal flip — both keep the
    depicted object intact. The resulting CLIP features sit very close to the
    eval-time center-crop feature, so gradients of the image-image InfoNCE fall in
    the same region of feature space that evaluation measures.

    Deliberately excludes:
      - vertical flip / rot90  (remote-sensing images have a canonical up in the dataset)
      - ColorJitter             (changes spectral signature; can swap land-cover class)
      - wide crops (< 0.9)      (risk of cropping to a different land-cover patch)
    """
    mean, std = normalize_stats(eval_transform)
    size = image_size(eval_transform)
    return T.Compose([
        T.RandomResizedCrop(size, scale=scale, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
