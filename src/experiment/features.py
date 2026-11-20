"""Frozen CLIP features for the deterministic eval transform.

One vector per image and per caption string. Training can reuse the file only
when it uses that same transform. A random crop produces a different image
feature on every read, so it cannot be stored once.
"""
import os

import torch
from torch.utils.data import Dataset

from src.data import CAPTION_COLS


class FeatureCache:
    def __init__(self, blob):
        self.arch = blob["arch"]
        self.pretrained = blob["pretrained"]
        self.image_index = {path: i for i, path in enumerate(blob["image_paths"])}
        self.image_features = blob["image_features"].float()
        self.text_index = {text: i for i, text in enumerate(blob["texts"])}
        self.text_features = blob["text_features"].float()

    def has_images(self, paths):
        return all(path in self.image_index for path in paths)

    def has_texts(self, texts):
        return all(text in self.text_index for text in texts)

    def image(self, path):
        return self.image_features[self.image_index[path]]

    def text(self, text):
        return self.text_features[self.text_index[text]]

    def images(self, paths):
        index = torch.tensor([self.image_index[path] for path in paths], dtype=torch.long)
        return self.image_features[index]

    def texts(self, texts):
        index = torch.tensor([self.text_index[text] for text in texts], dtype=torch.long)
        return self.text_features[index]


def load_feature_cache(path, model_cfg):
    """Return a cache for this checkpoint, or None when the file is absent."""
    if not path or not os.path.isfile(path):
        return None
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob["arch"] != model_cfg.arch or blob["pretrained"] != model_cfg.pretrained:
        raise ValueError(
            f"feature cache {path} was built for {blob['arch']} / {blob['pretrained']}, "
            f"not {model_cfg.arch} / {model_cfg.pretrained}. Run scripts/cache_features.py again.")
    if blob.get("transform") != "eval":
        raise ValueError(f"feature cache {path} is not an eval-transform cache")
    return FeatureCache(blob)


class AugmentedFeatureCache:
    """K frozen CLIP features per image, one per random pixel-level view (scripts/cache_features.py --aug-views)."""

    def __init__(self, blob):
        self.arch = blob["arch"]
        self.pretrained = blob["pretrained"]
        self.views = int(blob["views"])
        self.image_index = {path: i for i, path in enumerate(blob["image_paths"])}
        self.image_features = blob["image_features"].float()  # [n, K, d]

    def has_images(self, paths):
        return all(path in self.image_index for path in paths)

    def images(self, paths):
        """All K views: [len(paths), K, d]."""
        index = torch.tensor([self.image_index[path] for path in paths], dtype=torch.long)
        return self.image_features[index]


def load_aug_cache(path, model_cfg):
    """Return the augmented-view cache for this checkpoint, or None when the file is absent."""
    if not path or not os.path.isfile(path):
        return None
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob["arch"] != model_cfg.arch or blob["pretrained"] != model_cfg.pretrained:
        raise ValueError(
            f"augmented cache {path} was built for {blob['arch']} / {blob['pretrained']}, "
            f"not {model_cfg.arch} / {model_cfg.pretrained}. Run scripts/cache_features.py again.")
    if blob.get("transform") != "aug":
        raise ValueError(f"{path} is not an augmented-view cache")
    return AugmentedFeatureCache(blob)


class CachedPairs(Dataset):
    """Image and caption features already extracted by the frozen CLIP."""

    def __init__(self, df, cache, random_caption=True, caption_index=None):
        self.df = df
        self.cache = cache
        self.random_caption = random_caption
        self.caption_index = caption_index
        paths = df["filepath"].astype(str).tolist()
        missing = [path for path in paths if path not in cache.image_index]
        if missing:
            raise KeyError(f"{len(missing)} images are missing from the feature cache, for example {missing[0]}")
        texts = [str(value) for column in CAPTION_COLS for value in df[column].tolist()]
        missing_text = [text for text in texts if text not in cache.text_index]
        if missing_text:
            raise KeyError(
                f"{len(missing_text)} captions are missing from the feature cache, for example {missing_text[0]!r}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        if self.caption_index is not None:
            caption_index = self.caption_index
        elif self.random_caption:
            caption_index = torch.randint(5, ()).item()
        else:
            caption_index = 0
        return self.cache.image(str(row.filepath)), self.cache.text(str(row[CAPTION_COLS[caption_index]]))


def load_weak_aug_cache(path, model_cfg):
    """Return the weak-augmented-view cache for this checkpoint, or None when absent.

    Parallel to load_aug_cache but checks for transform='weak_aug'.
    Built by scripts/cache_features.py --weak-views K.
    """
    if not path or not os.path.isfile(path):
        return None
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob["arch"] != model_cfg.arch or blob["pretrained"] != model_cfg.pretrained:
        raise ValueError(
            f"weak-aug cache {path} was built for {blob['arch']} / {blob['pretrained']}, "
            f"not {model_cfg.arch} / {model_cfg.pretrained}. Run scripts/cache_features.py again.")
    if blob.get("transform") != "weak_aug":
        raise ValueError(f"{path} is not a weak-augmented-view cache (transform={blob.get('transform')!r})")
    return AugmentedFeatureCache(blob)
