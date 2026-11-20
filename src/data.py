import json
import os
import re

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

CAPTION_COLS = [f"caption_{k}" for k in range(5)]


def load_classes(manifest_dir):
    return json.load(open(os.path.join(manifest_dir, "classes.json")))


def class_prompt_names(classes, overrides):
    """Folder name -> prompt text. CamelCase split by default, e.g. BareLand -> 'bare land'."""
    return [overrides.get(c["name"], re.sub(r"(?<!^)(?=[A-Z])", " ", c["name"]).lower()) for c in classes]


def load_unlabeled(manifest_dir):
    """Split A: image paths only. Guards against caption/label leakage."""
    df = pd.read_csv(os.path.join(manifest_dir, "A_unlabeled.csv"))
    assert list(df.columns) == ["filepath"], f"split A must only contain filepath, got {list(df.columns)}"
    return df


def load_test(manifest_dir):
    return pd.read_csv(os.path.join(manifest_dir, "C_test.csv"))


class RSICDImages(Dataset):
    """Images (+ label if present) from a manifest DataFrame."""

    def __init__(self, df, transform):
        self.df = df
        self.transform = transform
        self.has_label = "label" in df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = self.transform(Image.open(r.filepath).convert("RGB"))
        return (img, int(r.label)) if self.has_label else img


class RSICDPairs(Dataset):
    """One image and one of its five captions.

    Training samples a caption on each access. A fixed caption_index uses that
    caption for every image, so a validation pass stays a one-to-one batch.
    """

    def __init__(self, df, transform, tokenizer, random_caption=True, caption_index=None):
        self.df = df
        self.transform = transform
        self.tokenizer = tokenizer
        self.random_caption = random_caption
        self.caption_index = caption_index

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        if self.caption_index is not None:
            k = self.caption_index
        elif self.random_caption:
            k = torch.randint(5, ()).item()
        else:
            k = 0
        img = self.transform(Image.open(r.filepath).convert("RGB"))
        return img, self.tokenizer([r[CAPTION_COLS[k]]])[0]
