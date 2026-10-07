import json
import os
import re

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
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


def b_folds(manifest_dir, b_split):
    """Divide split B into [(train_df, val_df), ...] according to the `data.b_split` config."""
    df = pd.read_csv(os.path.join(manifest_dir, "B_labeled.csv"))
    y = df["label"].values if b_split.stratify else None
    if b_split.mode == "holdout":
        tr, va = train_test_split(df.index.values, test_size=b_split.val_frac,
                                  random_state=b_split.seed, stratify=y)
        splits = [(tr, va)]
    elif b_split.mode == "kfold":
        kf = (StratifiedKFold if b_split.stratify else KFold)(n_splits=b_split.k, shuffle=True,
                                                               random_state=b_split.seed)
        splits = list(kf.split(df.index.values, y))
    else:
        raise ValueError(f"unknown b_split.mode: {b_split.mode}")
    return [(df.iloc[sorted(tr)].reset_index(drop=True), df.iloc[sorted(va)].reset_index(drop=True))
            for tr, va in splits]


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
    """Image-caption pairs from split B; samples one of the 5 captions per call."""

    def __init__(self, df, transform, tokenizer, random_caption=True):
        self.df = df
        self.transform = transform
        self.tokenizer = tokenizer
        self.random_caption = random_caption

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        k = torch.randint(5, ()).item() if self.random_caption else 0
        img = self.transform(Image.open(r.filepath).convert("RGB"))
        return img, self.tokenizer([r[CAPTION_COLS[k]]])[0]
