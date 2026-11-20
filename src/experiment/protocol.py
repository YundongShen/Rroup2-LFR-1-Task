"""Shared train/eval division for one experiment.

kfold: every round uses all of A plus 4 folds of B to train, and the held-out fold of B to evaluate.
held-out: the same division as kfold, but only the selected fold.
full: one round, train on all of A and B, evaluate on C.
"""
import os

import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from src.data import load_test, load_unlabeled


def iter_splits(manifest_dir, protocol):
    """
    Iterate over the splits for the given protocol.
    Parameters:
    - manifest_dir: the directory containing the manifest files
    - protocol: the protocol to use
    Returns:
    - A generator of dictionaries, each containing the split information for one fold.
    """
    unlabeled = load_unlabeled(manifest_dir) # load the unlabeled data, split A
    labeled = pd.read_csv(os.path.join(manifest_dir, "B_labeled.csv")) # load the labeled data, split B
    if protocol.name in ("kfold", "held-out"):
        folds = list(_b_folds(labeled, protocol))
        if protocol.name == "held-out":
            chosen = int(protocol.fold)
            if chosen < 0 or chosen >= len(folds):
                raise ValueError(f"held-out fold must be in 0..{len(folds) - 1}, got {chosen}")
            folds = [folds[chosen]]
        for i, (train_idx, eval_idx) in folds:
            yield _b_split(unlabeled, labeled, i, train_idx, eval_idx)
    elif protocol.name == "full":
        yield {
            "fold": None,
            "train_a": unlabeled,
            "train_b": labeled,
            "eval": load_test(manifest_dir),
        }
    else:
        raise ValueError(f"unknown protocol '{protocol.name}'")


def _b_folds(labeled, protocol):
    """Image-level folds of Split B. The same k, stratify, and seed always give the same folds."""
    y = labeled["label"].values if protocol.stratify else None
    folder = (StratifiedKFold if protocol.stratify else KFold)(
        n_splits=int(protocol.k), shuffle=True, random_state=int(protocol.seed))
    return list(enumerate(folder.split(labeled.index.values, y)))


def _b_split(unlabeled, labeled, fold, train_idx, eval_idx):
    return {
        "fold": fold,
        "train_a": unlabeled,
        "train_b": labeled.iloc[sorted(train_idx)].reset_index(drop=True),
        "eval": labeled.iloc[sorted(eval_idx)].reset_index(drop=True),
    }


def write_split(split_dir, split):
    """Save the exact file lists this round trains on and evaluates on."""
    os.makedirs(split_dir, exist_ok=True)
    prefix = "full" if split["fold"] is None else f"fold{split['fold']}"
    train = pd.concat([
        split["train_a"].assign(source="A")[["filepath", "source"]],
        split["train_b"].assign(source="B")[["filepath", "source"]],
    ], ignore_index=True)
    eval_source = "C" if split["fold"] is None else "B"
    eval_df = split["eval"].assign(source=eval_source)[["filepath", "source"]]
    train_path = os.path.join(split_dir, f"{prefix}_train.csv")
    eval_path = os.path.join(split_dir, f"{prefix}_eval.csv")
    train.to_csv(train_path, index=False)
    eval_df.to_csv(eval_path, index=False)
    return train_path, eval_path
