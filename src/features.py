"""Cache of frozen CLIP features for training adapters on top of a frozen backbone.

Image features: z = normalize(model.encode_image(eval_tf(img))), keyed by file path.
Text features:  t = normalize(model.encode_text(caption)), keyed by caption text.
Split A is cached by file path only; the path is used as an opaque key, never parsed.
Split C is never cached here (final evaluation goes through src.evaluate).
"""
import os

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data import CAPTION_COLS, RSICDImages
from src.evaluate import encode_texts


@torch.no_grad()
def _encode_paths(model, paths, transform, dev, bs, num_workers):
    df = pd.DataFrame({"filepath": paths})          # no label column -> dataset yields images only
    out = []
    for img in DataLoader(RSICDImages(df, transform), batch_size=bs, num_workers=num_workers):
        out.append(F.normalize(model.encode_image(img.to(dev)).float(), dim=-1).cpu())
    return torch.cat(out)


def _atomic_save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    torch.save(obj, tmp)
    os.replace(tmp, path)


class FeatureCache:
    """Loads or builds the image/text feature cache for one model on splits A and B."""

    def __init__(self, cache_dir, model_name):
        self.path = os.path.join(cache_dir, f"features_{model_name}.pt")

    def load_or_build(self, model, tokenizer, transform, a_df, b_df, dev, bs=256, num_workers=8):
        img_paths = list(a_df["filepath"]) + list(b_df["filepath"])
        texts = sorted(set(b_df[CAPTION_COLS].values.reshape(-1).tolist()))
        if os.path.exists(self.path):
            c = torch.load(self.path, weights_only=True)
            if set(img_paths) <= set(c["img_paths"]) and set(texts) <= set(c["texts"]):
                print(f"[cache] loaded {self.path}")
                return self._index(c)
            print(f"[cache] {self.path} incomplete, rebuilding")
        model.eval()
        print(f"[cache] encoding {len(img_paths)} images and {len(texts)} texts")
        c = {"img_paths": img_paths,
             "img": _encode_paths(model, img_paths, transform, dev, bs, num_workers),
             "texts": texts,
             "txt": encode_texts(model, tokenizer, texts, dev)}
        _atomic_save(c, self.path)
        print(f"[cache] saved {self.path}")
        return self._index(c)

    @staticmethod
    def _index(c):
        c["img_idx"] = {p: i for i, p in enumerate(c["img_paths"])}
        c["txt_idx"] = {t: i for i, t in enumerate(c["texts"])}
        return c

    @staticmethod
    def images(c, paths):
        return c["img"][[c["img_idx"][p] for p in paths]]

    @staticmethod
    def texts(c, texts):
        return c["txt"][[c["txt_idx"][t] for t in texts]]
