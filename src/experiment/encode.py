import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import CAPTION_COLS, RSICDImages


def _progress(iterable, total, desc):
    """Show a bar only when there is more than one batch worth of work."""
    if desc is None or total <= 1:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False, dynamic_ncols=True)


@torch.no_grad()
def encode_images(model, df, transform, device, batch_size, num_workers, desc=None):
    """
    Encode the images in the dataframe using the model.
    Parameters:
    - model: the model to use
    - df: the dataframe containing the images
    - transform: the transform to use for the images
    - device: the device to use for the model
    - batch_size: the batch size to use for the model
    - num_workers: the number of workers to use for the data loader
    Returns:
    - The encoded images and labels.
    """
    feats, labels = [], []
    cached = _cached_image_features(model, df)
    if cached is not None:
        features = _project_cached(model, cached, "adapt_image_features", device, batch_size)
        label_values = torch.tensor(df["label"].to_numpy(), dtype=torch.long) if "label" in df.columns else torch.zeros(len(df), dtype=torch.long)
        return features, label_values
    loader = DataLoader(RSICDImages(df, transform), batch_size=batch_size, num_workers=num_workers)
    for img, y in _progress(loader, len(loader), desc):
        feats.append(F.normalize(model.encode_image(img.to(device)).float(), dim=-1).cpu()) # L2 normalize the features
        labels.append(y)
    return torch.cat(feats), torch.cat(labels) # concatenate the features and labels, size is [n_images, d]


@torch.no_grad()
def encode_texts(model, tokenizer, texts, device, batch_size, desc=None):
    """
    Encode the texts using the model.
    Parameters:
    - model: the model to use
    - tokenizer: the tokenizer to tokenize the text inputs
    - texts: the texts to encode
    - device: the device to use for the model
    - batch_size: the batch size to use for the model
    Returns:
    - The encoded texts.
    """
    out = []
    cached = _cached_text_features(model, texts)
    if cached is not None:
        return _project_cached(model, cached, "adapt_text_features", device, batch_size)
    steps = (len(texts) + batch_size - 1) // batch_size if batch_size else 0
    for i in _progress(range(0, len(texts), batch_size), steps, desc):
        tokens = tokenizer(texts[i:i + batch_size]).to(device)
        out.append(F.normalize(model.encode_text(tokens).float(), dim=-1).cpu()) # L2 normalize the features
    return torch.cat(out) # concatenate the features, size is [n_texts, d]


def _cached_image_features(model, df):
    cache = getattr(model, "feature_cache", None)
    paths = df["filepath"].astype(str).tolist()
    if cache is None or not cache.has_images(paths):
        return None
    return cache.images(paths)


def _cached_text_features(model, texts):
    cache = getattr(model, "feature_cache", None)
    texts = [str(text) for text in texts]
    if cache is None or not cache.has_texts(texts):
        return None
    return cache.texts(texts)


def _project_cached(model, features, method, device, batch_size):
    project = getattr(model, method, None)
    chunks = []
    for start in range(0, len(features), batch_size):
        batch = features[start:start + batch_size].to(device)
        if project is not None:
            batch = project(batch)
        chunks.append(batch.detach().cpu())
    return torch.cat(chunks)


def caption_texts(df):
    caps = df[CAPTION_COLS].values.reshape(-1).tolist()
    if len(caps) != 5 * len(df):
        raise ValueError(f"expected 5 captions per image, got {len(caps)} captions for {len(df)} images")
    return caps


def recall_at(ranks, ks):
    if len(ks) == 0:
        raise ValueError("ks must list at least one k")
    return {f"R@{k}": (ranks < k).float().mean().item() for k in ks}


def retrieval_ranks(img_feats, txt_feats):
    """
    Ranks for both directions. txt_feats are 5 captions per image, in row order.
    Parameters:
    - img_feats: the image features
    - txt_feats: the text features
    Returns:
    - The ranks for both directions.
    """
    n = img_feats.shape[0]
    if txt_feats.shape[0] != 5 * n:
        raise ValueError(f"expected {5 * n} caption features, got {txt_feats.shape[0]}")
    sim = img_feats @ txt_feats.T
    txt_owner = torch.arange(n).repeat_interleave(5)
    image_order = sim.argsort(1, descending=True)
    i2t = (txt_owner[image_order] == torch.arange(n)[:, None]).float().argmax(1)
    caption_order = sim.T.argsort(1, descending=True)
    t2i = (caption_order == txt_owner[:, None]).float().argmax(1)
    return i2t, t2i
