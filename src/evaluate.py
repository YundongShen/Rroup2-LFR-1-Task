"""Evaluation tasks: zero-shot classification and image<->text retrieval."""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data import CAPTION_COLS, RSICDImages


@torch.no_grad()
def encode_images(model, df, transform, dev, bs=128, num_workers=4):
    feats, labels = [], []
    for img, y in DataLoader(RSICDImages(df, transform), batch_size=bs, num_workers=num_workers):
        feats.append(F.normalize(model.encode_image(img.to(dev)).float(), dim=-1).cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


@torch.no_grad()
def encode_texts(model, tokenizer, texts, dev, bs=256):
    out = []
    for i in range(0, len(texts), bs):
        out.append(F.normalize(model.encode_text(tokenizer(texts[i:i + bs]).to(dev)).float(), dim=-1).cpu())
    return torch.cat(out)


@torch.no_grad()
def zero_shot(model, tokenizer, img_feats, labels, class_names, templates, dev):
    """Class embedding = normalized mean over templates (single template = plain prompt)."""
    W = []
    for name in class_names:
        e = encode_texts(model, tokenizer, [t.format(name) for t in templates], dev).mean(0)
        W.append(F.normalize(e, dim=-1))
    pred = (img_feats @ torch.stack(W).T).argmax(1)
    present = labels.unique()
    per_cls = torch.stack([(pred[labels == k] == k).float().mean() for k in present])
    return {"zs_acc": (pred == labels).float().mean().item(),
            "zs_mean_per_class_acc": per_cls.mean().item()}


@torch.no_grad()
def retrieval(img_feats, txt_feats, ks=(1, 5, 10)):
    """img_feats [N,d]; txt_feats [5N,d] where captions 5i..5i+4 belong to image i."""
    n = img_feats.shape[0]
    sim = img_feats @ txt_feats.T                        # [N, 5N]
    txt_owner = torch.arange(n).repeat_interleave(5)
    # image->text: rank of best-ranked correct caption
    order = sim.argsort(1, descending=True)
    i2t_rank = (txt_owner[order] == torch.arange(n)[:, None]).float().argmax(1)
    # text->image: rank of the single correct image
    order_t = sim.T.argsort(1, descending=True)
    t2i_rank = (order_t == txt_owner[:, None]).float().argmax(1)
    res = {}
    for k in ks:
        res[f"i2t_R@{k}"] = (i2t_rank < k).float().mean().item()
        res[f"t2i_R@{k}"] = (t2i_rank < k).float().mean().item()
    res["mR"] = sum(res.values()) / len(res)
    return res


def evaluate(model, tokenizer, transform, df, dev, eval_cfg, class_names, templates):
    model.eval()
    img_feats, labels = encode_images(model, df, transform, dev, eval_cfg.batch_size, eval_cfg.num_workers)
    out = {}
    if eval_cfg.tasks.zero_shot.enabled:
        out.update(zero_shot(model, tokenizer, img_feats, labels, class_names, templates, dev))
    if eval_cfg.tasks.retrieval.enabled:
        caps = df[CAPTION_COLS].values.reshape(-1).tolist()   # row-major: image i -> 5i..5i+4
        out.update(retrieval(img_feats, encode_texts(model, tokenizer, caps, dev), eval_cfg.tasks.retrieval.ks))
    return out
