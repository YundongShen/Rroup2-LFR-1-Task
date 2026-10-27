"""Evaluation tasks: zero-shot classification, image<->text retrieval, rotation robustness."""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data import CAPTION_COLS, RSICDImages
from src.ops import encode_images_d4


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
def class_embeddings(model, tokenizer, class_names, templates, dev):
    """[C, d]: class embedding = normalized mean over templates (single template = plain prompt)."""
    W = []
    for name in class_names:
        e = encode_texts(model, tokenizer, [t.format(name) for t in templates], dev).mean(0)
        W.append(F.normalize(e, dim=-1))
    return torch.stack(W)


def zs_metrics(pred, labels):
    present = labels.unique()
    per_cls = torch.stack([(pred[labels == k] == k).float().mean() for k in present])
    return {"zs_acc": (pred == labels).float().mean().item(),
            "zs_mean_per_class_acc": per_cls.mean().item()}


@torch.no_grad()
def zero_shot(model, tokenizer, img_feats, labels, class_names, templates, dev):
    W = class_embeddings(model, tokenizer, class_names, templates, dev)
    return zs_metrics((img_feats @ W.T).argmax(1), labels)

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


@torch.no_grad()
def rotation_robustness(model, tokenizer, transform, df, labels, caps, dev, eval_cfg, class_names, templates):
    """Images under all 8 D4 rotations/flips (overhead imagery has no canonical orientation): mean and
    worst case over the group, plus orbit-averaged features (test-time augmentation)."""
    fd = encode_images_d4(model, df, transform, dev, eval_cfg.batch_size, eval_cfg.num_workers)  # [N, 8, d]
    W = class_embeddings(model, tokenizer, class_names, templates, dev)
    txt = encode_texts(model, tokenizer, caps, dev)
    ks = eval_cfg.tasks.retrieval.ks
    acc = [zs_metrics((fd[:, g] @ W.T).argmax(1), labels)["zs_acc"] for g in range(8)]
    mr = [retrieval(fd[:, g], txt, ks)["mR"] for g in range(8)]
    tta = F.normalize(fd.mean(1), dim=-1)
    return {"rot_zs_acc_mean": sum(acc) / 8, "rot_zs_acc_min": min(acc), "rot_mR_mean": sum(mr) / 8,
            "tta_zs_acc": zs_metrics((tta @ W.T).argmax(1), labels)["zs_acc"], "tta_mR": retrieval(tta, txt, ks)["mR"]}


def evaluate(model, tokenizer, transform, df, dev, eval_cfg, class_names, templates):
    model.eval()
    img_feats, labels = encode_images(model, df, transform, dev, eval_cfg.batch_size, eval_cfg.num_workers)
    caps = df[CAPTION_COLS].values.reshape(-1).tolist()   # row-major: image i -> 5i..5i+4
    out = {}
    if eval_cfg.tasks.zero_shot.enabled:
        out.update(zero_shot(model, tokenizer, img_feats, labels, class_names, templates, dev))
    if eval_cfg.tasks.retrieval.enabled:
        out.update(retrieval(img_feats, encode_texts(model, tokenizer, caps, dev), eval_cfg.tasks.retrieval.ks))
    if eval_cfg.tasks.get("rotation", {}).get("enabled", False):
        out.update(rotation_robustness(model, tokenizer, transform, df, labels, caps, dev, eval_cfg,
                                       class_names, templates))
    return out
