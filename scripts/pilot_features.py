"""Pilot on frozen CLIP features (no training): is there usable signal in split A?

  1. zero-shot + retrieval on all of B, course vs ensemble prompts, with / without D4 orbit averaging
  2. pseudo-label quality with B as a stand-in unlabeled set (labels only used for scoring): argmax vs Sinkhorn
  3. image-space class prototypes from A (Sinkhorn soft labels) added to the text classifier -> zero-shot on B

    python -m scripts.pilot_features model=vit_b32_laion2b device=cuda
"""
import json
import os

import hydra
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.data import CAPTION_COLS, b_folds, class_prompt_names, load_classes, load_unlabeled
from src.evaluate import class_embeddings, encode_texts, retrieval, zs_metrics
from src.models import build_model, get_device
from src.ops import encode_images_d4, sinkhorn


def precision_at_coverage(scores, labels, fracs=(0.3, 0.5, 1.0)):
    """Pseudo-label accuracy on the most confident fraction of samples."""
    conf, pred = scores.max(1)
    order = conf.argsort(descending=True)
    return {f"acc@{int(100 * f)}%": (pred[order[:int(f * len(order))]] == labels[order[:int(f * len(order))]])
            .float().mean().item() for f in fracs}


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg):
    dev = get_device(cfg.device)
    out_dir = os.path.join("runs", "pilot", cfg.model.name)
    os.makedirs(out_dir, exist_ok=True)
    model, _, eval_tf, tok = build_model(cfg.model, dev)
    model.eval()
    md = cfg.data.manifest_dir
    names = class_prompt_names(load_classes(md), {})
    b = b_folds(md, OmegaConf.create({"mode": "all"}))[0][1]
    yb = torch.tensor(b.label.values)
    fb = encode_images_d4(model, b, eval_tf, dev)                       # [Nb, 8, d]
    fa = encode_images_d4(model, load_unlabeled(md), eval_tf, dev)     # [Na, 8, d]
    caps = encode_texts(model, tok, b[CAPTION_COLS].values.reshape(-1).tolist(), dev)
    Ws = {p: class_embeddings(model, tok, names, list(OmegaConf.load(f"configs/prompts/{p}.yaml").templates), dev)
          for p in ("course", "ensemble")}
    torch.save({"fb": fb.half(), "fa": fa.half(), "yb": yb, "caps": caps.half(), "W": Ws}, f"{out_dir}/feats.pt")

    res = {}
    vb = {"id": fb[:, 0], "d4": F.normalize(fb.mean(1), dim=-1)}
    va = {"id": fa[:, 0], "d4": F.normalize(fa.mean(1), dim=-1)}
    for v, f in vb.items():
        res[f"1.ret/{v}"] = retrieval(f, caps)
        for p, W in Ws.items():
            res[f"1.zs/{p}/{v}"] = zs_metrics((f @ W.T).argmax(1), yb)

    W = Ws["ensemble"]
    for v, f in vb.items():
        s = f @ W.T
        res[f"2.pl/{v}/argmax"] = precision_at_coverage(s.softmax(1), yb)
        for eps in (0.01, 0.02, 0.05):
            res[f"2.pl/{v}/sinkhorn_eps{eps}"] = precision_at_coverage(sinkhorn(s, eps), yb)

    for v in ("id", "d4"):
        sa = va[v] @ W.T
        for lab, Q in {"argmax": F.one_hot(sa.argmax(1), W.shape[0]).float(),
                       "sinkhorn": sinkhorn(sa, 0.02)}.items():
            mu = F.normalize(Q.T @ va[v], dim=-1)                       # image-space class prototypes from A
            for beta in (0.0, 0.5, 1.0, 2.0, 100.0):
                pred = (vb[v] @ W.T + beta * vb[v] @ mu.T).argmax(1)
                res[f"3.proto/{v}/{lab}/beta{beta}"] = zs_metrics(pred, yb)

    json.dump(res, open(f"{out_dir}/pilot.json", "w"), indent=1)
    for k, m in res.items():
        print(f"{k:42s} " + "  ".join(f"{n}={100 * x:.2f}" for n, x in m.items()))


if __name__ == "__main__":
    main()
