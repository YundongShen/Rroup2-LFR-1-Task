"""Method registry. A method adapts a pretrained CLIP using B-train (pairs) and A (images).

Interface (implement in src/methods/<name>.py and register below):

    class Method:
        trains: bool                       # False => model is not changed, eval can be shared across folds
        def __init__(self, cfg): ...       # full Hydra cfg (method.*, train.*, model.*, ...)
        def fit(self, model, tokenizer, train_tf, eval_tf,
                b_train, b_val, a_unlabeled, device, out_dir) -> model

`b_val` may be used for checkpoint selection (cfg.train.select_metric); split C is never passed in.
"""
from src.methods.rose import ROSE
from src.methods.zeroshot import ZeroShot

METHODS = {"zeroshot": ZeroShot, "rose": ROSE}


def build_method(cfg):
    if cfg.method.name not in METHODS:
        raise KeyError(f"unknown method '{cfg.method.name}', available: {list(METHODS)}")
    return METHODS[cfg.method.name](cfg)
