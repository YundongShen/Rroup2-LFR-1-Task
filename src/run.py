"""Experiment entry point.

    python -m src.run                                   # zero-shot, ViT-B-32 laion2b, holdout B-val
    python -m src.run model=vit_l14_openai prompts=ensemble data=kfold5
    python -m src.run eval.split=test eval.confirm_test=true   # final evaluation on split C
    python -m src.run -m model=vit_b32_openai,vit_b32_laion2b  # sweep (Hydra multirun)

For each fold of B: build pretrained model -> method.fit(...) -> evaluate on that fold's
B-val (eval.split=val) or on split C (eval.split=test). Results: <run_dir>/metrics.json.
"""
import json
import os
import random
import time

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.data import b_folds, class_prompt_names, load_classes, load_test, load_unlabeled
from src.evaluate import evaluate
from src.methods import build_method
from src.models import build_model, get_device

TEST_LOG = "runs/test_access.log"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def guard_test(cfg, out_dir):
    if cfg.eval.split not in ("val", "test"):
        raise ValueError(f"eval.split must be 'val' or 'test', got {cfg.eval.split}")
    if cfg.eval.split == "test":
        if not cfg.eval.confirm_test:
            raise RuntimeError("Split C is for final evaluation only. "
                               "Re-run with eval.confirm_test=true if this is the final evaluation.")
        os.makedirs(os.path.dirname(TEST_LOG), exist_ok=True)
        with open(TEST_LOG, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{out_dir}\t"
                    f"{' '.join(HydraConfig.get().overrides.task)}\n")


def summarize(fold_results):
    keys = fold_results[0]["metrics"].keys()
    return {k: {"mean": float(np.mean([r["metrics"][k] for r in fold_results])),
                "std": float(np.std([r["metrics"][k] for r in fold_results]))} for k in keys}


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    out_dir = HydraConfig.get().runtime.output_dir
    guard_test(cfg, out_dir)
    set_seed(cfg.seed)
    dev = get_device(cfg.device)
    print(OmegaConf.to_yaml(cfg))

    md = cfg.data.manifest_dir
    classes = load_classes(md)
    class_names = class_prompt_names(classes, cfg.prompts.class_names)
    templates = list(cfg.prompts.templates)
    a_unlabeled = load_unlabeled(md)
    c_test = load_test(md) if cfg.eval.split == "test" else None
    folds = b_folds(md, cfg.data.b_split)

    method = build_method(cfg)
    split_dir = os.path.join(out_dir, "splits")
    os.makedirs(split_dir, exist_ok=True)

    results, model = [], None
    for i, (b_train, b_val) in enumerate(folds):
        b_train[["filepath"]].to_csv(f"{split_dir}/fold{i}_train.csv", index=False)
        b_val[["filepath"]].to_csv(f"{split_dir}/fold{i}_val.csv", index=False)

        # A non-training method gives identical test results for every fold: evaluate once.
        if c_test is not None and not method.trains and results:
            break
        if model is None or method.trains:
            model, train_tf, eval_tf, tok = build_model(cfg.model, dev)
        fold_dir = os.path.join(out_dir, f"fold{i}")
        model = method.fit(model, tok, train_tf, eval_tf, b_train, b_val, a_unlabeled, dev, fold_dir)

        eval_df = c_test if c_test is not None else b_val
        m = evaluate(model, tok, eval_tf, eval_df, dev, cfg.eval, class_names, templates)
        results.append({"fold": i, "n_b_train": len(b_train), "n_b_val": len(b_val),
                        "eval_split": cfg.eval.split, "n_eval": len(eval_df), "metrics": m})
        print(f"[fold {i}] " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))

    summary = summarize(results)
    json.dump({"exp_name": cfg.exp_name, "eval_split": cfg.eval.split, "n_folds": len(results),
               "summary": summary, "folds": results}, open(os.path.join(out_dir, "metrics.json"), "w"), indent=1)
    print("[summary] " + "  ".join(f"{k}={v['mean']:.4f}±{v['std']:.4f}" for k, v in summary.items()))
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
