"""Run the experiment named in configs/experiment/config.yaml.

    python -m src.experiment.run
    python -m src.experiment.run task=i2t protocol=kfold
    python -m src.experiment.run training=residual task=classification
    python -m src.experiment.run training=semi task=classification
    python -m src.experiment.run training=semi protocol=full seed=0      # stage 6, final numbers on C
    python -m src.experiment.run protocol=full

One run performs one task. With protocol=kfold it evaluates all five folds, training
first when the training config is not `none`. protocol=held-out runs one of those folds.
With protocol=full it evaluates once on C: pretrained weights, or training=semi after
training on A plus B. training=residual does not implement protocol=full. Each round's
train and eval file lists are saved under splits/.
"""
import json
import os
import random
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.data import load_classes
from src.experiment.features import load_aug_cache, load_feature_cache, load_weak_aug_cache
from src.experiment.protocol import iter_splits, write_split
from src.experiment.residual import (
    AdaptedCLIP, SupervisedCLIPTrainer, adapter_state, preprocess_metadata, split_train_val,
)
from src.experiment.semi import SemiSupervisedTrainer
from src.experiment.tasks import TASKS
from src.models import build_model, get_device

ROOT = Path(__file__).resolve().parents[2]
OmegaConf.register_new_resolver("repo_root", lambda: ROOT.as_posix(), replace=True)

TEST_LOG = ROOT / "runs" / "test_access.log"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def training_enabled(cfg):
    return cfg.training is not None and cfg.training.name != "none"


def training_method(cfg):
    """residual: supervised adapters (src/experiment/residual.py). semi: four-loss semi-supervised adapters."""
    return str(cfg.training.get("method", "residual"))


def pretrained_record(model_cfg):
    return {
        "source": "pretrained",
        "name": model_cfg.name,
        "arch": model_cfg.arch,
        "pretrained": model_cfg.pretrained,
        "precision": model_cfg.precision,
    }


def log_test_access(out_dir):
    TEST_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(TEST_LOG, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{out_dir}\t"
                f"{' '.join(HydraConfig.get().overrides.task)}\n")


def summarize(fold_results):
    keys = fold_results[0]["metrics"].keys()
    return {k: {"mean": float(np.mean([r["metrics"][k] for r in fold_results])),
                "std": float(np.std([r["metrics"][k] for r in fold_results]))} for k in keys}


def train_fold(cfg, clip, train_tf, eval_tf, tokenizer, split, device, out_dir, split_dir, cache, aug_cache,
               classes, weak_aug_cache=None):
    """Train one fold from the frozen CLIP initialization and save the adapter checkpoint."""
    fold = split["fold"]
    # The same training seed on every fold. Which images sit in which fold stays with protocol.seed.
    fold_seed = int(cfg.seed)
    set_seed(fold_seed)
    method = training_method(cfg)
    if method == "semi":
        hold_out = float(cfg.training.val_fraction) > 0
    else:
        hold_out = bool(cfg.training.early_stopping)
    if hold_out:
        fit_df, val_df = split_train_val(split["train_b"], float(cfg.training.val_fraction), fold_seed)
    else:
        fit_df, val_df = split["train_b"], None
    prefix = "full" if fold is None else f"fold{fold}"
    fit_path = os.path.join(split_dir, f"{prefix}_fit.csv")
    fit_df[["filepath", "label"]].to_csv(fit_path, index=False)
    extra = {
        "seed": fold_seed,
        "n_fit": len(fit_df),
        "n_val": 0 if val_df is None else len(val_df),
        "fit_files": os.path.relpath(fit_path, out_dir),
    }
    if val_df is not None:
        val_path = os.path.join(split_dir, f"{prefix}_val.csv")
        val_df[["filepath", "label"]].to_csv(val_path, index=False)
        extra["val_files"] = os.path.relpath(val_path, out_dir)

    log_dir = os.path.join(out_dir, "logs")
    if method == "semi":
        train_transform = eval_tf
        adapted = AdaptedCLIP(
            clip,
            hidden_dim=int(cfg.training.hidden_dim),
            alpha=cfg.training.alpha,
            beta=cfg.training.beta,
            adapters=str(cfg.training.adapters),
            learn_logit_scale=bool(cfg.training.losses.con.learn_logit_scale),
        ).to(device)
        adapted.feature_cache = cache
        trainer = SemiSupervisedTrainer(
            adapted, tokenizer, cfg.training, cfg.prompts, classes, device, fold_seed, cache, aug_cache,
            manifest_dir=cfg.manifest_dir, weak_aug_cache=weak_aug_cache)
        trained = trainer.fit(fit_df, val_df, split["train_a"], log_dir, prefix)
    elif method == "residual":
        train_transform = train_tf if cfg.training.use_train_transform else eval_tf
        adapted = AdaptedCLIP(
            clip,
            hidden_dim=int(cfg.training.hidden_dim),
            alpha=cfg.training.alpha,
            beta=cfg.training.beta,
            adapters=str(cfg.training.adapters),
        ).to(device)
        adapted.feature_cache = cache
        train_cache = cache if cache is not None and not cfg.training.use_train_transform else None
        trainer = SupervisedCLIPTrainer(
            adapted, tokenizer, train_transform, eval_tf, cfg.training, device, fold_seed, cfg.num_workers,
            cache=train_cache)
        trained = trainer.fit(fit_df, val_df, log_dir, fold)
    else:
        raise ValueError(f"unknown training.method {method!r}, use residual | semi")

    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{prefix}.pt")
    torch.save({
        **adapter_state(adapted),
        "teacher": trained.get("teacher_state"),
        "metadata": {
            "source": f"{method}_adapter",
            "fold": fold,
            "epoch": trained["epoch"],
            "seed": fold_seed,
            "early_stopped": trained["early_stopped"],
            "val_loss": trained["val_loss"],
            "selection_metric": trained.get("selection_metric"),
            "best_val_metric": trained.get("best_val_metric"),
            "model_name": cfg.model.name,
            "arch": cfg.model.arch,
            "pretrained": cfg.model.pretrained,
            "precision": cfg.model.precision,
            "training": OmegaConf.to_container(cfg.training, resolve=True),
            "prompts": OmegaConf.to_container(cfg.prompts, resolve=True),
            "preprocess": preprocess_metadata(train_transform, eval_tf),
        },
    }, ckpt_path)
    weights = {
        "source": f"{method}_adapter",
        "checkpoint": os.path.relpath(ckpt_path, out_dir),
        "epoch": trained["epoch"],
        "name": cfg.model.name,
        "arch": cfg.model.arch,
        "pretrained": cfg.model.pretrained,
        "precision": cfg.model.precision,
        "adapters": str(cfg.training.adapters),
        "alpha": float(cfg.training.alpha),
        "beta": float(cfg.training.beta),
        "hidden_dim": int(cfg.training.hidden_dim),
    }
    if method == "residual":
        weights["lambda_reg"] = float(cfg.training.lambda_reg)
    else:
        weights["losses"] = {
            key: {"enabled": bool(value.enabled), "weight": float(value.weight)}
            for key, value in cfg.training.losses.items()}
        weights["selection_metric"] = trained.get("selection_metric")
        weights["best_val_metric"] = trained.get("best_val_metric")
        if trained.get("pseudo"):
            extra["pseudo_labels"] = os.path.relpath(os.path.join(log_dir, f"{prefix}_pseudo.json"), out_dir)
            last = trained["pseudo"][-1]
            extra["pseudo_last"] = {
                "epoch": last["epoch"],
                "n_adopted": last["unlabeled"]["n_adopted"],
                "adopt_rate": last["unlabeled"]["adopt_rate"],
                "val_precision": last.get("val", {}).get("precision"),
            }
        if trained.get("pseudo_cap"):
            extra["pseudo_captions"] = os.path.relpath(os.path.join(log_dir, f"{prefix}_pseudo_cap.json"), out_dir)
            last = trained["pseudo_cap"][-1]
            extra["pseudo_cap_last"] = {k: last[k] for k in ("epoch", "n_accepted", "r_threshold", "weight_entropy")}
    extra["history"] = os.path.relpath(os.path.join(log_dir, f"{prefix}.csv"), out_dir)
    extra["curves"] = os.path.relpath(os.path.join(log_dir, f"{prefix}.png"), out_dir)
    return adapted, weights, extra


def evaluate_split(task_name, model, tokenizer, transform, df, device, cfg, classes, progress):
    common = (model, tokenizer, transform, df, device, cfg.batch_size, cfg.num_workers)
    if task_name == "classification":
        return TASKS[task_name](*common, classes, cfg.prompts, progress)
    if task_name in ("i2t", "t2i", "mean_r1"):
        return TASKS[task_name](*common, list(cfg.task.get("ks", [1])), progress)
    raise KeyError(f"unknown task '{task_name}', available: {list(TASKS)}")


@hydra.main(config_path="../../configs/experiment", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    if cfg.task.name not in TASKS:
        raise KeyError(f"unknown task '{cfg.task.name}', available: {list(TASKS)}")
    enabled = training_enabled(cfg)
    if enabled and training_method(cfg) == "residual" and cfg.protocol.name not in ("kfold", "held-out"):
        raise ValueError(
            "supervised residual adaptation trains on Split B with protocol=kfold or protocol=held-out. "
            "protocol=full is not implemented for this method. training=semi supports it: it trains on "
            "A plus all of B (minus the B-val selection split) and evaluates once on C.")

    out_dir = HydraConfig.get().runtime.output_dir
    if cfg.protocol.name == "full":
        log_test_access(out_dir) # full evaluation will touch split C
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    print(OmegaConf.to_yaml(cfg))

    classes = load_classes(cfg.manifest_dir) # load all the classes in the dataset

    # ============================ Build the model ============================
    model, train_tf, eval_tf, tokenizer = build_model(cfg.model, device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    cache = load_feature_cache(cfg.get("feature_cache", None), cfg.model)
    model.feature_cache = cache
    aug_cache = None
    if cache is None:
        print("feature cache: none, CLIP features are computed during the run")
    else:
        print(f"feature cache: {cfg.feature_cache}  "
              f"images={len(cache.image_index)}  captions={len(cache.text_index)}")
        if enabled and training_method(cfg) == "residual" and cfg.training.use_train_transform:
            print("training uses random crops, so it does not read the image cache. "
                  "Set training.use_train_transform=false to train on the cached features.")
    if enabled and training_method(cfg) == "semi":
        aug_cache = load_aug_cache(cfg.get("aug_feature_cache", None), cfg.model)
        if aug_cache is None:
            print("augmented-view cache: none (unlabeled.view=pixel is unavailable)")
        else:
            print(f"augmented-view cache: {cfg.aug_feature_cache}  images={len(aug_cache.image_index)}  "
                  f"views={aug_cache.views}")
        weak_aug_cache = load_weak_aug_cache(cfg.get("weak_aug_feature_cache", None), cfg.model)
        if weak_aug_cache is None:
            print("weak-aug cache: none (losses.cons.enabled=true will raise if used)")
        else:
            print(f"weak-aug cache: {cfg.weak_aug_feature_cache}  images={len(weak_aug_cache.image_index)}  "
                  f"views={weak_aug_cache.views}")
    pretrained = pretrained_record(cfg.model)
    # ============================ Save the weights ============================
    with open(os.path.join(out_dir, "pretrained.json"), "w") as f:
        json.dump(pretrained, f, indent=1)

    split_dir = os.path.join(out_dir, "splits")
    results = []
    for split in iter_splits(cfg.manifest_dir, cfg.protocol):
        train_path, eval_path = write_split(split_dir, split)
        eval_model = model
        weights = pretrained
        extra = {}
        if enabled:
            eval_model, weights, extra = train_fold(
                cfg, model, train_tf, eval_tf, tokenizer, split, device, out_dir, split_dir, cache, aug_cache,
                classes, weak_aug_cache=weak_aug_cache if training_method(cfg) == "semi" else None)
        # ============================ Evaluate the model ============================
        label = "full" if split["fold"] is None else f"fold {split['fold']}"
        metrics = evaluate_split(
            cfg.task.name, eval_model, tokenizer, eval_tf, split["eval"], device, cfg, classes, label)
        record = { # log the results for each fold
            "fold": split["fold"],
            "n_train_a": len(split["train_a"]),
            "n_train_b": len(split["train_b"]),
            "n_eval": len(split["eval"]),
            "train_files": os.path.relpath(train_path, out_dir),
            "eval_files": os.path.relpath(eval_path, out_dir),
            "weights": weights,
            "metrics": metrics,
        }
        record.update(extra)
        results.append(record)
        print(f"[{label}] " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    summary = summarize(results)
    payload = {
        "exp_name": cfg.exp_name,
        "task": cfg.task.name,
        "protocol": cfg.protocol.name,
        "training": None if not enabled else cfg.training.name,
        "n_folds": len(results),
        "summary": summary,
        "folds": results,
    }
    if enabled:
        payload["training_config"] = OmegaConf.to_container(cfg.training, resolve=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(payload, f, indent=1)
    print("[summary] " + "  ".join(f"{k}={v['mean']:.4f}±{v['std']:.4f}" for k, v in summary.items()))
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
