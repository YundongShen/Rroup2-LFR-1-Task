"""Supervised residual adapters on frozen OpenCLIP embeddings."""
import csv
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import RSICDPairs
from src.experiment.features import CachedPairs

ADAPTERS = ("both", "image", "text")


class ResidualAdapter(nn.Module):
    """Bottleneck MLP. The second layer starts at zero, so the residual starts at zero."""

    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, features):
        return self.fc2(F.gelu(self.fc1(features)))


class AdaptedCLIP(nn.Module):
    """Frozen OpenCLIP plus optional image and text residual adapters.

    encode_image and encode_text return L2-normalized adapted embeddings.
    forward also returns the original normalized embeddings used by the
    feature-preservation term.
    """

    def __init__(self, clip, hidden_dim, alpha, beta, adapters, learn_logit_scale=False):
        super().__init__()
        if adapters not in ADAPTERS:
            raise ValueError(f"adapters must be one of {ADAPTERS}, got {adapters!r}")
        # Keep the frozen CLIP out of this module so each fold can wrap the same
        # instance, and so adapter-only .parameters() cannot include it.
        object.__setattr__(self, "clip", clip)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.adapters = adapters
        for param in self.clip.parameters():
            param.requires_grad = False
        self.clip.eval()
        dim = int(self.clip.visual.output_dim)
        self.image_adapter = ResidualAdapter(dim, hidden_dim) if adapters in ("both", "image") else None
        self.text_adapter = ResidualAdapter(dim, hidden_dim) if adapters in ("both", "text") else None
        # Temperature. Starts at CLIP's pretrained value either way; only the semi
        # trainer may unfreeze it (training.losses.con.learn_logit_scale).
        init = self.clip.logit_scale.detach().float().clone()
        if learn_logit_scale:
            self.logit_scale = nn.Parameter(init)
        else:
            self.register_buffer("logit_scale", init)

    def scale(self):
        """exp(logit_scale), the scalar temperature used by every loss and by evaluation."""
        return self.logit_scale.exp()

    def image_residual(self, features):
        """z + alpha * h(z) on normalized cached features, before the final normalization."""
        features = features.float()
        if self.image_adapter is None:
            return features
        return features + self.alpha * self.image_adapter(features)

    def text_residual(self, features):
        features = features.float()
        if self.text_adapter is None:
            return features
        return features + self.beta * self.text_adapter(features)

    def train(self, mode=True):
        super().train(mode)
        self.clip.eval()
        return self

    def encode_image(self, images):
        adapted, _ = self._adapt_image(images)
        return adapted

    def encode_text(self, tokens):
        adapted, _ = self._adapt_text(tokens)
        return adapted

    def forward(self, images, tokens):
        image_adapted, image_original = self._adapt_image(images)
        text_adapted, text_original = self._adapt_text(tokens)
        return image_adapted, text_adapted, image_original, text_original

    def forward_cached(self, image_features, text_features):
        """Adapter forward on frozen, already normalized CLIP features."""
        return (
            self.adapt_image_features(image_features),
            self.adapt_text_features(text_features),
            image_features,
            text_features,
        )

    def adapt_image_features(self, features):
        return F.normalize(self.image_residual(features), dim=-1)

    def adapt_text_features(self, features):
        return F.normalize(self.text_residual(features), dim=-1)

    def _adapt_image(self, images):
        with torch.no_grad():
            raw = self.clip.encode_image(images)
        return _residual(raw, self.image_adapter, self.alpha)

    def _adapt_text(self, tokens):
        with torch.no_grad():
            raw = self.clip.encode_text(tokens)
        return _residual(raw, self.text_adapter, self.beta)


def _residual(raw, adapter, scale):
    original = F.normalize(raw.float(), dim=-1)
    if adapter is None:
        return original, original
    delta = adapter(original)
    adapted = F.normalize(original + scale * delta, dim=-1)
    return adapted, original


def batch_losses(image_adapted, text_adapted, image_original, text_original, logit_scale, lambda_reg):
    """Symmetric CLIP loss plus mean squared shift from the original normalized features."""
    scale = logit_scale.detach().float().exp()
    logits = scale * (image_adapted @ text_adapted.T)
    target = torch.arange(logits.shape[0], device=logits.device)
    loss_i2t = F.cross_entropy(logits, target)
    loss_t2i = F.cross_entropy(logits.T, target)
    loss_clip = 0.5 * (loss_i2t + loss_t2i)
    squared = (image_adapted - image_original).pow(2).sum(-1) + (text_adapted - text_original).pow(2).sum(-1)
    loss_reg = 0.5 * squared.mean()
    loss_total = loss_clip + lambda_reg * loss_reg
    stats = {
        "loss_total": loss_total.detach(),
        "loss_clip": loss_clip.detach(),
        "loss_i2t": loss_i2t.detach(),
        "loss_t2i": loss_t2i.detach(),
        "loss_reg": loss_reg.detach(),
        "image_feature_shift": (image_adapted - image_original).norm(dim=-1).mean().detach(),
        "text_feature_shift": (text_adapted - text_original).norm(dim=-1).mean().detach(),
    }
    return loss_total, stats


def lr_multiplier(step, total_steps, warmup_steps, min_ratio):
    """Linear warmup to 1, then cosine decay to min_ratio. `step` is zero-based."""
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    last = total_steps - 1
    step = min(max(step, 0), last)
    warmup_steps = min(max(int(warmup_steps), 0), total_steps)
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    span = total_steps - warmup_steps
    progress = 1.0 if span <= 1 else (step - warmup_steps) / (span - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def split_train_val(df, val_fraction, seed):
    """Stratified image-level split of the four training folds. Val rows are not for gradients."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")
    labels = df["label"].to_numpy()
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    fit_idx, val_idx = next(splitter.split(np.zeros(len(df)), labels))
    fit = df.iloc[np.sort(fit_idx)].reset_index(drop=True)
    val = df.iloc[np.sort(val_idx)].reset_index(drop=True)
    return fit, val


def adapter_state(adapted):
    def dump(module):
        if module is None:
            return None
        return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}

    return {
        "image_adapter": dump(adapted.image_adapter),
        "text_adapter": dump(adapted.text_adapter),
        "logit_scale": adapted.logit_scale.detach().cpu().clone(),
    }


def load_adapter_state(adapted, state):
    if state["image_adapter"] is not None:
        adapted.image_adapter.load_state_dict(state["image_adapter"])
    if state["text_adapter"] is not None:
        adapted.text_adapter.load_state_dict(state["text_adapter"])
    if state.get("logit_scale") is not None:
        with torch.no_grad():
            adapted.logit_scale.copy_(state["logit_scale"].to(adapted.logit_scale.device))


def preprocess_metadata(train_transform, eval_transform):
    mean, std = _normalize_stats(eval_transform)
    return {
        "image_size": _image_size(eval_transform),
        "mean": mean,
        "std": std,
        "train": str(train_transform),
        "eval": str(eval_transform),
    }


def _iter_transforms(transform):
    children = getattr(transform, "transforms", None)
    if children is None:
        yield transform
        return
    for child in children:
        yield from _iter_transforms(child)


def _normalize_stats(transform):
    for child in _iter_transforms(transform):
        if child.__class__.__name__ == "Normalize":
            return [float(value) for value in child.mean], [float(value) for value in child.std]
    return None, None


def _image_size(transform):
    for child in _iter_transforms(transform):
        if child.__class__.__name__ in ("CenterCrop", "Resize", "RandomResizedCrop"):
            size = child.size
            if isinstance(size, (tuple, list)):
                return int(size[0])
            return int(size)
    return None


class SupervisedCLIPTrainer:
    """AdamW on the adapters only, with linear warmup and cosine decay."""

    def __init__(self, adapted, tokenizer, train_transform, eval_transform, train_cfg, device, seed, num_workers,
                 cache=None):
        self.adapted = adapted
        self.tokenizer = tokenizer
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.cache = cache
        self.device = device
        self.seed = int(seed)
        self.num_workers = int(num_workers)
        self.batch_size = int(train_cfg.batch_size)
        self.lr = float(train_cfg.lr)
        self.min_lr = float(train_cfg.min_lr)
        self.weight_decay = float(train_cfg.weight_decay)
        self.max_epochs = int(train_cfg.max_epochs)
        self.warmup_epochs = int(train_cfg.warmup_epochs)
        self.log_every_n_epochs = max(1, int(train_cfg.get("log_every_n_epochs", 1)))
        self.lambda_reg = float(train_cfg.lambda_reg)
        self.early_stopping = bool(train_cfg.early_stopping)
        self.patience = int(train_cfg.patience)
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.max_epochs < 1:
            raise ValueError(f"max_epochs must be positive, got {self.max_epochs}")
        if self.warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be non-negative, got {self.warmup_epochs}")

    def fit(self, train_df, val_df, log_dir, fold):
        os.makedirs(log_dir, exist_ok=True)
        train_loader = self._loader(train_df, train=True, caption_index=None)
        steps_per_epoch = len(train_loader)
        if steps_per_epoch == 0:
            raise ValueError("training split has no batches")
        params = [param for param in self.adapted.parameters() if param.requires_grad]
        if not params:
            raise ValueError("no trainable adapter parameters")
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        total_steps = steps_per_epoch * self.max_epochs
        warmup_steps = steps_per_epoch * self.warmup_epochs
        min_ratio = self.min_lr / self.lr

        rows = []
        best_loss = math.inf
        best_epoch = None
        best_state = None
        wait = 0
        stopped = False
        for epoch in range(1, self.max_epochs + 1):
            train_stats = self._train_epoch(
                train_loader, optimizer, params, epoch, steps_per_epoch, total_steps, warmup_steps, min_ratio,
                f"fold {fold} epoch {epoch}/{self.max_epochs}")
            row = {"epoch": epoch, "lr": train_stats.pop("lr")}
            row.update({f"train/{key}": value for key, value in train_stats.items()})
            message = f"[fold {fold}] epoch {epoch}/{self.max_epochs}  lr={row['lr']:.3e}  " + _format_stats(train_stats, "train")
            if val_df is not None:
                val_stats = self._validate(val_df)
                row["val/loss_total"] = val_stats["loss_total"]
                message += f"  val/loss_total={val_stats['loss_total']:.4f}"
                if math.isfinite(val_stats["loss_total"]) and val_stats["loss_total"] < best_loss:
                    best_loss = val_stats["loss_total"]
                    best_epoch = epoch
                    best_state = adapter_state(self.adapted)
                    wait = 0
                else:
                    wait += 1
                    if self.early_stopping and wait >= self.patience:
                        stopped = True
            rows.append(row)
            _write_history(os.path.join(log_dir, f"fold{fold}.csv"), rows)
            if (epoch == 1 or epoch == self.max_epochs or stopped
                    or epoch % self.log_every_n_epochs == 0):
                print(message)
            if stopped:
                break

        if best_state is not None:
            load_adapter_state(self.adapted, best_state)
            if best_epoch != rows[-1]["epoch"]:
                why = "early stop" if stopped else "max epochs reached"
                print(f"[fold {fold}] {why} at epoch {rows[-1]['epoch']}, "
                      f"restore epoch {best_epoch} (val/loss_total={best_loss:.4f})")
        elif best_epoch is None:
            best_epoch = rows[-1]["epoch"]
        _save_curves(os.path.join(log_dir, f"fold{fold}.png"), rows)
        self.adapted.eval()
        return {
            "epoch": best_epoch,
            "val_loss": None if best_state is None else best_loss,
            "early_stopped": stopped,
            "history": rows,
        }

    def _train_epoch(self, loader, optimizer, params, epoch, steps_per_epoch, total_steps, warmup_steps, min_ratio, desc):
        self.adapted.train()
        totals = {}
        count = 0
        grad_sum = 0.0
        grad_steps = 0
        last_lr = self.lr
        lr_sum = 0.0
        steps = tqdm(loader, desc=desc, unit="step", leave=False, dynamic_ncols=True)
        for index, (images, tokens) in enumerate(steps):
            global_step = (epoch - 1) * steps_per_epoch + index
            last_lr = self.lr * lr_multiplier(global_step, total_steps, warmup_steps, min_ratio)
            lr_sum += last_lr
            for group in optimizer.param_groups:
                group["lr"] = last_lr
            images = images.to(self.device)
            tokens = tokens.to(self.device)
            optimizer.zero_grad(set_to_none=True)
            if self.cache is None:
                adapted_i, adapted_t, original_i, original_t = self.adapted(images, tokens)
            else:
                adapted_i, adapted_t, original_i, original_t = self.adapted.forward_cached(images, tokens)
            loss, stats = batch_losses(
                adapted_i, adapted_t, original_i, original_t, self.adapted.clip.logit_scale, self.lambda_reg)
            loss.backward()
            grad_sum += _grad_norm(params)
            grad_steps += 1
            optimizer.step()
            batch = images.shape[0]
            count += batch
            for key, value in stats.items():
                totals[key] = totals.get(key, 0.0) + float(value) * batch
        averaged = {key: value / count for key, value in totals.items()}
        averaged["gradient_norm"] = grad_sum / grad_steps
        averaged["lr"] = lr_sum / grad_steps
        return averaged

    def _validate(self, val_df):
        self.adapted.eval()
        totals = {}
        count = 0
        for caption_index in range(5):
            loader = self._loader(val_df, train=False, caption_index=caption_index)
            for images, tokens in loader:
                images = images.to(self.device)
                tokens = tokens.to(self.device)
                with torch.no_grad():
                    if self.cache is None:
                        adapted_i, adapted_t, original_i, original_t = self.adapted(images, tokens)
                    else:
                        adapted_i, adapted_t, original_i, original_t = self.adapted.forward_cached(images, tokens)
                    _, stats = batch_losses(
                        adapted_i, adapted_t, original_i, original_t, self.adapted.clip.logit_scale, self.lambda_reg)
                batch = images.shape[0]
                count += batch
                for key, value in stats.items():
                    totals[key] = totals.get(key, 0.0) + float(value) * batch
        self.adapted.train()
        return {key: value / count for key, value in totals.items()}

    def _loader(self, df, train, caption_index):
        # Validation always uses the eval transform, so it can read the cache.
        # Training can read it only when its transform is that same eval transform.
        use_cache = self.cache is not None and (not train or self.train_transform is self.eval_transform)
        if use_cache:
            dataset = CachedPairs(
                df, self.cache, random_caption=caption_index is None, caption_index=caption_index)
            generator = None
            if train:
                generator = torch.Generator()
                generator.manual_seed(self.seed)
            return DataLoader(
                dataset, batch_size=self.batch_size, shuffle=train, num_workers=0,
                drop_last=False, generator=generator)
        dataset = RSICDPairs(
            df,
            self.train_transform if train else self.eval_transform,
            self.tokenizer,
            random_caption=caption_index is None,
            caption_index=caption_index,
        )
        generator = None
        worker_init = None
        if train:
            generator = torch.Generator()
            generator.manual_seed(self.seed)
            worker_init = _seed_worker
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=self.num_workers,
            drop_last=False,
            generator=generator,
            worker_init_fn=worker_init,
            pin_memory=str(self.device).startswith("cuda"),
        )


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def _grad_norm(params):
    total = 0.0
    for param in params:
        if param.grad is None:
            continue
        total += float(param.grad.detach().float().pow(2).sum())
    return math.sqrt(total)


def _format_stats(stats, prefix):
    order = ("loss_total", "loss_clip", "loss_i2t", "loss_t2i", "loss_reg",
             "image_feature_shift", "text_feature_shift", "gradient_norm")
    return "  ".join(f"{prefix}/{key}={stats[key]:.4f}" for key in order if key in stats)


HISTORY_FIELDS = [
    "epoch", "lr",
    "train/loss_total", "train/loss_clip", "train/loss_i2t", "train/loss_t2i", "train/loss_reg",
    "train/image_feature_shift", "train/text_feature_shift", "train/gradient_norm",
    "val/loss_total",
]


def _write_history(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in HISTORY_FIELDS})


def _save_curves(path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in rows]
    figure, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    loss_keys = ("train/loss_total", "train/loss_clip", "train/loss_reg", "val/loss_total")
    for key in loss_keys:
        values = [row.get(key) for row in rows]
        if any(value != "" and value is not None for value in values):
            axes[0].plot(epochs, [None if value == "" else value for value in values], label=key)
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, [row["train/image_feature_shift"] for row in rows], label="train/image_feature_shift")
    axes[1].plot(epochs, [row["train/text_feature_shift"] for row in rows], label="train/text_feature_shift")
    axes[1].set_ylabel("L2 shift")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(epochs, [row["train/gradient_norm"] for row in rows], label="train/gradient_norm")
    axes[2].set_ylabel("grad norm")
    axes[2].set_xlabel("epoch")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
