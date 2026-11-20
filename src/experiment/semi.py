"""Semi-supervised residual adapter on frozen OpenCLIP features.

    L = L_con + lambda_cls L_cls + lambda_cons L_cons + lambda_anc L_anc + lambda_pcap L_pcap

L_pcap (losses.pseudo_cap, src/experiment/pseudo_caption.py) is the soft caption
pseudo-supervision on Split A: each epoch the teacher retrieves, for every A image,
its top-k B-train captions by the fused image-text / image-image score, and the
student fits a weighted cross-entropy over the whole B-train caption bank on the
accepted images.

Both encoders stay frozen. The student is the residual adapter inside an
AdaptedCLIP; the teacher is an EMA copy of it; the anchor is the frozen feature
itself. Everything is computed on features cached by scripts/cache_features.py,
so one epoch takes seconds and the labeled batch can be large.

Data in one fold:
    fit_df        B-train  image + 5 caption features (+ class label when use_b_labels)
    val_df        B-val    model selection only; the pseudo-label rule is also scored here
    unlabeled_df  A        image features and K augmented views; no caption, no label
Split C is never read here. The trainer asserts that none of its file paths
belongs to C_test.csv.
"""
import csv
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

from src.augment import feature_perturbation
from src.data import CAPTION_COLS, class_prompt_names, load_test
from src.experiment import pseudo as pseudo_labels
from src.experiment import pseudo_caption
from src.experiment.ema import EMATeacher
from src.experiment.evaluate import evaluate_features, scalar_metrics
from src.experiment.losses import (
    anchor_loss, clip_loss, cls_logits, cls_logits_per_image_template, consistency_loss, image_nce_loss, ramp_up,
)
from src.experiment.residual import adapter_state, load_adapter_state, lr_multiplier

VIEWS = ("pixel", "dropout", "noise", "none")
CLASS_PROMPT_MODES = ("fixed", "random")
PROMPT_BANK = Path(__file__).resolve().parents[2] / "configs" / "prompts" / "multiple.yaml"


def assert_no_test_leak(manifest_dir, *frames):
    """Fail if any training or selection frame contains an image of Split C."""
    test_paths = set(load_test(manifest_dir)["filepath"].astype(str))
    for frame in frames:
        if frame is None:
            continue
        leaked = set(frame["filepath"].astype(str)) & test_paths
        assert not leaked, f"{len(leaked)} Split C images reached the trainer, for example {sorted(leaked)[0]}"


@torch.no_grad()
def encode_prototypes(clip, tokenizer, classes, prompts, device):
    """P_c: frozen text features of the class prompts, averaged over templates, L2-normalized. [C, d]."""
    names = class_prompt_names(classes, prompts.class_names)
    templates = list(prompts.templates)
    weight = []
    for name in names:
        tokens = tokenizer([t.format(name) for t in templates]).to(device)
        embedding = F.normalize(clip.encode_text(tokens).float(), dim=-1).mean(0)
        weight.append(F.normalize(embedding, dim=-1))
    return torch.stack(weight).to(device)


@torch.no_grad()
def encode_prompt_bank(clip, tokenizer, classes, prompts, device):
    """One normalized text feature per class and template. [C, K, d], not averaged."""
    names = class_prompt_names(classes, prompts.class_names)
    templates = list(prompts.templates)
    rows = []
    for name in names:
        tokens = tokenizer([t.format(name) for t in templates]).to(device)
        rows.append(F.normalize(clip.encode_text(tokens).float(), dim=-1))
    return torch.stack(rows).to(device)


class SemiSupervisedTrainer:
    def __init__(self, adapted, tokenizer, train_cfg, prompts_cfg, classes, device, seed, cache, aug_cache=None,
                 manifest_dir=None, weak_aug_cache=None):
        if cache is None:
            raise ValueError(
                "training=semi needs the frozen feature cache. Run scripts/cache_features.py for this model first.")
        self.adapted = adapted
        self.clip = adapted.clip
        self.tokenizer = tokenizer
        self.cfg = train_cfg
        self.prompts = prompts_cfg
        self.classes = classes
        self.device = device
        self.seed = int(seed)
        self.cache = cache
        self.aug_cache = aug_cache
        self.weak_aug_cache = weak_aug_cache
        self.manifest_dir = manifest_dir

        self.batch_size = int(train_cfg.batch_size)
        self.lr = float(train_cfg.lr)
        self.min_lr = float(train_cfg.min_lr)
        self.weight_decay = float(train_cfg.weight_decay)
        self.max_epochs = int(train_cfg.max_epochs)
        self.warmup_epochs = int(train_cfg.warmup_epochs)
        self.log_every_n_epochs = max(1, int(train_cfg.get("log_every_n_epochs", 1)))
        self.selection_metric = str(train_cfg.selection_metric)
        self.early_stopping = bool(train_cfg.early_stopping)
        self.patience = int(train_cfg.patience)
        if self.lr <= 0 or self.max_epochs < 1 or self.warmup_epochs < 0:
            raise ValueError("lr must be positive, max_epochs >= 1, warmup_epochs >= 0")

        self.l = train_cfg.losses
        self.con = self.l.con
        self.cls = self.l.cls
        self.cons = self.l.cons
        self.anc = self.l.anc
        self.unl = train_cfg.unlabeled
        self.pcap = self.l.get("pseudo_cap", None)
        self.use_pseudo = bool(self.cls.enabled and self.cls.pseudo.enabled)
        self.use_pcap = self.pcap is not None and bool(self.pcap.enabled)
        self.use_unlabeled = self.use_pseudo or self.use_pcap or bool(self.cons.enabled) or (
            bool(self.anc.enabled) and str(self.anc.apply_to) in ("unlabeled", "both"))
        self.needs_teacher = (self.use_pseudo and str(self.cls.pseudo.source) == "teacher") or (
            self.use_pcap and str(self.pcap.source) == "teacher")
        if self.use_pcap:
            if str(self.pcap.source) not in ("clip", "teacher"):
                raise ValueError(f"pseudo_cap.source must be clip | teacher, got {self.pcap.source!r}")
            if str(self.pcap.input) not in ("original", "view"):
                raise ValueError(f"pseudo_cap.input must be original | view, got {self.pcap.input!r}")
            if not 0.0 < float(self.pcap.coverage) <= 1.0:
                raise ValueError("pseudo_cap.coverage must be in (0, 1]")
        if str(self.unl.view) not in VIEWS:
            raise ValueError(f"unlabeled.view must be one of {VIEWS}, got {self.unl.view!r}")
        self.class_prompt_mode = str(self.unl.get("class_prompt_mode", "fixed"))
        if self.class_prompt_mode not in CLASS_PROMPT_MODES:
            raise ValueError(
                f"unlabeled.class_prompt_mode must be one of {CLASS_PROMPT_MODES}, got {self.class_prompt_mode!r}")
        if self.use_unlabeled and str(self.unl.view) == "pixel" and aug_cache is None:
            raise ValueError(
                "unlabeled.view=pixel needs the augmented-view cache (scripts/cache_features.py --aug-views K). "
                "Use unlabeled.view=dropout|noise|none to train without it.")
        if bool(self.cons.enabled) and weak_aug_cache is None:
            raise ValueError(
                "losses.cons.enabled=true needs the weak-augmented-view cache. "
                "Run: scripts/cache_features.py --weak-views 4  "
                "or set losses.cons.enabled=false to skip the image-image contrastive term.")
        if not any(bool(x.enabled) for x in (self.con, self.cls, self.cons, self.anc)) and not self.use_pcap:
            raise ValueError("every loss is disabled; enable at least one of losses.{con,cls,cons,anc,pseudo_cap}")

        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(self.seed)
        self.prototypes0 = self._encode_prototypes()  # frozen P_c, [C, d], normalized; teacher, B, and eval
        self.prompt_bank = self._encode_prompt_bank() if self.class_prompt_mode == "random" else None
        self.num_classes = self.prototypes0.shape[0]
        self.scale0 = self.clip.logit_scale.detach().float().exp().to(device)

    # ------------------------------------------------------------------ setup
    def _encode_prototypes(self):
        return encode_prototypes(self.clip, self.tokenizer, self.classes, self.prompts, self.device)

    def _encode_prompt_bank(self):
        bank_prompts = OmegaConf.load(PROMPT_BANK)
        bank = encode_prompt_bank(self.clip, self.tokenizer, self.classes, bank_prompts, self.device)
        if bank.shape[0] != self.prototypes0.shape[0]:
            raise ValueError(
                f"prompt bank has {bank.shape[0]} classes, single prototypes have {self.prototypes0.shape[0]}")
        return bank

    def _random_template_logits(self, img, student, scale):
        """one template index per image, shared by all classes. The bank stays frozen."""
        indices = torch.randint(self.prompt_bank.shape[1], (img.shape[0],), generator=self.generator).to(img.device)
        if student.text_adapter is None:
            return cls_logits_per_image_template(img, self.prompt_bank, indices, scale)
        picked = self.prompt_bank[:, indices]
        classes, batch, dim = picked.shape
        picked = student.adapt_text_features(picked.reshape(classes * batch, dim)).view(classes, batch, dim)
        return scale * torch.einsum("bd,cbd->bc", F.normalize(img, dim=-1), F.normalize(picked, dim=-1))

    def _image_feats(self, df):
        return self.cache.images(df["filepath"].astype(str).tolist()).to(self.device)

    def _caption_feats(self, df):
        texts = [str(value) for row in df[CAPTION_COLS].itertuples(index=False) for value in row]
        return self.cache.texts(texts).view(len(df), 5, -1).to(self.device)

    def _labels(self, df):
        return torch.tensor(df["label"].to_numpy(), dtype=torch.long, device=self.device)

    def _prototypes(self, model):
        """Prototypes as the (student or teacher) model scores them: through its text adapter if it has one."""
        if model.text_adapter is None:
            return self.prototypes0
        return model.adapt_text_features(self.prototypes0)

    # ------------------------------------------------------------------ fit
    def fit(self, fit_df, val_df, unlabeled_df, log_dir, tag):
        """`tag` names the round's files: logs/<tag>.csv, <tag>_pseudo.json, <tag>.png."""
        os.makedirs(log_dir, exist_ok=True)
        if self.manifest_dir is not None:
            assert_no_test_leak(self.manifest_dir, fit_df, val_df, unlabeled_df)
        if self.cls.enabled and self.cls.use_b_labels and "label" not in fit_df.columns:
            raise ValueError("losses.cls.use_b_labels=true but the labeled frame has no label column")
        if self.use_unlabeled and unlabeled_df is not None:
            assert list(unlabeled_df.columns) == ["filepath"], "Split A frame must only carry file paths"

        # Labeled tensors
        Lf, Lt = self._image_feats(fit_df), self._caption_feats(fit_df)
        Ly = self._labels(fit_df) if "label" in fit_df.columns else None
        n_fit = Lf.shape[0]
        # Validation tensors
        has_val = val_df is not None and len(val_df) > 0
        if has_val:
            Vf, Vt, Vy = self._image_feats(val_df), self._caption_feats(val_df), self._labels(val_df)
        # Unlabeled tensors
        Uf = Ua = None
        if self.use_unlabeled:
            if unlabeled_df is None or len(unlabeled_df) == 0:
                raise ValueError("this configuration uses Split A but no unlabeled frame was given")
            Uf = self._image_feats(unlabeled_df)
            if str(self.unl.view) == "pixel":
                paths = unlabeled_df["filepath"].astype(str).tolist()
                if not self.aug_cache.has_images(paths):
                    raise KeyError("some Split A images have no augmented views in the cache")
                Ua = self.aug_cache.images(paths).to(self.device)  # [m, K, d]
        # Weak-augmented views for image-image contrastive (L_cons)
        weak_Ua = None
        if bool(self.cons.enabled) and self.weak_aug_cache is not None:
            if unlabeled_df is None or len(unlabeled_df) == 0:
                pass  # guard already raised above in use_unlabeled path
            else:
                wpaths = unlabeled_df["filepath"].astype(str).tolist()
                if not self.weak_aug_cache.has_images(wpaths):
                    raise KeyError("some Split A images have no weak-augmented views in the cache")
                weak_Ua = self.weak_aug_cache.images(wpaths).to(self.device)  # [m, K_w, d]
        p_clip_U = None if Uf is None else F.softmax(cls_logits(Uf, self.prototypes0, self.scale0), dim=-1)
        p_clip_L = F.softmax(cls_logits(Lf, self.prototypes0, self.scale0), dim=-1)
        caption_weak = None
        if self.cls.enabled and bool(self.cls.get("caption_labels", {}).get("enabled", False)):
            if self.cls.use_b_labels:
                raise ValueError("losses.cls.use_b_labels and losses.cls.caption_labels are exclusive")
            caption_weak = self._caption_weak_labels(Lt, Ly)
            with open(os.path.join(log_dir, f"{tag}_caption_labels.json"), "w") as handle:
                json.dump(caption_weak["stats"], handle, indent=1)
            print(f"[{tag}] caption-derived labels on B-train: " + ", ".join(
                f"{k}={v}" for k, v in caption_weak["stats"].items() if not isinstance(v, list)))

        # B-train caption bank for the soft caption pseudo-supervision (unique strings, frozen features)
        bank = None
        if self.use_pcap:
            captions, sources = pseudo_caption.build_caption_bank(fit_df, CAPTION_COLS)
            bank = {"captions": captions, "sources": sources, "txt": self.cache.texts(captions).to(self.device)}
            print(f"[{tag}] caption bank: {len(captions)} unique captions of {n_fit} B-train images")

        teacher = EMATeacher(self.adapted, float(self.cfg.ema.momentum)) if self.needs_teacher else None

        params = [p for p in self.adapted.parameters() if p.requires_grad]
        if not params:
            raise ValueError("no trainable adapter parameters")
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        steps_per_epoch = math.ceil(n_fit / self.batch_size)
        total_steps = steps_per_epoch * self.max_epochs
        warmup_steps = steps_per_epoch * self.warmup_epochs
        min_ratio = self.min_lr / self.lr
        u_batch = int(self.unl.batch_size)
        warmup_unlabeled = int(round(float(self.unl.get("warmup_fraction", 0.0)) * total_steps))
        u_perm, u_ptr = None, 0

        rows, pseudo_log, pcap_log = [], [], []
        best_metric, best_epoch, best_state, wait, stopped = -math.inf, None, None, 0, False
        assignment = cap_assignment = None
        step = 0
        for epoch in range(1, self.max_epochs + 1):
            # ---- caption candidates for this epoch (teacher or frozen CLIP, never the student)
            if self.use_pcap and (cap_assignment is None or str(self.pcap.source) == "teacher"):
                cap_assignment, stats = self._assign_captions(Uf, Lf, bank, teacher)
                stats["epoch"] = epoch
                pcap_log.append(stats)
                with open(os.path.join(log_dir, f"{tag}_pseudo_cap.json"), "w") as handle:
                    json.dump(pcap_log, handle, indent=1)
                if epoch == 1 or epoch % 10 == 0:
                    self._dump_caption_examples(os.path.join(log_dir, f"{tag}_pseudo_cap_examples.json"),
                                                unlabeled_df, bank, cap_assignment, epoch)
            # ---- pseudo-labels for this epoch (from the frozen CLIP or the EMA teacher, never the student)
            if self.use_pseudo and (assignment is None or str(self.cls.pseudo.source) == "teacher"):
                assignment, stats = self._assign_pseudo(Uf, p_clip_U, teacher, Vf if has_val else None,
                                                        Vy if has_val else None)
                stats["epoch"] = epoch
                pseudo_log.append(stats)
                with open(os.path.join(log_dir, f"{tag}_pseudo.json"), "w") as handle:
                    json.dump(pseudo_log, handle, indent=1)

            # ---- one epoch over B-train
            self.adapted.train()
            perm = torch.randperm(n_fit, generator=self.generator)
            totals, count = {}, 0
            bar = tqdm(range(steps_per_epoch), desc=f"{tag} epoch {epoch}/{self.max_epochs}", unit="step",
                       leave=False, dynamic_ncols=True)
            lr_sum = 0.0
            for s in bar:
                lr_now = self.lr * lr_multiplier(step, total_steps, warmup_steps, min_ratio)
                lr_sum += lr_now
                for group in optimizer.param_groups:
                    group["lr"] = lr_now
                idx = perm[s * self.batch_size:(s + 1) * self.batch_size].to(self.device)
                uidx = None
                # Split A enters only after `unlabeled.warmup_fraction` of all steps, so the teacher
                # that labels it is already a B-trained adapter and not the zero-shot model.
                if self.use_unlabeled and step >= warmup_unlabeled:
                    if u_perm is None or u_ptr + u_batch > len(u_perm):
                        u_perm = torch.randperm(Uf.shape[0], generator=self.generator)
                        u_ptr = 0
                    uidx = u_perm[u_ptr:u_ptr + u_batch].to(self.device)
                    u_ptr += u_batch
                loss, stats = self._step_losses(
                    idx, Lf, Lt, Ly, p_clip_L, uidx, Uf, Ua, p_clip_U, assignment, teacher, step, total_steps,
                    caption_weak, weak_Ua, bank, cap_assignment)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if teacher is not None:
                    teacher.update(self.adapted)
                step += 1
                b = idx.shape[0]
                count += b
                for key, value in stats.items():
                    totals[key] = totals.get(key, 0.0) + float(value) * b
            row = {"epoch": epoch, "lr": lr_sum / steps_per_epoch, "logit_scale": float(self.adapted.scale())}
            row.update({f"train/{k}": v / count for k, v in totals.items()})
            if self.use_pseudo:
                row["pseudo/n_adopted"] = pseudo_log[-1]["unlabeled"]["n_adopted"]
                row["pseudo/val_precision"] = pseudo_log[-1].get("val", {}).get("precision", "")
            if self.use_pcap:
                row["pseudo_cap/n_accepted"] = pcap_log[-1]["n_accepted"]
                row["pseudo_cap/r_threshold"] = pcap_log[-1]["r_threshold"]
                row["pseudo_cap/weight_entropy"] = pcap_log[-1]["weight_entropy"]

            # ---- B-val metrics (selection) for the student, plus the teacher's accuracy for the log
            message = f"[{tag}] epoch {epoch}/{self.max_epochs}  lr={row['lr']:.2e}  " + \
                "  ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in row.items() if k.startswith("train/"))
            if has_val:
                val = scalar_metrics(self._evaluate(self.adapted, Vf, Vt, Vy))
                row.update({f"val/{k}": v for k, v in val.items()})
                if teacher is not None:
                    row["val/teacher_cls_acc"] = self._evaluate(teacher.model, Vf, None, Vy)["cls_acc"]
                message += f"  val/cls_acc={val['cls_acc']:.4f}  val/i2t_r1={val['i2t_r1']:.4f}  val/t2i_r1={val['t2i_r1']:.4f}"
                current = val[self.selection_metric]
                if current > best_metric:
                    best_metric, best_epoch, best_state, wait = current, epoch, adapter_state(self.adapted), 0
                else:
                    wait += 1
                    if self.early_stopping and wait >= self.patience:
                        stopped = True
            rows.append(row)
            _write_history(os.path.join(log_dir, f"{tag}.csv"), rows)
            if (epoch == 1 or epoch == self.max_epochs or stopped
                    or epoch % self.log_every_n_epochs == 0):
                print(message)
            if stopped:
                break

        if best_state is not None:
            load_adapter_state(self.adapted, best_state)
            if best_epoch != rows[-1]["epoch"]:
                why = "early stop" if stopped else "max epochs reached"
                print(f"[{tag}] {why} at epoch {rows[-1]['epoch']}, restore epoch {best_epoch} "
                      f"(val/{self.selection_metric}={best_metric:.4f})")
        else:
            best_epoch = rows[-1]["epoch"]
        _save_curves(os.path.join(log_dir, f"{tag}.png"), rows, self.selection_metric)
        self.adapted.eval()
        return {
            "epoch": best_epoch,
            "selection_metric": self.selection_metric,
            "best_val_metric": None if best_state is None else best_metric,
            "val_loss": None,
            "early_stopped": stopped,
            "history": rows,
            "pseudo": pseudo_log,
            "pseudo_cap": pcap_log,
            "teacher_state": None if teacher is None else {k: v.cpu() for k, v in teacher.state_dict().items()},
        }

    # ------------------------------------------------------------------ pieces
    @torch.no_grad()
    def _caption_weak_labels(self, Lt, Ly):
        """Class labels for B-train read off its captions, which the course allows.

        p(c | image) = mean over the 5 captions of softmax(s * z_T . P_c), i.e. the frozen text
        encoder matching each caption against the class prompts; argmax is the weak label and the
        confidence must pass `caption_labels.threshold`. The true labels, when the frame has them,
        are used only to report the precision of this rule, never in the loss.
        """
        cfg = self.cls.caption_labels
        probs = F.softmax(cls_logits(Lt.flatten(0, 1), self.prototypes0, self.scale0), dim=-1)
        probs = probs.view(Lt.shape[0], Lt.shape[1], -1).mean(1)
        conf, labels = probs.max(dim=1)
        mask = conf >= float(cfg.get("threshold", 0.0))
        stats = {"n_total": int(mask.numel()), "n_adopted": int(mask.sum()),
                 "per_class_count": torch.bincount(labels[mask], minlength=self.num_classes).tolist()}
        if Ly is not None:
            correct = (labels == Ly) & mask
            stats["precision"] = float(correct.sum() / mask.sum().clamp_min(1))
            stats["coverage_x_precision"] = float(correct.sum() / mask.numel())
        return {"labels": labels, "mask": mask, "stats": stats}

    @torch.no_grad()
    def _source_probs(self, feats, p_clip, teacher):
        source = str(self.cls.pseudo.source)
        if source == "clip" or teacher is None:
            if source == "teacher":
                print("pseudo.source=teacher but no EMA teacher is active; using the frozen CLIP")
            return p_clip
        if source != "teacher":
            raise ValueError(f"pseudo.source must be clip | teacher, got {source!r}")
        model = teacher.model
        logits = cls_logits(model.adapt_image_features(feats), self._prototypes(model), model.scale())
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def _assign_pseudo(self, Uf, p_clip_U, teacher, Vf, Vy):
        probs = self._source_probs(Uf, p_clip_U, teacher)
        assignment = pseudo_labels.assign(
            probs, str(self.cls.pseudo.strategy), float(self.cls.pseudo.threshold), int(self.cls.pseudo.min_per_class))
        stats = {"unlabeled": pseudo_labels.summarize(assignment, num_classes=self.num_classes)}
        if Vf is not None:
            p_clip_V = F.softmax(cls_logits(Vf, self.prototypes0, self.scale0), dim=-1)
            val_probs = self._source_probs(Vf, p_clip_V, teacher)
            val_assignment = pseudo_labels.apply_thresholds(val_probs, assignment["thresholds"])
            stats["val"] = pseudo_labels.summarize(val_assignment, Vy, self.num_classes)
        return assignment, stats

    @torch.no_grad()
    def _assign_captions(self, Uf, Lf, bank, teacher):
        """Top-k B-train captions per A image by the fused score, soft weights, and the acceptance mask."""
        if str(self.pcap.source) == "teacher" and teacher is not None:
            model = teacher.model
            q_img, b_img = model.adapt_image_features(Uf), model.adapt_image_features(Lf)
            b_txt = model.adapt_text_features(bank["txt"]) if model.text_adapter is not None else bank["txt"]
        else:
            q_img, b_img, b_txt = Uf, Lf, bank["txt"]
        R, _, _ = pseudo_caption.fused_scores(q_img, b_img, b_txt, bank["sources"], float(self.pcap.lam))
        assignment, stats = pseudo_caption.assign_captions(
            R, int(self.pcap.top_k), float(self.pcap.tau_p), float(self.pcap.coverage))
        return assignment, stats

    def _dump_caption_examples(self, path, unlabeled_df, bank, assignment, epoch, n=20):
        """A few accepted A images with their weighted candidates, for a manual look."""
        if unlabeled_df is None:
            return
        accepted = assignment["mask"].nonzero().flatten()
        picked = accepted[torch.randperm(len(accepted), generator=self.generator)[:n]].tolist()
        paths = unlabeled_df["filepath"].astype(str).tolist()
        examples = [{
            "image": paths[u],
            "r_max": round(float(assignment["r_max"][u]), 3),
            "candidates": [{"caption": bank["captions"][j], "weight": round(float(w), 3)}
                           for j, w in zip(assignment["candidates"][u].tolist(), assignment["weights"][u].tolist())],
        } for u in picked]
        with open(path, "w") as handle:
            json.dump({"epoch": epoch, "examples": examples}, handle, indent=1)

    def _unlabeled_view(self, uidx, Uf, Ua):
        view = str(self.unl.view)
        if view == "pixel":
            k = torch.randint(Ua.shape[1], (uidx.shape[0],), generator=self.generator).to(self.device)
            return Ua[uidx, k]
        return F.normalize(feature_perturbation(
            Uf[uidx], view, float(self.unl.dropout_p), float(self.unl.noise_sigma)), dim=-1)

    def _step_losses(self, idx, Lf, Lt, Ly, p_clip_L, uidx, Uf, Ua, p_clip_U, assignment, teacher, step, total_steps,
                     caption_weak=None, weak_Ua=None, bank=None, cap_assignment=None):
        student = self.adapted
        scale = student.scale()
        P = self._prototypes(student)
        zero = Lf.new_zeros(())
        stats = {}

        # labeled
        zL = Lf[idx]
        k = torch.randint(5, (idx.shape[0],), generator=self.generator).to(self.device)
        tL = Lt[idx, k]
        rawL = student.image_residual(zL)
        aL = F.normalize(rawL, dim=-1)
        aT = student.adapt_text_features(tL) if student.text_adapter is not None else tL

        # unlabeled: original and perturbed view
        zU = rawU = aU_view = logitsU_view = None
        if uidx is not None:
            zU = Uf[uidx]
            rawU = student.image_residual(zU)
            vU = self._unlabeled_view(uidx, Uf, Ua)
            aU_view = F.normalize(student.image_residual(vU), dim=-1)
            logitsU_view = cls_logits(aU_view, P, scale)

        total = zero
        if self.con.enabled:
            l_con, parts = clip_loss(aL, aT, scale, img_ids=idx)
            total = total + float(self.con.weight) * l_con
            stats["loss_con"] = l_con.detach()
            stats.update(parts)

        if self.cls.enabled:
            logits, labels, mask = [], [], []
            if self.cls.use_b_labels and Ly is not None:
                logits.append(cls_logits(aL, P, scale))
                labels.append(Ly[idx])
                mask.append(torch.ones(idx.shape[0], dtype=torch.bool, device=self.device))
            elif caption_weak is not None:
                logits.append(cls_logits(aL, P, scale))
                labels.append(caption_weak["labels"][idx])
                mask.append(caption_weak["mask"][idx])
            if self.use_pseudo and uidx is not None:
                inp = aU_view if str(self.cls.pseudo.input) == "view" else F.normalize(rawU, dim=-1)
                if self.class_prompt_mode == "random":
                    logits.append(self._random_template_logits(inp, student, scale))
                else:
                    logits.append(logitsU_view if inp is aU_view else cls_logits(inp, P, scale))
                labels.append(assignment["labels"][uidx])
                mask.append(assignment["mask"][uidx])
            if logits:
                logits, labels, mask = torch.cat(logits), torch.cat(labels), torch.cat(mask)
                l_cls = F.cross_entropy(logits[mask], labels[mask]) if bool(mask.any()) else zero
                total = total + float(self.cls.weight) * l_cls
                stats["loss_cls"] = l_cls.detach()
                stats["cls_rows"] = mask.float().sum()

        if self.cons.enabled and uidx is not None and weak_Ua is not None:
            # Image-image contrastive on Split A (standard SSL / SimCLR-style InfoNCE).
            # Two weakly-augmented views of the same image are positive pairs; all other
            # images in the unlabeled batch are negatives.  Both views pass through the
            # student adapter — symmetric, no stop-gradient.
            # Weak views (scale 0.9-1.0, hflip only) stay close to the eval center-crop
            # in feature space, so gradients update the adapter in the region that matters
            # at evaluation time, not towards a view that may show a different land-cover patch.
            m, K_w = weak_Ua.shape[0], weak_Ua.shape[1]
            k1 = torch.randint(K_w, (uidx.shape[0],), generator=self.generator).to(self.device)
            k2 = (k1 + 1 + torch.randint(max(1, K_w - 1), (uidx.shape[0],),
                                          generator=self.generator).to(self.device)) % K_w
            v1 = weak_Ua[uidx, k1]          # [u, d]
            v2 = weak_Ua[uidx, k2]          # [u, d]
            a1 = F.normalize(student.image_residual(v1), dim=-1)
            a2 = F.normalize(student.image_residual(v2), dim=-1)
            l_cons = image_nce_loss(a1, a2, scale)
            warm = int(round(float(self.unl.get("warmup_fraction", 0.0)) * total_steps))
            ramp = ramp_up(step - warm, total_steps - warm, float(self.cons.rampup_fraction))
            total = total + float(self.cons.weight) * ramp * l_cons
            stats["loss_cons"] = l_cons.detach()
            stats["cons_ramp"] = torch.tensor(ramp)

        if self.use_pcap and uidx is not None and cap_assignment is not None:
            inp = aU_view if str(self.pcap.input) == "view" else F.normalize(rawU, dim=-1)
            b_txt = student.adapt_text_features(bank["txt"]) if student.text_adapter is not None else bank["txt"]
            l_pcap = pseudo_caption.pseudo_caption_loss(
                inp, b_txt, cap_assignment["candidates"][uidx], cap_assignment["weights"][uidx],
                cap_assignment["mask"][uidx], scale)
            warm = int(round(float(self.unl.get("warmup_fraction", 0.0)) * total_steps))
            ramp = ramp_up(step - warm, total_steps - warm, float(self.pcap.rampup_fraction))
            total = total + float(self.pcap.weight) * ramp * l_pcap
            stats["loss_pcap"] = l_pcap.detach()
            stats["pcap_rows"] = cap_assignment["mask"][uidx].float().sum()
            stats["pcap_ramp"] = torch.tensor(ramp)

        if self.anc.enabled:
            kind, on = str(self.anc.kind), str(self.anc.apply_to)
            terms = []
            if on in ("labeled", "both"):
                terms.append(anchor_loss(rawL, zL, kind, P, scale, p_clip_L[idx]))
            if on in ("unlabeled", "both") and uidx is not None:
                terms.append(anchor_loss(rawU, zU, kind, P, scale, p_clip_U[uidx]))
            if terms:
                l_anc = torch.stack(terms).mean()
                total = total + float(self.anc.weight) * l_anc
                stats["loss_anc"] = l_anc.detach()

        stats["loss_total"] = total.detach()
        stats["image_feature_shift"] = (aL - zL).norm(dim=-1).mean().detach()
        return total, stats

    @torch.no_grad()
    def _evaluate(self, model, img, txt, labels):
        model.eval()
        a_img = model.adapt_image_features(img)
        a_txt = None
        if txt is not None:
            a_txt = model.adapt_text_features(txt.flatten(0, 1)) if model.text_adapter is not None else txt
        out = evaluate_features(a_img, a_txt, self._prototypes(model), labels)
        if model is self.adapted:
            model.train()
        return out


# ---------------------------------------------------------------------- logging
def _write_history(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _save_curves(path, rows, selection_metric):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [r["epoch"] for r in rows]
    panels = [
        ("loss", [k for k in rows[-1] if k.startswith("train/loss")]),
        ("B-val", [f"val/{m}" for m in ("cls_acc", "mean_r1", "i2t_r1", "t2i_r1", "teacher_cls_acc") if f"val/{m}" in rows[-1]]),
        ("pseudo", [k for k in ("pseudo/n_adopted", "pseudo/val_precision") if k in rows[-1]]),
        ("pseudo_cap", [k for k in ("pseudo_cap/n_accepted", "pseudo_cap/weight_entropy") if k in rows[-1]]),
    ]
    panels = [p for p in panels if p[1]]
    figure, axes = plt.subplots(len(panels), 1, figsize=(8, 3 * len(panels)), sharex=True, squeeze=False)
    for ax, (title, keys) in zip(axes[:, 0], panels):
        for key in keys:
            values = [r.get(key, None) for r in rows]
            values = [None if v == "" else v for v in values]
            if key in ("pseudo/n_adopted", "pseudo_cap/n_accepted"):
                ax2 = ax.twinx()
                ax2.plot(epochs, values, color="gray", alpha=0.6, label=key)
                ax2.set_ylabel("n images")
                continue
            ax.plot(epochs, values, label=key + (" (selection)" if key == f"val/{selection_metric}" else ""))
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    axes[-1, 0].set_xlabel("epoch")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
