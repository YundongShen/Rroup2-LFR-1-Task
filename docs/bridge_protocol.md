# Bridge experiment protocol

Status: section 1 written 2026-11-10 before any training run; sections 2–6 written 2026-11-10 after
the phase-3 smoke runs (seed 0, mostly fold 0; listed in `docs/PROGRESS.md`) and before any phase-4
run. Nothing here is changed after phase-4 results are seen.

## 1. Use of B class labels (data-rule interpretation)

The course slides list split B as "images + captions" and do not mention B's scene-class labels.
We interpret this as follows:

- **Allowed:** B class labels of the *held-out* fold may be used for evaluation (classification
  accuracy, per-class accuracy), diagnosis (confusions, prediction histograms, bias metrics), and
  hyperparameter / configuration selection.
- **Never allowed:** B class labels as a training signal in any form: no loss term, no pseudo-label
  initialisation, no class prior computed from labels, no use of labels of the B-train part of a
  fold during training. Class priors used by the bridge (`bridge.prior`) come only from class-name
  text and B-train captions (TASK.md section 5).
- This matches the existing pipeline: stratified 5-fold split of B and `zs_acc` on held-out folds.
- Split A (images only) and split C (final evaluation only, behind `eval.confirm_test`) are
  unaffected.

**Retrieval visibility rule.** Whenever a configuration is selected with a label-based metric, the
mean R@1 (average of image→text and text→image R@1 on the same held-out folds) is reported next to
it, so that any retrieval cost of the selection is visible.

## 2. Fixed settings

- Backbone, adapter, optimiser, T = 500 steps, method-2 pseudo-captions: as in `docs/reproduction.md`.
- Bridge (TASK.md §5): training template "an aerial photo of {}.", evaluation template
  "A satellite image of {}."; λ_bridge ramps linearly over the first 20 % of steps;
  Sinkhorn: scaling-vector form, log domain, float64, ≤ 500 iterations (tol 1e-10 on log β);
  `mix` prior = (1 − γ)·uniform + γ·caption (not in the grid).
- Method-2 reference for the retrieval constraint: `runs/p1/semi_s{0,1,2}` (bridge off is bit-identical).
- B held-out folds (`data=kfold5`), seeds 0–2 on B. C is not touched in phase 4.

## 3. τ_text check (before the grid)

Statistic: mean entropy H(r) over all A images at the first epoch, fold 0, seed 0, default config.
Rule: keep τ_text = 0.01 if 0.05·log C ≤ H(r) ≤ 0.8·log C (log C = log 31 = 3.43); if below, double
τ_text, if above, halve it, until inside; record each step. Already observed in the phase-3 smoke:
H(r) = 0.50 (caption source) → 0.15·log C, inside the band → **τ_text = 0.01 is kept** for all runs,
including `source=image` (H = 0.85 in the smoke).

## 4. Pseudo-label quality on held-out B (descriptive; user request 2026-11-10; before the grid)

For each fold, the held-out B images are treated as unlabeled and pass through exactly the training
pseudo-label pipeline: teacher-adapted features, top-3 captions from the fold's B-train bank
(R = 0.2 z(S_IT) + 0.8 z(S_II), τ = 0.1), r (caption source: r = Σ_j q_uj a_jc; image source:
softmax(cos(z̃ᵀ_u, v_c)/τ_text)), Sinkhorn over the fold's held-out images, ŷ = argmax Q.
Settings: κ ∈ {0, 0.5, 1} × prior ∈ {uniform, caption} × source ∈ {caption, image}; ε_b = 1, τ_text = 0.01.
Teachers: (a) frozen CLIP (= the teacher at the first epoch), (b) the final EMA teacher of the
method-2 runs, seeds 0–2. Reported: pseudo-label accuracy and per-class mean accuracy against the B
labels, on all held-out images and on the coverage-0.5 accepted subset; per-class accuracy; and the
classes that receive zero caption-derived mass (zero hard labels, and Σ_u r_uc share) at κ = 0, on held-out B and
on A (A: hard-label histogram from the κ = 0 training logs; A labels are never read).
B labels are used for evaluation only. **This analysis does not enter any selection.**
Note: Sinkhorn here balances ≈219 held-out images per fold, versus 8,734 A images in training.

## 5. Selection rule (B labels usable for selection, TASK.md §6)

Notation: acc = classification accuracy, mR1 = mean R@1; each value is the mean over the 5 folds
(and over seeds where stated). Eligible = mR1 ≥ mR1(method 2, same seeds) − 0.5.

**Caption-source grid** (30 distinct configs; at κ = 0 the prior has no effect, so it is run once):
λ_bridge ∈ {0.25, 0.5, 1.0} × ε_b ∈ {0.5, 1.0} × [κ = 0, or κ ∈ {0.5, 1} × prior ∈ {uniform, caption}].

1. Coarse screen: every config with seed 0. Rank eligible configs (vs method 2 seed 0, mR1 23.69) by acc.
2. Re-check: the top 3 with seeds 0–2. Eligible vs method 2 over seeds 0–2 (mR1 23.84). The winner is
   the eligible config with the highest 3-seed acc.
3. Ties (acc equal at 0.01 %): higher mR1, then smaller λ_bridge, then larger κ, then uniform before
   caption, then ε_b = 1 before 0.5.
4. If none of the 3 is eligible at step 2, the next eligible configs of the coarse ranking are
   re-checked in order (3 at a time) until one is eligible or the list is exhausted; if exhausted,
   the main method is the TASK default (τ_text 0.01, ε_b 1, κ 1, uniform, λ 0.5) and this is
   reported as a failed selection.

Every reported selection lists acc, per-class mean acc and mR1 together (section 1).

## 6. Experiments (TASK.md §6) and how each is selected

| # | experiment | selection |
|---|---|---|
| 1 | method 2 | none |
| 2 | method 2 + bridge, κ = 0 (no balancing) | rule of §5 restricted to the κ = 0 configs of the caption grid |
| 3 | method 2 + balanced bridge (**main**) | rule of §5 restricted to κ ∈ {0.5, 1} |
| 4 | balanced bridge, `source=image` | same rule on the analogous image-source grid (κ ∈ {0.5, 1}, 24 configs) |
| 5 | main method with `target=soft` | none: the configuration of #3 with soft targets |
| 6 | method 2 + test-time balancing (**transductive**, uses unlabeled evaluation images) | no training; Sinkhorn on r = softmax(s·cos) of method 2's held-out predictions, per fold; grid κ ∈ {0.5, 1} × ε_b ∈ {0.5, 1} × prior ∈ {uniform, caption}, same rule (retrieval unchanged) |

Recorded for each: acc, per-class mean acc, TV(predicted, true class distribution), I→T and T→I
R@1/5/10, mR1, per-class accuracy (from `val_logits_fold*.pt`). #6 is marked as transductive in every table.
Results go to `docs/bridge_results.md`, produced by scripts under `scripts/` from `runs/`.
Phase 5 (C, seeds 0–4) uses the selected configurations only after user approval.

## 7. Phase 5: final evaluation on C (written 2026-11-10, after user approval and before any C run)

Configurations are frozen from `runs/p4/chosen.json`; nothing is changed after C results are seen and C is
run once (`eval.split=test eval.confirm_test=true`, logged in `runs/test_access.log`).

- **Pipeline:** `data=kfold5` as in development: for each seed, the 5 fold models (each trained on that fold's
  B-train part and A) are each evaluated on all of C; the per-seed value is the mean over the 5 fold models.
  Seeds 0–4; mean ± std (population) over seeds. Zero-shot is deterministic and is run once.
- **Rows:** zero-shot; method 1; #1 method 2; #2 caption_l0.25_e1.0_k0; #3 (main)
  caption_l0.25_e1.0_k0.5_uniform; #4 image_l0.25_e0.5_k1_caption; #5 = #3 with target=soft;
  "other prior" = #3 with prior=caption (README row; fixed by construction, not selected);
  #6 † = method 2's C logits + test-time balancing (κ 1, ε_b 1, uniform; Sinkhorn over the 1,093 C images
  for each fold model; transductive, not comparable to other rows).
- **Metrics:** acc, mean per-class acc, TV(predicted, true C class distribution), I→T and T→I R@1/5/10,
  mR1, per-class accuracy (mean over fold models and seeds).
- **Paired test (TASK.md §7), #3 vs #1, same seed:** the 5 per-seed differences of acc, with mean, std,
  two-sided paired t-test p-value (t distribution, 4 d.o.f.) and exact two-sided Wilcoxon signed-rank
  p-value (with n = 5 its minimum is 0.0625, so it cannot reach 0.05). The same is reported for per-class acc
  and mR1. Primary criterion: acc, paired t-test at α = 0.05.
