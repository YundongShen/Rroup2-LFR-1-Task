# Bridge experiment protocol

Status: section 1 written 2026-11-10, before any training run. The selection rule for phase 4
(section 2) is fixed by `docs/TASK.md` section 6; the full protocol is completed here before the
first phase-4 run and is not changed after results are seen.

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

## 2. Selection rule for phase 4 (from TASK.md section 6, B labels usable)

- Selection metric: classification accuracy on the B held-out folds (`data=kfold5`).
- Constraint: mean R@1 may drop by at most 0.5 points relative to method 2 (same folds, same seeds).
- Grid (seed 0 coarse screen, then top 3 re-checked with seeds 0–2):
  λ_bridge ∈ {0.25, 0.5, 1.0}; κ ∈ {0, 0.5, 1}; prior ∈ {uniform, caption}; ε_b ∈ {0.5, 1.0}.
- Every reported selection lists accuracy, per-class mean accuracy, and mean R@1 together.

(Remaining details, such as tie-breaking and the τ_text adjustment rule, are added here before
the first phase-4 run.)
