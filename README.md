# Rroup2-LFR-1-Task

Semi-supervised CLIP fine-tuning on RSICD (WASP LFR Module 1).

## A-only milestone: ROSE — results for discussion

**ROSE** (Rotation-Orbit Self-training with Equipartitioned pseudo-labels) adapts CLIP using **split A images only**.
No captions or labels are used; the only text is the 31 class names in the course prompt.
Every epoch the current model relabels A:
- image features, averaged over the 8 rotations/flips, are matched to the class prompts;
- Sinkhorn balances the assignment over classes;
- the image tower is trained on the resulting hard labels, with the text tower frozen.

Full details, sensitivity analysis and failure cases are in [docs/a_only_rose.md](docs/a_only_rose.md).

**Setup.** Evaluated on **all of split B**; nothing from B is trained on.
ViT-B-32 `laion2b_s34b_b79k`, prompt "A satellite image of {}.", 5 epochs.
Trained methods report mean ± std over 3 seeds.

| Method | zs acc | mean per-class acc | i2t R@1 | t2i R@1 | mR |
|---|---|---|---|---|---|
| Zero-shot CLIP (baseline) | 63.62 | 63.51 | 3.75 | 3.78 | 12.39 |
| Vanilla self-training (argmax pseudo-labels) | 68.68 ± 1.98 | 68.89 ± 2.07 | 3.63 ± 0.11 | 4.27 ± 0.13 | 13.59 ± 0.11 |
| **ROSE** | **78.31 ± 0.82** | **80.03 ± 0.87** | **4.39 ± 0.27** | **5.00 ± 0.12** | **14.49 ± 0.16** |

| Ablation (one component removed, 3 seeds) | zs acc | mR |
|---|---|---|
| ROSE | 78.31 ± 0.82 | 14.49 ± 0.16 |
| − class balancing (argmax labels) | 71.39 ± 1.06 | 14.42 ± 0.09 |
| − per-epoch relabelling | 73.00 ± 0.09 | 14.20 ± 0.28 |
| − rotation orbit | 77.45 ± 0.65 | 14.41 ± 0.16 |
| − relational anchor | 78.28 ± 0.64 | 14.34 ± 0.08 |
| soft instead of hard labels (seed 0) | 33.82 (collapse) | 5.08 |

**Other backbone.** ViT-B-16, seed 0, same hyperparameters, zs acc / mR:

| Method | zs acc | mR |
|---|---|---|
| Zero-shot | 66.73 | 12.03 |
| Vanilla self-training | 64.81 | 12.87 |
| ROSE | 76.69 | 14.55 |

**Points for discussion**

1. **Why it works.** Zero-shot CLIP's errors on RSICD are mostly class bias.
   - It never predicts *center*, rarely predicts *meadow* or *bare land*, and over-predicts *stadium* (9.4% of predictions vs 2.7% true share).
   - Plain self-training amplifies the bias: +5 ± 2 on B-32 and −1.9 on B-16.
   - Balancing the pseudo-labels fixes it: *center* 0 → 73%, *meadow* 29 → 100%, *bare land* 16 → 74%.
2. **What matters.** Class balancing (+6.9) and relabelling (+5.3).
   - The rotation orbit adds +0.9.
   - The anchor leaves accuracy unchanged; it trades i2t R@1 (−0.6) for t2i R@1 (+1.1).
   - Soft labels collapse: pseudo-label confidence falls 0.41 → 0.05.
   - Learning curves: [docs/figs/curves.png](docs/figs/curves.png).
3. **Failure cases.** Near-synonymous classes stay unsolved ([docs/figs/confusions.png](docs/figs/confusions.png)):
   - *playfields* 0% (→ *playground*, *baseball field*, *stadium*)
   - *sparse residential* 0% (→ *medium residential*)
   - *commercial* 9% (→ *square*)
4. **Retrieval** improves only modestly (+2.1 mR). Only class names are used, with no caption-level text; this is what split B captions can add in the next phase.
5. **Caveat.** In this phase B is the evaluation set, and the Sinkhorn temperature was chosen on it. It is sensitive: 0.05 drops zs acc to 70.1. For the final report: select hyperparameters on part of B and report once on C.

```bash
python -m src.run data=allb model=vit_b32_laion2b device=cuda                    # zero-shot baseline on all of B
python -m src.run method=rose train=aonly data=allb model=vit_b32_laion2b device=cuda seed=0
python scripts/summarize_aonly.py                                                # mean ± std over seeds
```

## Setup

```bash
conda create -n lfr-clip python=3.11 && conda activate lfr-clip
pip install -r requirements.txt
```

Raw data (from [RSICD_optimal](https://github.com/201528014227051/RSICD_optimal)) goes into `data/raw/`:
`RSICD_images/`, `dataset_rsicd.json`, `txtclasses_rsicd/`.

```bash
python scripts/prepare_data.py       # -> data/splits/{A_unlabeled,B_labeled,C_test}.csv, classes.json
python scripts/download_weights.py   # optional: pre-cache OpenCLIP weights
```


| Split | Manifest          | Content                   | Use                                                  |
| ----- | ----------------- | ------------------------- | ---------------------------------------------------- |
| A     | `A_unlabeled.csv` | image paths only          | unlabeled training images                            |
| B     | `B_labeled.csv`   | images, class, 5 captions | divided per run into B-train / B-val (`data` config) |
| C     | `C_test.csv`      | images, class, 5 captions | final evaluation only (locked, see below)            |




## Running experiments

All settings live in `configs/` (Hydra-style). Each run writes to `runs/<exp_name>/<timestamp>/`
(`.hydra/config.yaml`, `splits/` snapshot of the B division, `metrics.json`).

```bash
python -m src.run                                        # defaults: zero-shot, ViT-B-32 laion2b, holdout
python -m src.run model=vit_l14_openai prompts=ensemble  # swap model / prompt set
python -m src.run data=kfold5                            # 5-fold over B, reports mean±std
python -m src.run data.b_split.val_frac=0.3 data.b_split.seed=1
python scripts/collect_results.py [--split val] [--out results.md]         # table of all runs
```


| Group     | Options                                                              | Controls                                                             |
| --------- | -------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `data`    | `holdout`, `kfold5`, `allb`                                          | B-train/B-val division: `mode`, `val_frac` / `k`, `stratify`, `seed`; `allb`: evaluate on all of B (A-only methods) |
| `model`   | `vit_{b32,b16,l14}_{openai,laion2b,datacomp}`, `vit_{b16,l14}_dfn2b` | OpenCLIP `arch`, `pretrained`, `precision`                           |
| `prompts` | `course`, `ensemble`                                                 | zero-shot `templates` and `class_names` overrides                    |
| `eval`    | `default`                                                            | `split` (val/test), tasks on/off, retrieval `ks`, `tasks.rotation` |
| `method`  | `zeroshot`, `rose`                                                   | adaptation method (registry in `src/methods/`)                       |
| `train`   | `default`, `aonly`                                                   | training hyperparameters; `aonly`: training on split A only          |


New model: add `configs/model/<name>.yaml` with any pair from `open_clip.list_pretrained()`.
New method: implement the interface in `src/methods/__init__.py`, register it, add `configs/method/<name>.yaml`.

### Split C lock

`eval.split=test` refuses to run unless `eval.confirm_test=true` is also given; every test access is
appended to `runs/test_access.log`. With k folds, each fold's model is evaluated on C and mean±std is
reported (a non-training method is evaluated once).

## Evaluation

- **Zero-shot classification** (31 classes): `zs_acc`, `zs_mean_per_class_acc`; class embedding is the mean over templates.
- **Retrieval**: `i2t_R@k` (hit if any of the image's 5 captions is in top-k), `t2i_R@k`, `mR` = mean of all R@k.
- **Rotation robustness** (`eval.tasks.rotation.enabled=true`): all 8 rotations/flips of the eval images —
  `rot_zs_acc_mean`/`rot_zs_acc_min`, `rot_mR_mean`, and orbit-averaged test-time augmentation `tta_zs_acc`, `tta_mR`.
