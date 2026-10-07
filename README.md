# Rroup2-LFR-1-Task

Semi-supervised CLIP fine-tuning on RSICD (WASP LFR Module 1).

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
python -m src.run -m model=vit_b32_openai,vit_b32_laion2b,vit_b16_laion2b   # sweep
python scripts/collect_results.py [--split val] [--out results.md]         # table of all runs
```


| Group     | Options                                                              | Controls                                                             |
| --------- | -------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `data`    | `holdout`, `kfold5`                                                  | B-train/B-val division: `mode`, `val_frac` / `k`, `stratify`, `seed` |
| `model`   | `vit_{b32,b16,l14}_{openai,laion2b,datacomp}`, `vit_{b16,l14}_dfn2b` | OpenCLIP `arch`, `pretrained`, `precision`                           |
| `prompts` | `course`, `ensemble`                                                 | zero-shot `templates` and `class_names` overrides                    |
| `eval`    | `default`                                                            | `split` (val/test), tasks on/off, retrieval `ks`                     |
| `method`  | `zeroshot`                                                           | adaptation method (registry in `src/methods/`)                       |
| `train`   | `default`                                                            | training hyperparameters (for upcoming methods)                      |


New model: add `configs/model/<name>.yaml` with any pair from `open_clip.list_pretrained()`.
New method: implement the interface in `src/methods/__init__.py`, register it, add `configs/method/<name>.yaml`.

### Split C lock

`eval.split=test` refuses to run unless `eval.confirm_test=true` is also given; every test access is
appended to `runs/test_access.log`. With k folds, each fold's model is evaluated on C and mean±std is
reported (a non-training method is evaluated once).

## Evaluation

- **Zero-shot classification** (31 classes): `zs_acc`, `zs_mean_per_class_acc`; class embedding is the mean over templates.
- **Retrieval**: `i2t_R@k` (hit if any of the image's 5 captions is in top-k), `t2i_R@k`, `mR` = mean of all R@k.

