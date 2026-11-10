# Reproduction of method 1 (supervised) and method 2 (semi-supervised)

Code: `src/methods/semi.py` (method), `src/features.py` (frozen-feature cache),
`configs/method/semi.yaml` (all hyperparameters), tests in `tests/test_semi.py`.
Method 2 = `method=semi`; method 1 = `method=semi method.pcap.enabled=false`.

## 1. Implementation

- **Backbone:** OpenCLIP ViT-B/32 `laion2b_s34b_b79k`, fp32, image and text towers frozen.
- **Adapter:** z̃ = z + α h(z), α = 0.2, where z = L2-normalised frozen CLIP image feature.
  h = Linear(512, 512) → GELU → Linear(512, 512), last layer zero-initialised (identity at start).
  All cosines use L2-normalised z̃.
- **Feature cache:** z for all A and B images (eval transform, no augmentation) and t for all B
  caption texts are encoded once and cached in `runs/cache/features_<model>.pt`. A paths are used
  only as opaque keys. C is never cached.
- **L_con (B-train):** each step draws 128 B-train images (shuffled passes, cycled) and one of their
  5 caption slots uniformly; logits s·cos(z̃_i, t_j) with s = CLIP's own `logit_scale.exp()` (frozen).
  If the caption text drawn for k also belongs to image i (k ≠ i), that entry is masked in both
  directions (duplicate captions are common in RSICD).
- **Caption bank:** unique caption texts of the current fold's B-train part (exact string match);
  src(j) = B-train images owning text j.
- **Teacher assignment (every epoch):** EMA teacher adapter on all A and B-train features;
  S_IT = cos(z̃ᵀ_u, t_j), S_II = max_{i∈src(j)} cos(z̃ᵀ_u, z̃ᵀ_i); per-image z-score over the bank
  (population std); R = 0.2 z(S_IT) + 0.8 z(S_II); top-3 candidates with w = softmax(R/0.1) inside
  the candidates; the 50 % of A images with the highest max_j R are accepted for this epoch.
- **L_pcap:** soft cross-entropy −Σ_{j∈N_u} w_uj log softmax_{j'}(s·cos(z̃_u, t_j')) over the whole bank.
- **Schedule:** one epoch = one pass over the accepted A images in batches of 128
  (⌈4367/128⌉ = 35 steps, last batch 15); the teacher re-assigns at each epoch start. λ_pcap rises
  linearly from 0 to 1 over the first 20 % of steps.
- **Method 1:** identical code path without A, the teacher, and L_pcap; same adapter, optimiser,
  B batches, and total number of steps.
- **Evaluation:** after training, the student adapter is wrapped into `model.encode_image`
  (`wrap_encode_image`) and the repository's standard `evaluate` runs on the B held-out fold.
  The same metrics are also computed from the cached features; `run.py` stores both and their
  maximum absolute difference (`cached_max_abs_diff` in `metrics.json`).
- **Seeds:** per fold the adapter initialisation and all sampling use seed·1000 + fold.

## 2. Defaults not specified in the README (from docs/TASK.md, phase 1)

| item | value |
|---|---|
| h | 2-layer MLP 512→512→512, GELU, last layer zero-init |
| optimiser | AdamW, lr 1e-3, weight decay 1e-4, cosine decay to 0, no warm-up |
| EMA momentum | 0.999, updated every step (method 2) |
| batch | 128 B pairs + 128 accepted A images per step |
| epoch | one pass over accepted A images; B is cycled |
| training length | fixed from the fold-0 curve, same for all methods/folds/seeds (section 3) |
| adapter used at evaluation | student (teacher only produces pseudo-captions) |
| z-score std | population (unbiased=False) |
| feature extraction | eval transform (no augmentation), features cached |

## 3. Training length

**Rule (fixed before the curve run, 2026-11-10).** Fold 0, seed 0, method 2, total steps
T ∈ {500, 1000, 2000, 4000, 8000} (each run with its own full cosine schedule and λ_pcap ramp);
B-val mean R@1 logged every 100 steps. T is the smallest grid value whose final student mean R@1 is
within 0.5 points of the best final value over the grid. The chosen T is then used for every method,
fold and seed. Method 1 is run on the same grid only to check that the chosen T is not too short
for it as well; it does not enter the choice.

Smoke run (300 steps) observation that motivated the grid: the student reaches ≈25 mean R@1 on
fold 0 within 100–200 steps, but the EMA teacher (momentum 0.999, time constant ≈1000 steps) is
still near zero-shot at step 300, so short runs would make the pseudo-captions essentially
zero-shot assignments.

**Result** (`python scripts/steps_table.py`, fold 0, seed 0; final values, %):

| T | method 2 mean R@1 | method 2 acc | method 1 mean R@1 | method 1 acc | method 2 EMA teacher mean R@1 (end) |
|---|---|---|---|---|---|
| 500 | 25.02 | 81.28 | 23.84 | 83.11 | 17.90 |
| 1000 | 22.74 | 79.45 | 22.10 | 78.08 | 21.23 |
| 2000 | 20.78 | 75.80 | 19.36 | 75.34 | 23.93 |
| 4000 | 19.63 | 77.63 | 18.54 | 69.86 | 19.86 |
| 8000 | 17.99 | 75.80 | 17.08 | 71.23 | 17.99 |

Rule → **T = 500** (≈14 pseudo-caption epochs of 35 steps). The curves do not show slow convergence
but overfitting: the student peaks at ≈100–300 steps and then declines while L_con on B-train goes
to ≈0.07 (T = 8000); method 1 behaves the same. T = 500 is the lower edge of the grid; shorter runs
were not tried, because the rule was fixed in advance and T is not a tuned hyperparameter here.
Consequence: with EMA 0.999 the teacher moves little within 500 steps (17.9 mean R@1 at the end vs
14.3 zero-shot), so pseudo-captions are mostly assigned from near-frozen CLIP features.

## 4. Results on B held-out folds (`data=kfold5`, seeds 0, 1, 2)

`python scripts/phase1_table.py` (runs: `runs/p1/{semi,sup}_s{0,1,2}`).
Commands (each run covers all 5 folds; s ∈ {0, 1, 2}):

```
python -m src.run model=vit_b32_laion2b device=cuda data=kfold5 method=semi seed=$s exp_name=p1/semi_s$s
python -m src.run model=vit_b32_laion2b device=cuda data=kfold5 method=semi seed=$s exp_name=p1/sup_s$s method.pcap.enabled=false
```

The fold-0 step curve (section 3) uses the same command with `folds=[0] seed=0 method.monitor_every=100
method.steps=$T exp_name=steps/{semi,sup}_T$T`; `python scripts/steps_table.py` builds its table.
Each seed's value is the mean over the 5 held-out folds; mean ± std over the 3 seeds (%).

| method | acc | per-class acc | mean R@1 | I→T R@1 | I→T R@5 | I→T R@10 | T→I R@1 | T→I R@5 | T→I R@10 |
|---|---|---|---|---|---|---|---|---|---|
| Zero-shot | 63.62 | 63.50 | 13.76 | 13.71 | 31.08 | 46.16 | 13.80 | 49.25 | 66.14 |
| Method 1 (supervised) | 82.97 ± 0.31 | 84.66 ± 0.33 | 22.72 ± 0.12 | 23.61 ± 0.19 | 51.80 ± 0.86 | 71.09 ± 0.86 | 21.83 ± 0.42 | 68.33 ± 0.30 | 86.02 ± 0.08 |
| Method 2 (semi) | 83.43 ± 0.23 | 84.59 ± 0.25 | 23.84 ± 0.12 | 25.71 ± 0.26 | 49.67 ± 0.55 | 70.75 ± 0.20 | 21.97 ± 0.08 | 67.90 ± 0.09 | 86.50 ± 0.32 |

Comparison with the README (coverage 0.5, top-3, held-out B protocol, 3 seeds):

| | README | ours |
|---|---|---|
| method 2 mean R@1 | 22.53 ± 0.69 | 23.84 ± 0.12 (+1.31) |
| method 2 I→T R@5 | 53.27 ± 1.14 | 49.67 ± 0.55 |
| method 2 T→I R@5 | 64.60 ± 2.48 | 67.90 ± 0.09 |

**Acceptance check**
- Method 2 mean R@1 within ≈2 points of 22.53: **yes** (+1.31).
- Both methods clearly above zero-shot in classification: **yes** (83.0 / 83.4 vs 63.6).
- Method 2 retrieval above method 1: **yes on mean R@1** (+1.12, seed std 0.12) and on I→T R@1
  (+2.10); **not on R@5**: I→T R@5 −2.13, T→I R@5 −0.43; R@10 mixed (I→T −0.34, T→I +0.48).
  See section 5.

**Cached vs standard evaluation path.** Over 40 fold runs (phase 1 main + step curve) the metrics from
cached features and from the wrapped `model.encode_image` + `evaluate` path are identical in 35;
in 5 a single image→text rank moves by one position (|Δ| = 1/218 or 1/219 on one I→T R@k), never
classification. Cause: duplicate caption texts have identical text features, so similarity ties are
exact and their order depends on floating-point noise between the cached features (batch 256) and
the evaluation pass (batch 128).

## 5. Differences from the README

1. **R@5 trend.** In the README test table method 2 is above method 1 on both R@5 directions. On the
   B held-out folds method 2 gains at R@1 (mostly I→T R@1) but is lower at I→T R@5 (−2.1) and
   slightly lower at T→I R@5 (−0.4). The README sweep reports I→T R@5 53.27 for method 2; ours is 49.67.
2. **Training length.** The README does not state it; the fixed rule gives T = 500 steps, at which
   the EMA teacher is still close to zero-shot. Longer training overfits B-train with these
   defaults (no augmentation on cached features).
3. **Classification gain from captions.** Method 2 − method 1 = +0.46 accuracy (−0.07 per-class),
   versus +0.91 on C in the README; both small, consistent with the research question.
4. **Absolute accuracy level.** Adapted accuracy on B held-out folds (≈83 %) is far above the README's
   C numbers (73.8 / 74.7), while zero-shot agrees (63.6 vs 63.8). The README C models may be trained
   differently (e.g. on all of B); we cannot check this without touching C and do not interpret it.
