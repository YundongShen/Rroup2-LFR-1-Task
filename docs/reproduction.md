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

(results to be filled)

## 4. Results on B held-out folds (`data=kfold5`, seeds 0, 1, 2)

(to be filled)

## 5. Differences from the README

(to be filled)
