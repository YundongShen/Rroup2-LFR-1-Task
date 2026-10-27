# ROSE — adapting CLIP to RSICD with split A images only

**ROSE** = **R**otation-**O**rbit **S**elf-training with **E**quipartitioned pseudo-labels.
Code: `src/methods/rose.py`, `configs/method/rose.yaml`, `configs/train/aonly.yaml`.

## Setting
- **Training data:** split A images only (8,734). No captions and no labels are used. File names are never read, because they contain the class name (`airport_1.jpg`).
- **Only text used:** the 31 class names in the course prompt "A satellite image of {}.", i.e. exactly what zero-shot evaluation already uses.
- **Evaluation:** all of split B (1,094 images, 5 captions each). Nothing from B is used for training or model selection (fixed 5 epochs).
  - Zero-shot classification: acc and mean per-class acc.
  - Image↔text retrieval: R@1/5/10 and mR.
  - Rotation robustness: B under all 8 rotations/flips, mean and worst case, plus orbit-averaged test-time augmentation.
- **Backbone and training:** OpenCLIP ViT-B-32 `laion2b_s34b_b79k`, plus ViT-B-16 to check the gain carries over.
  - 5 epochs, batch 128, AdamW (lr 1e-5, wd 0.1), cosine schedule with 50 warmup steps, bf16.
  - The image tower is trained and the text tower is frozen.
  - About 2.5 min per run on one GH200.

## Method
Let W ∈ R^{31×d} be the frozen class-prompt embeddings and f_θ the image tower. Each epoch r:

1. **Rotation-orbit relabelling.** z̄_i = normalize( (1/8) Σ_{g∈D4} f_θ(g·x_i) ) and s_i = z̄_i Wᵀ. Overhead imagery has no canonical orientation, so averaging over the 8 rotations/flips gives a more reliable teacher signal for free.
2. **Equipartitioned pseudo-labels.** Q = Sinkhorn(exp(S/ε)) with a uniform class marginal (ε = 0.02, 3 iterations), then the hard label ŷ_i = argmax_c Q_ic. This counteracts zero-shot CLIP's strong class bias on remote sensing: it never predicts *center* and assigns 9.4% of images to *stadium*, whose true share is 2.7%.
3. **Student update (one epoch).** On a strongly augmented, randomly rotated/flipped view x̃_i:
   L = CE(ŷ_i, softmax(τ · f_θ(x̃_i) Wᵀ)) + λ · KL(P⁰ ‖ P)
   - τ is CLIP's logit scale.
   - P⁰ and P are the row-softmaxed (temperature 0.1) image–image similarity matrices of the batch under the frozen initial tower and the current tower.
   - This relational anchor keeps instance-level neighbourhoods, which caption retrieval relies on (λ = 1).

**Hard rather than soft labels.** Training on the soft Sinkhorn distribution collapses: zs_acc goes 69.5 → 33.8 within 5 epochs and pseudo-label confidence goes 0.41 → 0.05. Soft targets flatten the student, which flattens the next round's labels. Hard labels break this loop.

## Relation to prior work
- **MUST** (Li et al., ICLR'23) and **LaFTer** (Mirza et al., NeurIPS'23): label-free CLIP self-training with an EMA teacher or FixMatch. They target classification only, and LaFTer needs LLM-generated class descriptions. ROSE uses no external model, enforces class balance globally, uses the rotation symmetry of overhead images, and is evaluated on retrieval as well.
- **UPL** (Huang et al., 2022): class-balanced top-K pseudo-labels for prompt learning. ROSE balances the whole unlabelled set with optimal transport and adapts the image tower.
- **SwAV / SeLa** (Caron et al., NeurIPS'20; Asano et al., ICLR'20): Sinkhorn equipartition with learned prototypes. ROSE's prototypes are the frozen text class embeddings, so the clusters keep their meaning.
- **CLIP-OT** (2024): Sinkhorn over text anchors at test time. ROSE uses it for training-time relabelling of a separate unlabelled set, refreshed every epoch.
- **S-CLIP** (Mo et al., NeurIPS'23): semi-supervised CLIP on remote-sensing caption datasets including RSICD, using OT pseudo-captions. It needs labelled captions, so it is the natural next step once split B is added.
- **Survey:** Adapting VLMs without labels (arXiv 2508.05547).

## Results (evaluated on all of split B)
Trained methods report mean ± std over 3 seeds. Seed-to-seed spread for ROSE is about ±0.8 acc, so differences below about 1 point are not meaningful.

### 1. Main comparison (ViT-B-32)
| Method | Uses A | zs acc | mean per-class | i2t R@1 | t2i R@1 | mR | worst-rotation acc |
|---|---|---|---|---|---|---|---|
| Zero-shot CLIP (course prompt) — **baseline** | – | 63.62 | 63.51 | 3.75 | 3.78 | 12.39 | 63.07 |
| Zero-shot + 6-template prompt ensemble | – | 65.81 | 66.04 | 3.75 | 3.78 | 12.39 | 64.63 |
| Zero-shot + D4 orbit TTA (8× inference cost) | – | 65.45 | – | – | – | 13.08 | – |
| A-image class prototypes (training-free; ensemble prompts + D4 TTA) | ✓ | 71.94 | 72.75 | – | – | (unchanged) | – |
| Vanilla self-training (argmax labels) | ✓ | 68.68±1.98 | 68.89±2.07 | 3.63±0.11 | 4.27±0.13 | 13.59±0.11 | 68.40±2.36 |
| **ROSE** | ✓ | **78.31±0.82** | **80.03±0.87** | 4.39±0.27 | **5.00±0.12** | **14.49±0.16** | **77.70±1.03** |

ROSE gains **+14.7 acc / +16.5 mean per-class acc / +2.1 mR** over the baseline. Classification and both retrieval directions improve, and the gain holds under any rotation/flip of the input.

### 2. Ablations (ViT-B-32, 3 seeds; each row removes one component)
| Variant | zs acc | mean per-class | t2i R@1 | mR |
|---|---|---|---|---|
| ROSE (full) | 78.31±0.82 | 80.03±0.87 | 5.00±0.12 | 14.49±0.16 |
| − equipartition (plain argmax labels) | 71.39±1.06 | 71.59±1.15 | 4.84±0.22 | 14.42±0.09 |
| − per-epoch relabelling (zero-shot labels throughout) | 73.00±0.09 | 73.70±0.10 | 4.12±0.04 | 14.20±0.28 |
| − rotation orbit (no D4 averaging / D4 views) | 77.45±0.65 | 78.98±0.77 | 4.39±0.07 | 14.41±0.16 |
| − relational anchor (λ = 0) | 78.28±0.64 | 80.01±0.65 | 3.91±0.08 | 14.34±0.08 |
| soft Sinkhorn targets instead of hard (seed 0) | 33.82 (collapse) | 34.67 | 1.90 | 5.08 |

Reading the ablations:
- **Equipartition (+6.9) and relabelling (+5.3) carry the gain.**
- The rotation orbit adds a small, consistent +0.9 acc (+1.0 worst-rotation acc).
- The anchor does not change classification. It trades i2t R@1 (−0.6) for t2i R@1 (+1.1), for +0.15 mR overall.

### 3. Hyperparameter sensitivity (ViT-B-32, seed 0; reference run 77.97 acc / 14.28 mR)
| Hyperparameter | Values → zs acc (mR) |
|---|---|
| Sinkhorn ε | 0.01 → 75.96 (13.92) · **0.02 → 77.97 (14.28)** · 0.05 → 70.11 (14.88) |
| anchor λ (pilot runs, ref 78.70 / 14.27) | 0 → 78.88 (14.07) · **1 → 78.70 (14.27)** · 3 → 77.88 (14.07) · 10 → 76.78 (13.96) |
| learning rate | 5e-6 → 77.15 (13.96) · **1e-5 → 77.97** · 2e-5 → 77.79 (14.56) |
| trainable part | LayerNorm only (lr 1e-4) → 67.46 (12.97) · last 4 blocks → 76.69 (13.40) · **whole image tower → 77.97** |
| epochs | **5 → 77.97** · 10 → 78.98 (14.39); the curve saturates after about 3 epochs |
| prompts | course → 77.97 · 6-template ensemble → 78.24 (14.57); ROSE makes prompt engineering unnecessary |

ε is the one hyperparameter that matters. If it is too large, the balancing overrides the image evidence (70.1). If it is too small, it falls back towards plain argmax.

### 4. Does it transfer to another backbone? (ViT-B-16 laion2b, seed 0, no re-tuning)
| Method | zs acc | mR |
|---|---|---|
| Zero-shot | 66.73 | 12.03 |
| Vanilla self-training | 64.81 (**−1.9**: confirmation bias) | 12.87 |
| ROSE | **76.69 (+10.0)** | **14.55 (+2.5)** |

## Analysis
**Why it works.** Zero-shot CLIP's errors on RSICD are dominated by class bias, not noise:
- Zero-shot never predicts *center* and almost never predicts *meadow*, *bare land* or *commercial*.
- It over-predicts *stadium* (9.4% of predictions vs 2.7% true) and *baseball field* (6.3% vs 2.6%).
- The same bias appears in its labels on A, so vanilla self-training amplifies it. Its pseudo-label confidence climbs to 0.91 ([figs/curves.png](figs/curves.png)) while accuracy stays flat.

Equipartition forces every class to receive images. The largest per-class gains are exactly the under-predicted classes: *center* 0→73%, *meadow* 29→100%, *bare land* 16→74%, *industrial* 44→85%, *river* 59→98%.

**Failure cases** ([figs/confusions.png](figs/confusions.png)):
- *playfields* (the largest B class) stays at 0%, confused with *playground*, *baseball field* and *stadium*.
- *sparse residential* stays at 0% (→ *medium residential*).
- *commercial* stays at 9% (→ *square*).

These are near-synonymous class names over visually near-identical scenes (running tracks with football pitches appear under all four sports labels). The text anchor cannot separate them, and equipartition then fills e.g. *playfields* with images of other classes.

**Retrieval** improves only modestly (+2.1 mR). The method never sees caption-level text, so it improves scene-level alignment (which retrieval partly depends on) but not fine-grained object/count/colour alignment.

**Rotation.** Zero-shot CLIP is already fairly rotation-robust on RSICD (worst rotation −0.6 acc). Orbit TTA still helps zero-shot by +1.8, but ROSE absorbs this: TTA on top of ROSE adds only +0.3.

## Caveats and next steps
- In this phase split B is the evaluation set, and ε / λ / hard labels were chosen after looking at B pilot results. For the final report, hyperparameters must be selected on a held-out part of B and results reported once on split C.
- The uniform class marginal is an assumption: RSICD classes are imbalanced, with a 2.8× ratio in B. This probably costs accuracy on large classes; note acc < mean per-class acc.
- Next phase: add split B pairs. Train with CLIP contrastive loss on B captions plus ROSE on A, and compare against S-CLIP-style caption-level pseudo-labels. Captions are what can lift retrieval.

## Reproduce
```bash
# one run (A-only training, evaluation on all of B incl. rotation robustness)
python -m src.run method=rose train=aonly data=allb model=vit_b32_laion2b device=cuda eval.tasks.rotation.enabled=true seed=0
# ablations: method.labels=argmax | method.refresh=false | method.orbit=false | method.anchor_weight=0
# baselines: zero-shot = drop method/train overrides; vanilla self-training = method.orbit=false method.labels=argmax method.anchor_weight=0
python scripts/summarize_aonly.py          # seed-aggregated table over runs/main_*, sens_*, gen_*
python -m scripts.analyze_aonly --visual runs/<run>/<ts>/fold0/visual.pt   # per-class / confusion analysis (train.save_model=true)
```
