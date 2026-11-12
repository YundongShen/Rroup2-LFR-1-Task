# Balanced Caption-to-Class Bridging

Branch `shipeng/bridge` · **Complete.** On C the bridge does **not** improve classification accuracy over Semi-supervised (81.04 vs 81.07, paired p = 0.65); it gives a small, consistent mean-per-class gain (+0.34, p = 0.002) and slightly better retrieval. Details: `docs/bridge_results.md`.

## 1. Motivation

Baselines from the `main` README (frozen OpenCLIP ViT-B/32 `laion2b_s34b_b79k` + image-side adapter):

| Method | Cls. acc | I→T R@5 | T→I R@5 |
| --- | --- | --- | --- |
| Zero-shot CLIP | 63.76 | 17.84 | 21.23 |
| Supervised (B captions) | 73.82 ± 0.27 | 20.22 ± 0.62 | 21.62 ± 0.25 |
| Semi-supervised (+ A pseudo-captions) | 74.73 ± 0.25 | 22.71 ± 0.56 | 22.28 ± 0.33 |

Pseudo-captions on A add ~2.5 points of I→T R@5 but only 0.9 points of classification. Semi-supervised trains *which captions an image matches*, never *which class prompt it matches*, although the latter is how classification is scored. Three candidate explanations:

- **Conversion gap:** pseudo-captions carry class evidence that is never aggregated into the image–class-prompt relation.
- **Class bias:** ROSE (`yundong/a-only`) found zero-shot CLIP never predicts some classes (center) and over-predicts others (stadium); caption alignment does not fix this.
- **Near-synonyms:** errors concentrate on classes with near-identical names (playfields / stadium / playground).

We test these first (§4.1), then address the first two with one added loss.

## 2. Method

### 2.1 Overview

The Semi-supervised baseline is kept unchanged: caption bank from B, EMA teacher, top-3 pseudo-captions, coverage 0.5, and the losses $`\mathcal L_{\text{con}}`$ and $`\mathcal L_{\text{pcap}}`$. The only trainable module is still the image-side adapter

```math
\tilde z = z + 0.2\,h(z).
```

We add one loss. At the start of every epoch, class pseudo-labels are generated for the A images (steps 1–3, no gradients); the adapter is then trained on them (step 4).

### 2.2 Step 1: caption-to-class table (computed once)

Each class $`c`$ gets a prompt embedding from the frozen text encoder $`g`$:

```math
v_c = g(\text{"an aerial photo of } c\text{."})
```

Each caption $`j`$ in the bank gets a distribution over the 31 classes:

```math
a_{jc} = \mathrm{softmax}_c\!\left(\frac{\cos\big(g(\text{caption}_j),\, v_c\big)}{\tau_{\text{text}}}\right)
```

### 2.3 Step 2: bridging

The teacher assigns each A image $`u`$ its top-3 captions with weights $`q_{uj}`$. The image's caption-derived class distribution is

```math
r_{uc} = \sum_j q_{uj}\, a_{jc}.
```

Differently worded captions about the same scene pool their evidence; vague captions give a flat distribution instead of a forced decision.

### 2.4 Step 3: partial balancing and hard labels

An unbalanced Sinkhorn over **all** A images finds per-class scalings $`\alpha_c`$:

```math
Q_{uc} \propto r_{uc}^{1/\varepsilon_b}\,\alpha_c, \qquad \sum_c Q_{uc} = 1.
```

In each iteration the class masses are rescaled by $`(\pi_c / \text{mass}_c)^{\kappa}`$:

- $`\kappa = 1`$ matches the prior exactly; $`\kappa = 0`$ disables balancing.
- The prior $`\pi`$ is uniform, or the mean of $`a_{jc}`$ over the B-train captions. Neither uses class labels.

The pseudo-label is

```math
\hat y_u = \arg\max_c Q_{uc}.
```

### 2.5 Step 4: training

On the images $`\mathcal U_{\text{sel}}`$ selected by the baseline, the frozen class prompts act as the classifier:

```math
\mathcal L_{\text{bridge}} = -\frac{1}{|\mathcal U_{\text{sel}}|}\sum_{u\in\mathcal U_{\text{sel}}} \log \mathrm{softmax}_c\big(s\cos(\tilde z_u, v_c)\big)_{\hat y_u}
```

```math
\mathcal L = \mathcal L_{\text{con}} + \lambda_{\text{pcap}}\,\mathcal L_{\text{pcap}} + \lambda_{\text{bridge}}\,\mathcal L_{\text{bridge}}
```

$`\lambda_{\text{bridge}}`$ ramps linearly from 0 over the first 20% of training. Gradients reach only the adapter $`h`$.

### 2.6 Design choices

- **Balancing on all of A:** the confident subset selected by the baseline is skewed towards easy classes.
- **Hard labels:** ROSE showed that soft targets with per-epoch relabelling collapse.
- **Partial balancing:** the true class prior is unknown.
- **Different prompts:** training uses "an aerial photo of {}.", evaluation uses "A satellite image of {}.", so gains do not come from fitting the evaluation wording.

### 2.7 Relation to other branches

| Method | Pseudo-label source | Balancing |
| --- | --- | --- |
| Semi-supervised (`main`) | none | – |
| ROSE (`yundong/a-only`) | image–class-prompt similarity | full |
| **Ours** | **pseudo-captions** | **partial** |

Our hypothesis is that captions hold class evidence that class names lack, especially for near-synonymous classes where ROSE scores 0% (playfields, sparse residential).

## 3. Data usage

- **A:** images only; captions, labels and filenames (contain class names) never read.
- **B:** images + captions for training; caption bank from each fold's B-train part only. Class labels used **only** for evaluation and hyperparameter selection on held-out folds, never for training. The slides list B as "images + captions" without addressing its labels; this interpretation matches `main`'s pipeline (stratified folds, held-out zero-shot accuracy).
- **C:** final evaluation only, once, after configs are frozen, behind the repo's C lock.
- **Class names** are used in training (no image labels). Not strictly zero-shot w.r.t. class names; ROSE makes the same choice.

## 4. Experiments

Development: `data=kfold5` (stratified, seed 0), seeds 0–2. Final: C, seeds 0–4. Selection on held-out B by classification accuracy, subject to mean R@1 ≤ 0.5 below Semi-supervised; rule and grid fixed beforehand in `docs/bridge_protocol.md`. Supervised and Semi-supervised are our reproductions (differences in `docs/reproduction.md`).

### 4.1 Diagnosis (held-out B; `scripts/diagnose_bias.py`; criteria pre-registered in `docs/TASK.md`)

| Metric | Zero-shot | Supervised | Semi-sup. |
| --- | --- | --- | --- |
| Accuracy / mean per-class | 63.62 / 63.51 | 82.97 / 84.57 | 83.42 / 84.57 |
| TV distance, predicted vs. true class distribution | 23.58 | 12.22 | 11.52 |
| Classes never predicted | 1 (center) | 0 | 1 (sparse residential) |
| Acc. after test-time balancing: uniform / oracle prior | 68.10 / 67.55 | 83.00 / 81.75 | 84.13 / 83.52 |
| Pseudo-caption correct when image wrong | 47.74 | 30.76 | 30.70 |
| Acc., image × pseudo-caption evidence | 78.34 | 83.70 | 84.40 |
| Errors into 3 text-nearest classes (chance 10) | 28.14 | 56.17 | 53.13 |

Seeds 0–2 means (%); zero-shot is deterministic.

Conclusion (pre-registered criteria, Semi-supervised):

- **Class bias: not supported.** Balancing gains only +0.70 points, against a required 2.
- **Conversion gap: borderline, not supported.** The pseudo-caption is correct for 30.70 % of the images the classifier gets wrong (criterion ≥ 30 %), but combining image and pseudo-caption evidence adds +0.98 points (criterion > 1).
- **Near-synonyms: supported.** 53 % of errors go into the 3 text-nearest classes, 5.3× chance.

The dominant errors are playfields → playground and sparse residential → medium residential. Captions do not separate these classes either: of 335 playfields captions, 135 say "playground" and none says "playfield". Details: `docs/diagnosis.md`.

### 4.2 Main results and ablations (C, 5 seeds)

| Method | Cls. acc | Mean per-class | TV dist. | I→T R@1/5/10 | T→I R@1/5/10 |
| --- | --- | --- | --- | --- | --- |
| Zero-shot CLIP | 62.76 | 63.03 | 22.87 | 6.59 / 17.47 / 27.90 | 6.40 / 21.21 / 33.80 |
| Supervised | 78.77 ± 0.17 | 80.38 ± 0.16 | 14.18 ± 0.07 | 8.63 / 21.53 / 32.60 | 6.87 / 23.73 / 37.93 |
| Semi-supervised | 81.07 ± 0.14 | 82.20 ± 0.17 | 13.32 ± 0.14 | 8.76 / 23.22 / 34.97 | 6.87 / 24.41 / 38.73 |
| **Ours** | 81.04 ± 0.20 | **82.54 ± 0.22** | 14.30 ± 0.12 | 8.97 / 23.37 / 35.21 | 6.93 / 24.42 / 38.88 |
| Ours, no balancing ($`\kappa=0`$) | 78.58 ± 0.25 | 79.89 ± 0.28 | 16.94 ± 0.12 | 8.93 / 23.50 / 34.98 | 6.89 / 24.32 / 38.76 |
| Ours, image-derived labels | 77.88 ± 0.13 | 79.05 ± 0.15 | 16.42 ± 0.06 | 8.76 / 23.30 / 34.56 | 6.85 / 24.32 / 38.71 |
| Ours, soft targets | 80.43 ± 0.17 | 81.94 ± 0.21 | 14.52 ± 0.20 | 8.97 / 23.55 / 35.17 | 6.91 / 24.36 / 38.71 |
| Ours, other prior (caption) | 79.06 ± 0.21 | 80.36 ± 0.23 | 16.65 ± 0.18 | 8.85 / 23.46 / 35.04 | 6.90 / 24.31 / 38.83 |
| Semi-sup. + test-time balancing † | 82.13 ± 0.11 | 83.44 ± 0.14 | 8.11 ± 0.12 | 8.76 / 23.22 / 34.97 | 6.87 / 24.41 / 38.73 |

**Setup.**

- Values are % on C, mean ± std over seeds 0–4. For each seed, the 5 fold models (`data=kfold5`) are each evaluated on all of C and averaged. Zero-shot is deterministic. R@k values are seed means.
- Ours = λ_bridge 0.25, ε_b 1, κ 0.5, uniform prior, hard targets, selected on held-out B.
- C was run once, with all configurations frozen beforehand (`docs/bridge_protocol.md` §7).
- Our reproductions score higher in accuracy than the `main` README (§1); see `docs/reproduction.md`.

† Transductive (uses unlabeled evaluation images); not comparable to other rows. The best balancing setting (κ 1, ε_b 1, uniform prior) was selected on B.

**Paired test, ours vs. Semi-supervised** (same seed, n = 5, two-sided t-test):

| Metric | Difference | p |
| --- | --- | --- |
| Accuracy | −0.03 ± 0.15 | 0.65 (not significant) |
| Mean per-class | +0.34 ± 0.11 (positive in all 5 seeds) | 0.002 |
| Mean R@1 | +0.14 ± 0.14 | 0.10 |

**Findings.**

1. **No accuracy gain.** The +0.45 seen on held-out B came from selecting the best of 24 configurations on the same folds and did not carry over to C.
2. **Balancing is necessary.** Without it, the bridge lowers accuracy by 2.5 points; balancing recovers this. A uniform prior beats the caption prior, caption-derived labels beat image-derived ones, and hard targets beat soft ones (the same ordering on B and C).
3. **Pseudo-labels are worse than the student.** The bridge's pseudo-labels on held-out B are at most 79.5 % correct, below Semi-supervised's own 83.4 %.
4. **Playfields gets worse.** It receives no caption-derived class mass (0 hard pseudo-labels at κ = 0), so its accuracy falls from 17.0 to 1.3 % and the TV distance rises.
5. **Only transductive balancing helps.** Balancing Semi-supervised's predictions at test time gives +1.06 accuracy in every seed. Bias correction itself is useful, but routing it through caption pseudo-labels loses the effect.

### 4.3 Sensitivity (held-out B, seed-0 grid, caption source; cls. acc / mean R@1, each value averaged over the other grid axes)

| $`\lambda_{\text{bridge}}`$ 0.25 / 0.5 / 1 | $`\kappa`$ 0 / 0.5 / 1 | $`\varepsilon_b`$ 0.5 / 1 | $`\pi`$ uniform / caption |
| --- | --- | --- | --- |
| 83.09 / 23.85 · 82.31 / 23.71 · 81.43 / 23.25 | 81.42 / 23.54 · 82.87 / 23.81 · 82.11 / 23.43 | 82.17 / 23.56 · 82.38 / 23.65 | 83.34 / 23.55 · 81.64 / 23.68 |

**Reference and how to read it.**

- Semi-supervised, seed 0: 83.18 / 23.68.
- Accuracy falls as $`\lambda_{\text{bridge}}`$ grows. κ = 0.5 is best. $`\varepsilon_b`$ hardly matters.
- The prior column covers κ > 0 only (at κ = 0 the prior has no effect).
- Seed-to-seed std is about 0.2–0.4 for accuracy and 0.1–0.35 for mean R@1.
- Full grid: `docs/bridge_results.md` §2 and `docs/figs/bridge_sensitivity.png`.

### 4.4 Per-class accuracy (C)

| Class | Zero-shot | Semi-sup. | Ours |
| --- | --- | --- | --- |
| center | 0.0 | 11.8 | 20.5 |
| stadium | 100.0 | 93.1 | 94.3 |
| playfields | 0.0 | 17.0 | 1.3 |
| sparse residential | 0.0 | 0.0 | 1.6 |
| commercial | 17.1 | 68.8 | 69.0 |

Values are % on C, means over seeds and fold models.

- **Largest gains** (ours vs. Semi-sup.): airport +11.2, center +8.7, meadow +6.7.
- **Largest losses:** playfields −15.7, medium residential −10.6, park −9.2.
- All 31 classes: `docs/bridge_results.md` §6.

## 5. Limitations

- Class names are used in training (§3).
- Bridging cannot create evidence absent from the captions. This is confirmed: playfields gets no caption-derived class mass and is lost (§4.4).
- The balancing prior is approximate; partial balancing and the prior ablation measure the cost.
- Pulling features towards 31 class prompts may compress within-class variation and hurt retrieval, hence retrieval is reported everywhere.

## 6. Files

`docs/bridge.md` (this file) · `docs/TASK.md`, `CLAUDE.md` (task spec, working rules) · `scripts/diagnose_bias.py` · added per phase: `docs/reproduction.md`, `docs/diagnosis.md`, `docs/bridge_protocol.md`, `docs/bridge_results.md`.

## 7. Use of AI tools

Method developed in discussion with Claude (Anthropic). All design decisions, data-usage interpretations and results are reviewed by the author.