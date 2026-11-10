# Balanced Caption-to-Class Bridging

Branch `shipeng/bridge` · **Work in progress**: result tables are placeholders.

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
| Accuracy / mean per-class | — | — | — |
| TV distance, predicted vs. true class distribution | — | — | — |
| Classes never predicted | — | — | — |
| Acc. after test-time balancing: uniform / oracle prior | — | — | — |
| Pseudo-caption correct when image wrong | — | — | — |
| Acc., image × pseudo-caption evidence | — | — | — |
| Errors into 3 text-nearest classes (chance —) | — | — | — |

Conclusion: —

### 4.2 Main results and ablations (C, 5 seeds)

| Method | Cls. acc | Mean per-class | TV dist. | I→T R@1/5/10 | T→I R@1/5/10 |
| --- | --- | --- | --- | --- | --- |
| Zero-shot CLIP | — | — | — | — | — |
| Supervised | — | — | — | — | — |
| Semi-supervised | — | — | — | — | — |
| **Ours** | — | — | — | — | — |
| Ours, no balancing ($`\kappa=0`$) | — | — | — | — | — |
| Ours, image-derived labels | — | — | — | — | — |
| Ours, soft targets | — | — | — | — | — |
| Ours, other prior | — | — | — | — | — |
| Semi-sup. + test-time balancing † | — | — | — | — | — |

† Transductive (uses unlabeled evaluation images); not comparable to other rows. Paired test, ours vs. Semi-supervised: —

### 4.3 Sensitivity (held-out B, 3 seeds; cls. acc / mean R@1)

| $`\lambda_{\text{bridge}}`$ 0.25 / 0.5 / 1 | $`\kappa`$ 0 / 0.5 / 1 | $`\varepsilon_b`$ 0.5 / 1 | $`\pi`$ uniform / caption |
| --- | --- | --- | --- |
| — | — | — | — |

### 4.4 Per-class accuracy (C)

| Class | Zero-shot | Semi-sup. | Ours |
| --- | --- | --- | --- |
| center / stadium / playfields / sparse residential / commercial | — | — | — |

## 5. Limitations

- Class names are used in training (§3).
- Bridging cannot create evidence absent from the captions.
- The balancing prior is approximate; partial balancing and the prior ablation measure the cost.
- Pulling features towards 31 class prompts may compress within-class variation and hurt retrieval, hence retrieval is reported everywhere.

## 6. Files

`docs/bridge.md` (this file) · `docs/TASK.md`, `CLAUDE.md` (task spec, working rules) · `scripts/diagnose_bias.py` · added per phase: `docs/reproduction.md`, `docs/diagnosis.md`, `docs/bridge_protocol.md`, `docs/bridge_results.md`.

## 7. Use of AI tools

Method developed in discussion with Claude (Anthropic). All design decisions, data-usage interpretations and results are reviewed by the author.