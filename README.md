# Rroup2-LFR-1-Task

Semi-supervised CLIP fine-tuning on RSICD (WASP LFR Module 1).

## Results

Trained on Split A (unlabeled) + Split B (captions), evaluated on Split C. ViT-B/32 (LAION-2B),
mean ± std over 5 seeds. Zero-shot is deterministic, so it has no std.

| Method          | Classification Acc | Image → Text R@5 | Text → Image R@5 |
| --------------- | :----------------: | :--------------: | :--------------: |
| Zero-shot       | 63.76              | 17.84            | 21.23            |
| Supervised      | 73.82 ± 0.27       | 20.22 ± 0.62     | 21.62 ± 0.25     |
| Semi-supervised | 74.73 ± 0.25       | 22.71 ± 0.56     | 22.28 ± 0.33     |

Supervised trains the adapter on B with the contrastive loss only. Semi-supervised adds the
pseudo-caption loss on A.

## Method

CLIP stays frozen. A residual adapter on the image side is the only trained module:

$$
\tilde{z} = z + \alpha\, h(z), \qquad \alpha = 0.2
$$

where $z$ is the frozen CLIP image feature and $h$ a small MLP. The training objective is

$$
\mathcal{L} = \mathcal{L}_{\mathrm{con}} + \lambda_{\mathrm{pcap}}\, \mathcal{L}_{\mathrm{pcap}}
$$

**Contrastive loss** ($\mathcal{L}_{\mathrm{con}}$, Split B). Standard symmetric InfoNCE between
the adapted image features and the CLIP text features of their captions. For a batch of $N$
(image, caption) pairs with logits $\ell_{ij} = s \cos(\tilde{z}_i, t_j)$ and the CLIP
temperature $s$,

$$
\mathcal{L}_{\mathrm{con}} = -\frac{1}{2N} \sum_{i=1}^{N} \left[ \log \frac{e^{\ell_{ii}}}{\sum_{j} e^{\ell_{ij}}} + \log \frac{e^{\ell_{ii}}}{\sum_{j} e^{\ell_{ji}}} \right]
$$

Captions of the same image are masked out of each other's negatives.

**Pseudo-caption loss** ($\mathcal{L}_{\mathrm{pcap}}$, Split A). Split A has no captions, so
every A image borrows captions from B. All B captions form a caption bank. For an A image $u$
and a bank caption $j$, an EMA teacher of the adapter (never the student) computes two scores:

$$
S_{\mathrm{IT}}(u, j) = \cos(z_u, t_j), \qquad S_{\mathrm{II}}(u, j) = \max_{i \in \mathrm{src}(j)} \cos(z_u, z_i)
$$

where $\mathrm{src}(j)$ is the set of B images that caption $j$ belongs to. Both are z-scored
per image over the bank and fused:

$$
R(u, j) = \lambda\, \mathrm{z}(S_{\mathrm{IT}}) + (1 - \lambda)\, \mathrm{z}(S_{\mathrm{II}}), \qquad \lambda = 0.2
$$

The image-to-image path dominates because CLIP's image-image similarity on RSICD is far more
reliable than its image-text similarity.

The top-$k$ captions by $R$ are the candidates $\mathcal{N}_u$ for $u$, weighted by

$$
w_{uj} = \frac{e^{R(u, j)/\tau}}{\sum_{j' \in \mathcal{N}_u} e^{R(u, j')/\tau}}, \qquad \tau = 0.1
$$

Only the `coverage` fraction of A images with the highest $\max_j R(u, j)$ is accepted in an
epoch, so low-confidence images contribute nothing. On the accepted images the student
minimizes a soft cross-entropy over the whole bank:

$$
\mathcal{L}_{\mathrm{pcap}} = -\sum_{j \in \mathcal{N}_u} w_{uj}\, \log \frac{e^{\,s \cos(\tilde{z}_u, t_j)}}{\sum_{j'} e^{\,s \cos(\tilde{z}_u, t_{j'})}}
$$

Candidates and weights are recomputed by the teacher every epoch, and $\lambda_{\mathrm{pcap}}$
ramps up linearly from 0 to 1 over the first 20 % of training.

## Parameter sweeps

Held-out protocol: train on A plus 4 folds of B, evaluate on the 5th fold. Mean ± std over
3 seeds. mean R@1 is the average of the two retrieval directions and is the selection metric.

**Coverage** (top-$k$ = 3)

| coverage | mean R@1     | Image → Text R@5 | Text → Image R@5 |
| :------: | :----------: | :--------------: | :--------------: |
| 0.2      | 20.82 ± 0.31 | 51.75 ± 1.51     | 64.51 ± 1.06     |
| 0.5      | 22.53 ± 0.69 | 53.27 ± 1.14     | 64.60 ± 2.48     |
| 0.7      | 21.48 ± 0.21 | 53.27 ± 3.01     | 64.54 ± 2.96     |
| 0.9      | 22.86 ± 0.41 | 51.60 ± 2.91     | 65.75 ± 2.59     |

**Top-$k$** (coverage = 0.5)

| top-$k$ | mean R@1     | Image → Text R@5 | Text → Image R@5 |
| :-----: | :----------: | :--------------: | :--------------: |
| 1       | 21.32 ± 0.47 | 51.75 ± 1.14     | 63.74 ± 1.37     |
| 3       | 22.53 ± 0.69 | 53.27 ± 1.14     | 64.60 ± 2.48     |
| 5       | 22.42 ± 0.60 | 50.23 ± 2.33     | 65.33 ± 3.25     |

The final configuration uses coverage = 0.5 and top-$k$ = 3.
