# LR Assignment 1: Validation Results

Full validation results for the CLIPJepa model on the held-out RSICD validation subset. Retrieval scores and accuracies are shown as percentages; loss is unchanged.

## Classification and retrieval

| Metric | Score |
|---|---:|
| Zero-shot accuracy (`zs_acc`) | 82.11% |
| Mean per-class accuracy (`zs_mean_per_class_acc`) | 81.73% |
| Image-to-text R@1 | 13.76% |
| Text-to-image R@1 | 11.65% |
| Image-to-text R@5 | 35.32% |
| Text-to-image R@5 | 46.15% |
| Image-to-text R@10 | 50.00% |
| Text-to-image R@10 | 70.46% |
| Mean retrieval recall (`mR`) | 37.89% |
| Symmetric retrieval loss | 9.6398 |
| Image-to-text accuracy | 13.76% |
| Text-to-image accuracy | 11.65% |

The image-to-text and text-to-image accuracy values match their respective R@1 scores.

## Rotation robustness

| Metric | Score |
|---|---:|
| Mean zero-shot accuracy across eight views (`rot_zs_acc_mean`) | 83.37% |
| Lowest zero-shot accuracy across views (`rot_zs_acc_min`) | 80.73% |
| Mean retrieval recall across views (`rot_mR_mean`) | 38.38% |
| Zero-shot accuracy with orbit-averaged test-time augmentation (`tta_zs_acc`) | 83.94% |
| Retrieval recall with orbit-averaged test-time augmentation (`tta_mR`) | 38.36% |

These results are from one run; standard deviation across independent runs is not available.
