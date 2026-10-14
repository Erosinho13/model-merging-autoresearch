# Baseline results relative to the reference algorithm

These are the operator's supplied results, transcribed to two decimals.
The reference algorithm is anonymous; its implementation and settings are private.

## Test split

| Method | SNLI | MNLI | SciTail | Avg |
|---|---:|---:|---:|---:|
| Base model | 45.93 | 39.22 | 64.48 | 49.88 |
| SNLI adapter | 101.85 | 77.66 | 86.46 | 88.66 |
| MNLI adapter | 97.76 | 102.54 | 91.01 | 97.10 |
| SciTail adapter | 56.13 | 53.37 | 102.50 | 70.67 |
| Averaging | 100.11 | 92.85 | 99.80 | 97.58 |
| Task Arithmetic | 100.51 | 101.09 | 97.35 | 99.65 |
| TIES | 100.50 | 101.54 | 96.60 | 99.55 |
| DARE-TIES | 100.26 | 94.12 | 99.60 | 97.99 |
| Reference algorithm | 100.00 | 100.00 | 100.00 | 100.00 |

## Validation split

| Method | SNLI | MNLI | SciTail | Avg |
|---|---:|---:|---:|---:|
| Base model | 45.61 | 42.07 | 63.16 | 50.28 |
| SNLI adapter | 101.86 | 79.89 | 83.37 | 88.38 |
| MNLI adapter | 98.07 | 102.39 | 92.28 | 97.58 |
| SciTail adapter | 55.34 | 55.88 | 101.27 | 70.83 |
| Averaging | 100.21 | 93.53 | 98.25 | 97.33 |
| Task Arithmetic | 100.57 | 100.81 | 98.64 | 100.01 |
| TIES | 100.57 | 100.93 | 98.16 | 99.89 |
| DARE-TIES | 100.50 | 94.69 | 98.40 | 97.87 |
| Reference algorithm | 100.00 | 100.00 | 100.00 | 100.00 |

100 means parity with the reference; values above 100 are better. Avg is the
arithmetic mean of the three reference-relative task scores. Values are not capped
at 100. Each table uses reference accuracies from its own split; compare candidates
with the baseline table for that same split. Displayed averages may differ slightly
from averages of the rounded displayed task scores. Base model, SNLI adapter,
MNLI adapter, and SciTail adapter rows show the
scores for the corresponding unmerged experts.
