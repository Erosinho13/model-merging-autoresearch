# Baseline results relative to the reference algorithm

These are the operator's supplied test-split results, transcribed to two decimals.
The reference algorithm is anonymous; its implementation and settings are private.

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

100 means parity with the reference; values above 100 are better. Avg is the
arithmetic mean of the three reference-relative task scores. Values are not capped
at 100. Base model, SNLI adapter, MNLI adapter, and SciTail adapter rows show the
scores for the corresponding unmerged experts.
