# Selected NLP datasets

`snli.py`, `mnli.py` and `scitail.py` retain the original loader implementations.
`huggingface_datasets.py` provides shared loading and tokenization; `configs.py`
retains the original settings for the selected tasks. MNLI uses the repository's
80/20 deterministic validation partition for test/validation evaluation.
