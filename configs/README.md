# NLP configurations

This folder is private: it holds model setup and released reference-method settings.
`nlp_benchmark.py` selects SNLI, MNLI and SciTail. Public baseline merge settings
live in the root `baseline_configs.py`, which the runner uses for averaging,
Task Arithmetic, TIES and DARE-TIES. CLI overrides remain available.
