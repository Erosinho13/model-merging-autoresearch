# AI research-agent evaluation: LoRA model merging

This repository evaluates whether an existing LLM agent can develop and test
new methods for merging LoRA-finetuned models. Inspect manages the research loop;
Modal runs candidate code and a fixed NLP evaluator on remote compute.

## Current benchmark

- **Model:** Llama-3-8B with rank-16 LoRA adapters.
- **Tasks:** SNLI, MNLI and SciTail, scored using classification accuracy.
- **Baselines:** averaging, Task Arithmetic, TIES, DARE-TIES, KnOTS-TIES and
  KnOTS-DARE-TIES, plus the base model and individual task adapters.
- **Primary reference:** locally tuned KnOTS-TIES, currently the best measured
  method in this benchmark. KnOTS-DARE-TIES remains an additional comparison;
  both variants need not outperform every other baseline.
- **Research feedback:** test reference-relative scores and raw validation accuracies.
  Optional hyperparameter tuning must use validation; extra unseen-dataset
  checks are deferred.
- **Current assessment:** measured performance within the model-merging research
  scope. Algorithmic novelty is not currently scored.
- **Status:** baseline evaluation and the agent interface are implemented;
  the revised full-size candidate path still needs remote integration validation.

## Getting started

Run commands from the repository root with the `eai-tests` environment activated
and Modal authenticated. See [MODAL.md](MODAL.md) for Hugging Face access and setup.

```bash
# Cache model weights on CPU.
python -m modal run modal_app.py --download

# Evaluate a baseline on validation.
python -m modal run modal_app.py --method task_arithmetic --split val

# Optional GPU smoke check or CPU dataset/access check.
python -m modal run modal_diagnostics.py
python -m modal run modal_diagnostics.py --prepare
```

Cloud commands incur compute charges. Evaluation defaults to test if `--split`
is omitted. Select hyperparameters using validation before final testing.

## Repository layout

| Location | Purpose |
|---|---|
| `modal_app.py` | Checkpoint setup, cache checks and baseline evaluation |
| `modal_diagnostics.py` | Optional GPU and dataset diagnostics |
| `eval_scripts/nli_fixed_test.py` | Fixed NLP evaluation pipeline |
| `baseline_configs.py` | Public settings used by averaging, TA, TIES and DARE-TIES |
| `configs/` | Private model/dataset setup and reference-method settings |
| `dataset/` | Private dataset loaders and preprocessing |
| `private_eval/` | Private model loading, scoring and reference alignment |
| `task_merger.py`, `merging_functions.py`, `masking_ops.py`, `ft_handlers.py` | Merging implementations and parameter handling |
| `utils.py` | Public merging dispatch and seed utilities |
| `research_agent/` | Preliminary Inspect tools, candidate execution and references |

See [AGENT.md](AGENT.md) for the agent scaffold and
[reference preparation](research_agent/references/README.md) for PDF conversion.
Baseline JSON reports are saved in the Modal volume `lora-merging-results`;
Inspect transcripts and candidate snapshots are saved locally during agent runs.

## Evaluation integrity

The intended agent workspace exposes baseline implementations and reference
papers, with one editable merging function. Evaluation code and KnOTS internals
remain private; the agent receives an evaluation tool that returns scores.
The agent's read tools use an explicit public source view, and candidate code
runs in a separate sandbox. This README is operator-facing and is excluded from
that view. `research_agent/interface.md` supplies the agent's public guide.

The three-task benchmark uses fixed released settings or explicit overrides.
Its scores are not directly equivalent to the original six-task, tuned results.

## Attribution

Baseline and evaluation code is adapted from the
[KnOTS repository](https://github.com/gstoica27/KnOTS), accompanying
[Model Merging with SVD to Tie the KnOTS](https://arxiv.org/abs/2410.19735).
The original license is retained in [LICENSE](LICENSE).
