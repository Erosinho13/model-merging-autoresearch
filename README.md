# AI research-agent evaluation: LoRA model merging

This repository evaluates whether an existing LLM agent can develop and test
new methods for merging LoRA-finetuned models. Inspect manages the research loop;
Modal runs candidate code and a fixed NLP evaluator on remote compute.

## Current benchmark

- **Model:** Llama-3-8B with rank-16 LoRA adapters.
- **Tasks:** SNLI, MNLI and SciTail, scored using classification accuracy.
- **Baselines:** averaging, Task Arithmetic, TIES, DARE-TIES, KnOTS-TIES and
  KnOTS-DARE-TIES, plus the base model and individual task adapters.
- **Primary reference:** locally tuned KnOTS-TIES, the best measured test Avg
  in this benchmark. Task Arithmetic slightly exceeds it on validation.
  KnOTS-DARE-TIES remains an additional operator comparison;
  both variants need not outperform every other baseline.
- **Research feedback:** reference-relative scores on both validation and test.
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

## Running and monitoring the research agent

Place selected paper Markdown in `research_agent/references/` or its paper
subfolders. See [reference preparation](research_agent/references/README.md)
for PDF conversion.

Before the first agent run, check the full candidate pipeline with averaging:

```bash
python -m modal run -m research_agent.modal_runner::main \
  --candidate research_agent/examples/average.py --split val
```

To verify cache reuse, run averaging followed by Task Arithmetic sequentially:

```bash
python -m modal run -m research_agent.modal_runner::main --cache-check --split test
```

The second preparation must report `cache_hit=True`, with the same input-cache key
and sandbox image ID; otherwise the check stops before its GPU evaluation.
Separate reports, candidate snapshots and timing summaries are saved under
`cache-checks/<run-id>/`. Compare preparation time separately from evaluation
time: both candidates still load the cached model and evaluate the full datasets.
If the input cache already exists, both preparations can be cache hits.
Task Arithmetic uses the current coefficient in `baseline_configs.py` (0.5).
On test, expected raw accuracies from the measured baselines are approximately:

| Candidate | SNLI | MNLI | SciTail |
|---|---:|---:|---:|
| Averaging | 90.92 | 81.90 | 93.98 |
| Task Arithmetic | 91.28 | 89.17 | 91.67 |

Compare returned scores with the public baseline table for the same split.
This check incurs Modal compute charges. Local orchestration builds
the sandbox image, then a CPU worker caches fixed inputs before requesting a GPU. Later
attempts reuse those files and image layers. The GPU remains allocated while each
fresh sandbox reconstructs the model and runs the candidate.

Fixed inputs are stored in `lora-research-inputs`. The cache key includes checkpoint
revisions and the source/runtime settings used to prepare them. Changed inputs
invalidate the cache automatically; an incomplete cache is rebuilt. Candidate
sandboxes receive only the public input subdirectory through a read-only mount.
Classification heads and evaluation settings stay in the private cache directory.
Library installation is cached in Modal image layers; changes to the worker script
rebuild its file layer while unchanged dependency layers can be reused.

Set your provider API credentials locally and use an API model identifier supported
by your provider. A Codex model label is not necessarily an API model identifier;
model API charges are separate from Modal charges.

```bash
python -m inspect_ai eval research_agent/task.py \
  --model <provider/model> \
  -T max_evaluations=3 \
  -T token_budget=30000 \
  -T time_limit=3600 \
  --log-dir agent-logs
```

These are the defaults; omit the corresponding `-T` arguments to use them.
The token limit counts input and output across all model calls, including cached
input and repeated conversation history; it is not just a generated-output limit.
`time_limit` is measured in seconds and enforced by Inspect's per-sample timer.
There is one sample per research run. It includes model calls, tool execution and
evaluation waits. The agent can check the timer through `get_budget`; evaluation
feedback also includes remaining time, tokens and attempts.

| Limit | Default | Enforced in |
|---|---:|---|
| Agent wall-clock runtime | 1 hour | `Task(time_limit=time_limit)` in `research_agent/task.py` |
| Agent tokens | 30,000 | `Task(token_limit=token_budget)` in `research_agent/task.py` |
| Messages | 60 | `Task(message_limit=60)` in `research_agent/task.py` |
| Evaluation attempts, shared across val/test | 3 | `evaluate_candidate` tool in `research_agent/task.py` |
| Candidate sandbox execution | 10 minutes | `timeout=600` in `research_agent/modal_runner.py` |
| CPU input preparation | 1 hour | `prepare_inputs` in `research_agent/modal_runner.py` |
| Each remote evaluation | 1 hour | `timeout=3600` in `research_agent/modal_runner.py` |

The agent runtime starts when Inspect begins the sample. Individual Modal timeouts
are separate limits. Stopping or timing out the local agent does not guarantee
cancellation of an already-running Modal evaluation; monitor it in the Modal
dashboard, where you can stop it if needed. These limits do not impose a dollar cap.

Keep the evaluation terminal open for progress. In another terminal, activate the
same environment, run from the repository root, and open the transcript viewer:

```bash
inspect view --log-dir agent-logs
```

Open the URL printed by Inspect and select the run to inspect messages, tool calls,
feedback and usage. The Modal dashboard shows remote job status and worker output.
For viewers on another machine, add `--log-shared 10` to the evaluation command to
sync live events to the log directory every ten seconds; that directory must be
accessible to the viewer.

| Results location | Contents |
|---|---|
| `agent-runs/<run-id>/` | `prompt.txt`, current `candidate.py`, public repository snapshot, numbered `candidate_001.py`, `algorithm_001.md` and `result_001.json` files |
| Same run directory, after submission or fallback | `submission.py`, `algorithm.md` and `submission.json`; if no evaluation succeeded, only a submission report explaining that no valid candidate exists |
| `agent-logs/` | Inspect transcripts, usage, tool activity and errors |
| Modal volume `lora-merging-results`, under `/agent/<experiment-id>/` | Candidate snapshot, `metadata.json`, `result.json` or `error.json` |
| Modal volume `lora-candidate-io`, under `/<experiment-id>/` | Per-attempt candidate code, public source and exported tensors; retained for now |
| Modal volume `lora-research-inputs`, under `/<cache-key>/` | Reusable fixed model inputs under `public/`; private classifier heads and evaluation configuration outside that subdirectory |

Inspect the numbered local result files as evaluations finish. Their
`experiment_id` links successful local reports to the corresponding remote results.

The agent can submit any exact source that has completed evaluation successfully.
Submission saves its code, matching algorithm explanation and report. If no explicit
submission is made, cleanup selects the best test Avg candidate, or the best
validation Avg if none has test results. Results from different splits are ranked
separately. Without a successful evaluation, no valid candidate is submitted. Abrupt process termination
can prevent cleanup; previously saved attempt files remain available.

To evaluate the frozen submission directly, replace `<run-id>` with its local folder:

```bash
python -m modal run -m research_agent.modal_runner::main \
  --candidate agent-runs/<run-id>/submission.py --split test
```

Both splits return `100 * candidate_accuracy / reference_accuracy` per task, with
Avg their arithmetic mean. The private reference accuracies are configured in
`private_eval/reference_scores.json`, separately for validation and test.
Calibration for the requested split is required before evaluation; there is no
raw-accuracy fallback. Raw accuracies remain in operator reports for auditing.

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
| `research_agent/inputs_cache.py` | Trusted fixed-input serialization, cache validation and invalidation |

Baseline JSON reports are saved in the Modal volume `lora-merging-results`.

## Evaluation integrity

The intended agent workspace exposes baseline implementations and reference
papers, with one editable merging function. Evaluation code and KnOTS internals
remain private; the agent receives an evaluation tool that returns scores.
The agent's read tools use an explicit public source view, and candidate code
runs in a separate sandbox. This README is operator-facing and is excluded from
that view. `research_agent/prompt.md` contains the agent's full interface contract.
The agent starts with an empty `merge(base_model, task_vectors, lora_adapters)`
function in `research_agent/candidate.py`. Public source files are exposed in full
through the `PUBLIC_FILES` allowlist in `research_agent/repository.py`; the result
table and selected paper Markdown are also readable. Private loading, scoring
and reference alignment remain in `private_eval/`.

The sandbox receives the model inputs and public code, while the trusted GPU worker
retains datasets, original classification heads and accuracy computation. Only the
backbone weights identified by task-vector keys are transferred from the returned
model to the evaluator.

The three-task benchmark uses fixed released settings or explicit overrides.
Its scores are not directly equivalent to the original six-task, tuned results.

## Attribution

Baseline and evaluation code is adapted from the
[KnOTS repository](https://github.com/gstoica27/KnOTS), accompanying
[Model Merging with SVD to Tie the KnOTS](https://arxiv.org/abs/2410.19735).
The original license is retained in [LICENSE](LICENSE).
