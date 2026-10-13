The agent uses Inspect tools to call our existing KnOTS-based validation pipeline.
No baseline scripts are modified. Candidate code runs in a separate Modal CPU
sandbox with the candidate, model inputs and public baseline source. The trusted GPU worker keeps
the original heads, data loaders and accuracy computation.

First run the averaging integration check (this spends GPU and CPU compute):

```bash
python -m modal run -m research_agent.modal_runner::main --candidate research_agent/examples/average.py
```

Compare its validation scores against the existing averaging baseline on validation.
The sandbox runs while the GPU worker remains allocated; this first version favors
simple integration over minimum compute cost. Tensor exchange also adds overhead.

The agent starts from `research_agent/candidate.py`, an empty
`merge(base_model, task_vectors, lora_adapters)` function that must return a model. Read-only
repository access uses an explicit source view from `research_agent/repository.py`;
operator files, dataset loaders, evaluator code and KnOTS internals are excluded.
Public Python files are whole files listed in `PUBLIC_FILES`. Private loading,
scoring and KnOTS alignment live in `private_eval/`; private configuration and
dataset folders are also excluded. `baseline_configs.py` provides the live public
baseline settings. Markdown continues to use the selected guide and paper discovery.
The root README is operator-only; `research_agent/interface.md` supplies the public guide.
Only LoRA-targeted weights from the returned model are transferred to the grader.

Put selected paper Markdown files in `research_agent/references/` or its paper subfolders. Start one agent
run using an API model identifier available in your provider account:

```bash
inspect eval research_agent/task.py --model <provider/model> -T max_evaluations=3 -T token_budget=30000 --log-dir agent-logs
```

Set your provider API credentials locally before running Inspect. A Codex model
label is not necessarily an API model identifier. Model/API spend is separate
from Modal spend; token and experiment limits do not guarantee a dollar ceiling.

Local code snapshots, errors, results and submission live in `agent-runs/<run-id>/`.
Each experiment includes `algorithm_<attempt>.md`. Final submission includes
`algorithm.md`, matched to the selected code, for explicit and fallback selection.
Inspect transcripts live in `agent-logs/`. Remote reports and immutable candidate
snapshots live in `lora-merging-results` under `/agent/<experiment-id>/`.
The separate `lora-candidate-io` volume holds tensor exchange files; these can be
large and are retained for now. It has no datasets or evaluator files.

Both val and test are available through the agent tool; select hyperparameters on val.
An operator can also evaluate the frozen candidate directly:

```bash
python -m modal run -m research_agent.modal_runner::main --candidate agent-runs/<run-id>/submission.py --split test
```

Test feedback contains three reference-relative percentages and their arithmetic
Avg. Private test calibration is in private_eval/reference_scores.json. Validation
feedback contains raw accuracies and their arithmetic mean; no validation reference
is needed. Hyperparameter tuning is optional, but tuning choices must use val.
Additional unseen-dataset checks are future work. Cleanup selects the best test Avg
candidate if available, otherwise the best raw validation Avg; its selection label
records the split. Explicit submissions are preserved. With no successful evaluation,
there is no valid candidate.
Abrupt process termination can prevent cleanup; saved experiment snapshots remain.
Reconstructing a full model in the sandbox increases tensor-exchange overhead;
the sandbox uses 96 GiB CPU memory. The revised full-size path still needs a Modal check.
