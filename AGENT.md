The agent uses Inspect tools to call our existing KnOTS-based validation pipeline.
No baseline scripts are modified. Candidate code runs in a separate Modal CPU
sandbox with only the candidate and tensor inputs. The trusted GPU worker keeps
the original heads, data loaders and accuracy computation.

First run the averaging integration check (this spends GPU and CPU compute):

```bash
python -m modal run -m research_agent.modal_runner::main --candidate research_agent/average.py
```

Compare its validation scores against the existing averaging baseline on validation.
The sandbox runs while the GPU worker remains allocated; this first version favors
simple integration over minimum compute cost. Tensor exchange also adds overhead.

Put selected paper Markdown files in `research_agent/references/`. Start one agent
run using an API model identifier available in your provider account:

```bash
inspect eval research_agent/task.py --model <provider/model> -T max_evaluations=3 -T token_budget=30000 --log-dir agent-logs
```

Set your provider API credentials locally before running Inspect. A Codex model
label is not necessarily an API model identifier. Model/API spend is separate
from Modal spend; token and experiment limits do not guarantee a dollar ceiling.

Local code snapshots, errors, results and submission live in `agent-runs/<run-id>/`.
Inspect transcripts live in `agent-logs/`. Remote reports and immutable candidate
snapshots live in `lora-merging-results` under `/agent/<experiment-id>/`.
The separate `lora-candidate-io` volume holds tensor exchange files; these can be
large and are retained for now. It has no datasets or evaluator files.

Final test evaluation is an operator action, never an agent tool:

```bash
python -m modal run -m research_agent.modal_runner::main --candidate agent-runs/<run-id>/submission.py --split test
```

Current objective is mean raw validation accuracy. KnOTS-relative scoring, paper
ingestion and automatic held-out final scoring are subsequent steps. If the run
ends before submission, there is no final candidate: inspect the saved experiments.
