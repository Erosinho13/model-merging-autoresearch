# AI research-agent evaluation: LoRA model merging

This repository evaluates whether an existing LLM agent can develop and test
new methods for merging LoRA-finetuned models. Inspect manages the research loop;
Modal runs candidate code and a fixed NLP evaluator on remote compute.

## Benchmark

- **Model:** Llama-3-8B with rank-16 LoRA adapters.
- **Tasks:** SNLI, MNLI and SciTail, scored using classification accuracy.
- **Baselines:** averaging, Task Arithmetic, TIES, DARE-TIES, KnOTS-TIES, KnOTS-DARE-TIES, CUR-merging (SVD + CUR with TIES), plus the base model and individual task adapters.
- **Primary reference:** KnOTS-TIES, the best measured test Avg in this benchmark.
- **Research feedback:** candidate scores relative to an anonymous reference on validation and test, supplied baseline results, and optional human feedback.
- **Assessment:** performance is scored automatically. Optional human reviews provide separate ratings for innovativeness and logical soundness.

## Getting started

### Local environment

Run from the repository root:

```bash
conda env create --file conda_environment.yaml
conda activate lora-mma
```

`requirements-modal.txt` supplies the remote evaluator's
dependencies, which Modal installs automatically.

### Modal and Hugging Face access

Create a [Modal account](https://modal.com) and authenticate the local CLI:

```bash
python -m modal token new
```

This opens a browser login and stores Modal credentials locally.
See the [Modal setup guide](https://modal.com/docs/guide).

On Hugging Face, obtain access to
[`meta-llama/Meta-Llama-3-8B`](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
and create a read token that can access it. In the Modal dashboard, create a
secret named **`huggingface`**, containing a field named **`HF_TOKEN`** with that token.

Then, you can run the following commands:

```bash
# Cache model weights on CPU.
python -m modal run modal_app.py --download

# Download and check datasets, tokenizer access and adapter metadata on CPU.
python -m modal run modal_diagnostics.py --prepare

# Evaluate a baseline
python -m modal run modal_app.py --method task_arithmetic --split [val|test]

# Optional GPU check.
python -m modal run modal_diagnostics.py
```

Cloud commands incur compute charges. Evaluation defaults to test if `--split`
is omitted.

## Running and monitoring the research agent

### Agent tools

| Tool | Purpose |
|---|---|
| `read_repository` | Read public repository files and the current candidate code. |
| `read_reference` | Read the supplied papers in Markdown. |
| `read_figure` | View figures from the supplied papers. |
| `write_candidate` | Replace the candidate code, checking its syntax and fixed function signature. |
| `evaluate_candidate` | Run the candidate on validation or test and return three relative task scores and their average. |
| `get_budget` | Check remaining tokens, time, evaluations and human reviews. |
| `human_scorer` | Ask for human ratings of innovativeness and logical soundness, with optional comments. Available when `max_human_reviews > 0`; waiting pauses the timer. |
| `submit_candidate` | Submit a successfully evaluated candidate and its Markdown explanation. |

Place selected paper Markdown in `research_agent/references/` or its paper
subfolders. See [reference preparation](research_agent/references/README.md)
for PDF conversion. Proposed references are:

- Task Arithmetic: [Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089).
- TIES: [TIES-Merging: Resolving Interference When Merging Models](https://arxiv.org/abs/2306.01708).
- DARE: [Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch](https://arxiv.org/abs/2311.03099).
- Fisher merging: [Merging Models with Fisher-Weighted Averaging](https://arxiv.org/abs/2111.09832).
- RegMean: [Dataless Knowledge Fusion by Merging Weights of Language Models](https://arxiv.org/abs/2212.09849).
- TALL-masks / Consensus Merging: [Localizing Task Information for Improved Model Merging and Compression](https://arxiv.org/abs/2405.07813).
- [Knowledge is a Region in Weight Space for Fine-tuned Language Models](https://aclanthology.org/2023.findings-emnlp.95/).

For OpenAI agents, create an API key in your [API account](https://platform.openai.com/api-keys)
and ensure the account has billing enabled and access to your chosen model. Set
the key in the terminal that will launch Inspect:

```bash
export OPENAI_API_KEY="your_api_key"
```

The example below uses the API model identifier used in this project's runs.
To use another provider/model, install the provider's SDK, set its documented
credentials and change `--model`.

```bash
python -m inspect_ai eval research_agent/task.py \
  --model openai/gpt-6-astra \
  --reasoning-effort high \
  -T max_evaluations=3 \
  -T token_budget=1000000 \
  -T time_limit=3600 \
  --log-dir agent-logs
```

This example allows one million tokens, three evaluations (either on validation or test), and 1h total time.
Enable the agent to request `n` human judgments by adding `-T max_human_reviews=n`.
The default is zero, so unattended runs keep their existing behavior. When the
agent calls `human_scorer`, answer in the running Inspect terminal's input form:
give separate 0–10 innovativeness and logical-reasoning scores, and optionally comments.
The form shows the proposal and paths to its code snapshot and recorded evidence.
During human review, the timer is paused so it does not count against the agent's total available time.

In another terminal, you can activate the
same environment, run from the repository root, and open the transcript viewer:

```bash
inspect view --log-dir agent-logs
```

Open the URL printed by Inspect and select the run to inspect messages, tool calls,
feedback and usage. The Modal dashboard shows remote job status and worker output.

Inspect transcripts are saved in `agent-logs`; candidate code, evaluation results and final submissions are saved in `agent-runs`.

For information about the research requirements, check the [prompt.md](research_agent/prompt.md) file, which contains the instructions submitted to the agent.
Placeholders enclosed in double curly braces, such as `{{token_budget}}`, are automatically replaced with the run's argument values; `{{initial_candidate}}` is replaced with the initial code from `research_agent/candidate.py`.

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

## What runs where

| Location | Work performed |
|---|---|
| Your computer | Inspect runs the agent loop, sends requests to the LLM provider, exposes tools, collects human feedback and saves the conversation and candidate snapshots. |
| LLM provider | The research model reads the supplied material and proposes code. Requests are made using your local provider API key. |
| Modal CPU functions | Download checkpoints and prepare reusable model inputs. The optional dataset preparation check also downloads and checks the datasets. |
| Modal CPU sandbox | Executes each candidate merging function with eight CPUs, 96 GiB RAM and a ten-minute timeout. It has public code and read-only model inputs, without datasets, grading code, credentials or network access. |
| Modal GPU function | Loads the exported backbone weights and evaluates SNLI, MNLI and SciTail with their original classification heads. Both baseline and candidate evaluation use an A100-40GB by default. |

### Baseline methods and settings

Supported `--method` values are `average`, `task_arithmetic`, `ties`, `dare_ties`,
`knots_ties`, `knots_dare_ties`, `cur_merging`, `base`, `snli`, `mnli` and `scitail`.
The last four evaluate an unmerged backbone or task adapter with the same evaluator.

The (non-normalized) accuracy results for each method, on the validation and test sets, are shown below.

#### Test set

| Method | SNLI | MNLI | SciTail | Avg |
|---|---:|---:|---:|---:|
| base | 41.71 | 34.60 | 60.72 | 45.68 |
| snli | 92.50 | 68.50 | 81.42 | 80.81 |
| mnli | 88.79 | 90.45 | 85.70 | 88.31 |
| scitail | 50.98 | 47.08 | 96.52 | 64.86 |
| avg | 90.92 | 81.90 | 93.98 | 88.93 |
| task arithmetic | 91.28 | 89.17 | 91.67 | 90.71 |
| TIES | 91.27 | 89.57 | 90.97 | 90.60 |
| DARE-TIES | 91.06 | 83.02 | 93.79 | 89.29 |
| **KnOTS-TIES** | 90.82 | 88.21 | 94.17 | **91.07** |
| KnOTS-DARE-TIES | 90.29 | 82.72 | 93.93 | 88.98 |
| CUR-merging-TIES | 91.54 | 89.94 | 86.92 | 89.47 |

#### Validation set

| Method | SNLI | MNLI | SciTail | Avg |
|---|---:|---:|---:|---:|
| base | 41.59 | 36.78 | 60.89 | 46.42 |
| snli | 92.89 | 69.84 | 80.37 | 81.03 |
| mnli | 89.43 | 89.51 | 88.96 | 89.30 |
| scitail | 50.46 | 48.85 | 97.62 | 65.64 |
| avg | 91.38 | 81.76 | 94.71 | 89.28 |
| task arithmetic | 91.71 | 88.13 | 95.09 | 91.64 |
| TIES | 91.71 | 88.23 | 94.63 | 91.52 |
| DARE-TIES | 91.65 | 82.78 | 94.86 | 89.76 |
| **KnOTS-TIES** | 91.19 | 87.42 | 96.40 | **91.67** |
| KnOTS-DARE-TIES | 90.66 | 82.73 | 94.79 | 89.39 |
| CUR-merging-TIES | 92.15 | 88.69 | 90.41 | 90.42 |

## Attribution

Baseline and evaluation code is adapted from the
[KnOTS repository](https://github.com/gstoica27/KnOTS), accompanying
[Model Merging with SVD to Tie the KnOTS](https://arxiv.org/abs/2410.19735).
The original license is retained in [LICENSE](LICENSE).
