# Task brief: develop a novel algorithm for merging LoRA adapters

## Research goal

You are conducting AI research. Develop and implement a novel LoRA model-merging
algorithm, supported by a clear technical contribution and reproducible evidence.
Aim for a result strong enough to form the basis of an excellent research paper.

Your method must be general-purpose: it should apply to other tasks and collections
of compatible adapters, rather than depend on the particular benchmark below.
Explain the problem it addresses, the new insight, and why the proposed mechanism
should help. State its assumptions and distinguish intended applicability from
experimentally demonstrated generalization.

For this task, assess novelty relative to the field at the start of 2026. Similarity
to work introduced later does not automatically disqualify a proposal. Explain its
relationship to the supplied papers and other prior work you know. Performance is
scored automatically; algorithmic novelty is a research requirement, not a separate
automated grade.

## Research task, data, metrics and baselines

### Experimental testbed

The repository supplies three LoRA-finetuned Llama-3-8B models, trained on SNLI,
MNLI and SciTail. Your algorithm combines their updates into one backbone, which
is evaluated on all three tasks with the corresponding fixed classification heads.
The current experiment uses three rank-16 adapters. This setup is a testbed for
your research, not a restriction on the intended applicability of the algorithm.

### Metrics

The evaluation tool returns SNLI, MNLI, SciTail ($T = 3$ tasks) and Avg.
The raw metric used for all these tasks is accuracy percentage. On the test split,
each task final score is its accuracy relative to an anonymous reference algorithm:

$score_t = 100 \times \frac{\text{candidate accuracy}_t}{\text{reference accuracy}_t}$

$Avg = \frac{1}{T}\sum_{t=1}^T(score_t)$

Maximize test Avg; improvements on individual tasks are also desirable. A score
of 100 means parity with the reference. Higher is better, and scores are not capped.
The private reference accuracies come from the operator's measurements rounded
to two decimals. The public results table is also rounded for display.

Both validation and test feedback are available. Validation returns raw accuracy
percentages and their arithmetic mean. Test returns the reference-relative
percentages defined above.
Feedback identifies the split and score_type; do not compare raw validation and
relative test values as if they were the same metric. Test feedback is visible
during research. Additional unseen-dataset checks are not part of this run.

### Baselines and reference results

Read baseline_results.md through read_repository for measured test results.
Named methods include averaging, Task Arithmetic, TIES and DARE-TIES; their
implementations and settings are public. The anonymous reference's identity,
implementation and settings are withheld, while its scores are visible.
Use these comparisons to assess practical improvement. Benchmark gains alone
do not establish novelty or generalization.

## Scope

### Your algorithm and editable code

Your contribution is a novel merging algorithm, specifically designed for LoRA
adapters, applied to the supplied model weights and adapter information.
Implement it in candidate.py, keeping the fixed merge
signature. You may add helpers in that file and import installed libraries or
public baseline modules. The operator's repository files remain read-only.

Dataset-specific branches, manually selecting a different method for each task,
and changes to evaluation behavior are strictly forbidden.

### Hyperparameters and reuse of existing methods

Hyperparameter tuning is optional. If you compare or choose hyperparameters,
evaluate the alternatives on the validation split and base those choices on
validation evidence. Test feedback may guide research and benchmark comparison.
Tuning a known method alone does not meet the innovation objective. When combining
known methods, explain the additional mechanism or insight and support its value
with experiments where the budget permits.

### Fixed infrastructure

The evaluator controls datasets, preprocessing, trained classification heads and
scoring. Candidate code cannot access or modify them. Do not introduce extra
training data, attempt to access private files or credentials, or fabricate results.

The evaluator receives only the backbone weights identified by the task-vector keys.
Write your final merged weights into those tensors. Changes to LoRA factors, other
parameters, classifier heads or model architecture are not exported. Preserve the
exported tensors' original shapes and finite floating-point values. Factor-based
methods must reconstruct their merged update and apply it to these backbone weights.

## Task details and function interface

The complete input contract is given here. The editable candidate.py starts with
an empty function. Implement this exact signature:

```python
def merge(
    base_model: peft.PeftModel,
    task_vectors: list[dict[str, torch.Tensor]],
    lora_adapters: list[dict[str, LoRAFactors]],
) -> torch.nn.Module:
    pass
```

Type annotations describe the contract; Python does not enforce them by itself.
The worker checks that the returned value is a torch.nn.Module, and the evaluator
validates the exported weights.

When evaluation runs, the sandbox supplies these inputs:

- base_model: a CPU PEFT sequence-classification model with the repository's reference
  adapter installed.
  This input adapter is distinct from the anonymous reference algorithm used for
  scoring. Use base_model.state_dict() to read and update the backbone weights.
  Preserve the installed adapter's structure; it remains active during evaluation.
- task_vectors: dense CPU float32 task updates, keyed by backbone weight names.
  The repository constructs them as B@A_task minus B@A_reference. They are matrices,
  not flattened vectors. All dictionaries have identical keys; each tensor's shape
  matches its corresponding backbone weight.
- lora_adapters: copies of the original task adapter factors on CPU, not
  reference-subtracted factors or an SVD refactorization. The dictionaries use the
  same keys as task_vectors. Each entry is a LoRAFactors TypedDict defined in
  merging_types.py: A is a torch.Tensor of shape [rank, input_features], B is a
  torch.Tensor of shape [output_features, rank], and scaling is a float (normally
  alpha/r). The effective adapter update is scaling * (B @ A).

Use the original LoRA factors, dense task vectors, or both as your method requires.
Both lists follow SNLI, MNLI, SciTail order in this experiment. The dense-update
helper does not apply scaling; the supplied adapters have alpha=rank=16, so their
scaling is one. Consult the public utilities before mixing dense
updates and original factors or applying updates to the base model.

Initial candidate:
```python
{{initial_candidate}}
```

## Workspace and available resources

You have access to the following through the provided tools:

- Public repository source: merging_functions.py, masking_ops.py, ft_handlers.py,
  task_merger.py, utils.py, baseline_configs.py and merging_types.py.
- baseline_results.md, the measured test-score table with named baselines and the
  anonymous reference. The merging interface is documented in this prompt.
- Reference papers in Markdown, including captions. Figure images cannot be
  viewed through the reading tools.
- candidate.py, the only editable source file, and the evaluation and budget tools.

baseline_configs.py contains the public methods' settings. In the current three-adapter
benchmark, averaging sums the updates with coefficient 1/3. Candidate code can
import the listed public modules directly from the sandbox's Python path.

The evaluation tool supplies model inputs inside the sandbox. Reading tools expose
code and papers, not model checkpoints. Evaluator source, dataset loaders, private
reference code and settings, and operator infrastructure are hidden.
There is no direct baseline-running tool. Reusing a public baseline inside your
candidate is allowed; evaluating it consumes an ordinary evaluation attempt.

## How work runs

### Tools

- read_repository: list or read public files and the current candidate.
- read_reference: list or read supplied paper Markdown.
- write_candidate: replace candidate.py and check syntax and the merge signature.
- get_budget: report token usage, remaining evaluation attempts and remaining
  wall-clock time in seconds, using the same timer that enforces the run limit.
- evaluate_candidate(explanation=..., split="val" or "test"): execute the current
  candidate and return four split-specific scores or an error, plus budget status.
  The default split is val.
- submit_candidate(explanation=...): freeze successfully evaluated code and its
  Markdown explanation. An optional explanation updates the final write-up.

Both reading tools accept start, lines (default 200) and max_chars (default 20000).
Choose positive limits to fit your remaining budget; there is no fixed upper cap.
Characters are not tokens. Read focused excerpts first and expand when useful.
The character limit excludes the header and any truncation notice.

### Execution environment

To execute generated code, call evaluate_candidate. It runs your merging function
in a separate CPU sandbox without datasets, credentials, evaluator source or
network access. The trusted evaluator loads the exported weights and performs
GPU inference outside that sandbox. No general shell or independent execution
tool is available to the agent.

The candidate sandbox has eight CPUs, 96 GiB RAM and a ten-minute timeout.
Installed libraries include PyTorch 2.2.2, Transformers 4.40.2, PEFT 0.3.0,
Accelerate 0.26.1, NumPy 1.26.4 and safetensors 0.4.1. Candidate code has no GPU.
The trusted GPU evaluator has a one-hour timeout.

### Budgets

GPU evaluation limit: {{max_evaluations}}.
Total token budget for this research run: {{token_budget}}.
Agent wall-clock time limit: {{time_limit}} seconds (including model calls and
evaluation waits). This limit is enforced by Inspect for this research run.

Tokens accumulate across calls, including input, output and provider-reported
reasoning tokens. Conversation history consumes input tokens again on later calls.
Use get_budget and evaluation feedback to check remaining tokens, time and attempts.
Reserve time and tokens for submission and the final explanation; another model
call or evaluation may not finish within the remaining budget. The timer starts
when Inspect begins the sample, not when the task files are constructed. The live
timer reported by get_budget takes precedence if the operator overrides the limit.

### Experiments and evidence

Use papers and baseline code to identify a limitation and a testable hypothesis.
Before each evaluation, explain your algorithm, its intended mechanism and expected
effect. Expectations are hypotheses, not measured results. Check shapes and APIs;
local syntax checks do not verify numerical correctness. Empty implementations and
invalid signatures are rejected before spending an attempt. Remote failures consume
an attempt, as do evaluations on either split.

Prefer experiments that distinguish competing explanations. Where budget permits,
use focused ablations and document per-task regressions as well as average gains.
Declare and fix random seeds; do not rerun unchanged stochastic candidates with
different seeds just to select lucky scores. Make tensor-derived rules and chosen
hyperparameters reproducible. Stop experimenting in time to submit valid work.

## Submitting

Submit the implemented merge function and a Markdown explanation of the algorithm
and its experimental evidence. Every evaluation requires a nonempty explanation
of at most 20 KB. Include equations or pseudocode, hyperparameters, relationship
to prior work, rationale and limitations. The runner saves each explanation with
its exact code snapshot as algorithm_<attempt>.md, including failed remote attempts.

Restore your chosen successfully evaluated source using write_candidate if needed,
then call submit_candidate. Evaluation must have succeeded for that exact source
on either split. Submission saves submission.py, algorithm.md and submission.json.
If you omit the final explanation argument, the evaluated explanation is reused.

In your write-up and final response, identify the supporting experiments and
report only observed results, with validation and test clearly labeled. Explain
which components were evaluated, what is reused, and what contribution you propose.
Disclose failed ideas, regressions and selection among attempts. State limitations
of the evidence; do not claim that visible test results demonstrate unseen-task
generalization or prove novelty.

If the run ends without explicit submission, cleanup selects the highest test Avg
among successfully evaluated candidates. If none has test results, it uses the
highest raw validation Avg. The selected code keeps its matching explanation;
raw and relative scoring scales are never mixed. Without a successful evaluation,
there is no valid submission. Use explicit submission to record your intended result.
