# Research task: develop a LoRA model-merging algorithm

Develop and implement a principled algorithm for merging three LoRA-finetuned
Llama-3-8B models into one backbone. The adapters were trained on SNLI, MNLI and
SciTail. Aim to improve on the available merging baselines, while explaining what
is new and why it should help. Hyperparameter tuning and combinations of known
methods are allowed experiments, but do not by themselves establish novelty.

## Objective and available evidence

The evaluation returns SNLI, MNLI, SciTail ($T=3$ tasks) and Avg. Maximize the test Avg;
improvements across individual tasks are also desirable.
Read baseline_results.md through read_repository: it
lists named baselines and the anonymous reference's scores. The reference's
identity, implementation and hyperparameters remain private. Do not infer an
innovation grade from performance; novelty is not automatically scored.

On test, each task's score is its accuracy relative to the reference algorithm:

$score_t = 100 \times \frac{\text{candidate accuracy}_t}{\text{reference accuracy}_t}$

$Avg = \frac{1}{T}\sum_{t=1}^T(score_t)$

100 means parity with the reference; higher is better and scores are not capped.
The private test reference accuracies supplied by
the operator are rounded to two decimals. The baseline table is display-rounded.

Both val and test are available. Test returns four reference-relative percentages.
Validation returns three raw accuracy percentages and their arithmetic mean;
validation reference accuracies are unavailable. Feedback labels its score_type.
Do not compare validation raw accuracy with test relative scores as the same metric.
Hyperparameter tuning is optional. If you compare or choose hyperparameters, you
must evaluate them on val and base those choices on validation evidence. You may
use test feedback for research and benchmark comparison; it is not an unseen holdout.
Submission requires a successful evaluation of the exact code, on either split.
Fallback selects the highest test Avg if any test candidates succeeded, otherwise
the highest raw validation Avg. It never mixes the two scoring scales.
No additional unseen-dataset evaluation is scheduled in this version.

Public code and settings are supplied through read_repository. There is no direct
baseline-evaluation tool. You may reuse a baseline inside your candidate, but
evaluating it consumes an ordinary attempt. Named baselines are distinguishable
from the anonymous reference; you can compare their per-task and aggregate scores.

## Interface and permitted changes

Read README.md through read_repository first; it is the public interface guide.
The initial candidate.py contains an empty function. Implement this exact signature:

```python
def merge(
    base_model: peft.PeftModel,
    task_vectors: list[dict[str, torch.Tensor]],
    lora_adapters: list[dict[str, LoRAFactors]],
) -> torch.nn.Module:
    pass
```

LoRAFactors is defined in merging_types.py. Each layer entry contains the original
A and B tensors and its effective scaling multiplier. Inputs are on CPU; task
vectors are dense float32 matrices, not flattened vectors. Both input lists use
SNLI, MNLI, SciTail order and matching layer keys. Read the guide for factor
shapes, reference subtraction and the distinction between dense updates and
scaled B@A products. The base model includes the reference adapter.

You may replace candidate.py, add helper functions in it, and import installed
libraries or public baseline modules. Return one merged model that is evaluated
on all three tasks. Only the backbone weights identified by the task-vector keys
are exported; write the resulting merged updates into those weights. Changes only
to LoRA factor parameters, other weights, architecture or classifier heads are
not transferred. The original shapes and finite floating values are required.

The research scope is the merging rule applied to the supplied weights and factors.
Use one general rule across tasks. Layer- or adapter-dependent quantities derived
from the tensors are allowed; dataset-name-specific hacks are outside scope.
Do not train on additional data, change evaluation behavior, attempt to discover
private files or credentials, or fabricate evaluation reports. Dataset loaders,
trained classification heads and scoring code remain under operator control.

## Tools and execution

- read_repository: list/read the public repository and current candidate; read-only.
- read_reference: list/read supplied paper Markdown, including paper subfolders.
  Both reading tools accept start, lines (default 200), and max_chars (default
  20000). Choose positive limits to match your remaining token budget; neither
  limit has a fixed upper cap. Characters are not tokens. Start with focused
  excerpts, expand when useful, and check get_budget before large reads. The
  character limit applies to the excerpt, excluding its header and truncation notice.
- write_candidate: replace candidate.py and check syntax and signature.
- get_budget: inspect cumulative tokens and remaining evaluation attempts.
- evaluate_candidate(explanation=..., split="val" or "test"): run the candidate
  and receive four split-specific scores or an error, together with budget status.
- submit_candidate(explanation=...): freeze a successfully evaluated candidate;
  the optional explanation can update its write-up with observed evidence.

There is no general shell, independent code-execution, internet or image-reading
tool. To execute your candidate, use evaluate_candidate. It runs inside a separate
CPU sandbox without datasets, credentials or evaluator source. Paper captions and
Markdown are readable; figure images are not directly viewable through these tools.

The candidate sandbox has eight CPUs, 96 GiB RAM and a ten-minute timeout.
Installed libraries include PyTorch 2.2.2, Transformers 4.40.2, PEFT 0.3.0,
Accelerate 0.26.1, NumPy 1.26.4 and safetensors 0.4.1. No GPU is provided to candidate
code. The trusted evaluator handles GPU inference and has a one-hour timeout.

## Experiment discipline

Use the supplied papers and baseline implementations to identify a concrete
limitation and a testable hypothesis. Before each evaluation, explain the proposed
mechanism and expected effect; expected improvements are hypotheses, not results.
Check dimensions and API usage while writing code. The empty pass stub and invalid
signatures are rejected locally without spending an experiment; remote failures
consume attempts. Syntax checks do not verify numerical correctness.

Prefer experiments that distinguish explanations of a result. When the budget
permits, use a focused ablation instead of adding several untested components.
Compare per-task results as well as the average, and document regressions. Do not
rerun unchanged stochastic candidates with different seeds just to select lucky
scores. If your method uses randomness, declare and fix its seed in the code.
Hyperparameter choices and tensor-derived weighting rules must be reproducible.

Use get_budget and returned budget status to plan remaining work. Tokens count
cumulatively across calls, including repeated conversation history. Reserve enough
for submission; stop when further experiments are unlikely to be informative or
would prevent a valid submission. The numerical budgets follow below.

## Required artifacts and honest reporting

Every evaluate_candidate call requires a nonempty Markdown explanation of at most
20 KB. Include the algorithm, equations or pseudocode, hyperparameters, relationship
to prior work, rationale and limitations. The runner saves it with that exact code
snapshot as algorithm_<attempt>.md, including for failed remote attempts.

Choose your best successfully evaluated candidate, restore its exact source with
write_candidate if needed, and call submit_candidate. In the final explanation,
identify the matching experiment, report only tool-observed scores, labeling val and test separately,
state which components were actually evaluated, and distinguish reuse, tuning and
proposed new contributions. Disclose negative results and selection among attempts.
Do not describe the visible test results as unseen holdout performance, or claim
broad generalization or proven novelty.

Submission saves submission.py, algorithm.md and submission.json. If you omit the
submission explanation, the saved evaluation explanation is reused. If the run
ends without explicit submission, cleanup selects the best successfully evaluated
candidate by test Avg (or raw validation Avg if none has test results) and retains its matching explanation. With
no successful evaluation, there is no valid submission. Do not rely on fallback
selection to replace a deliberate, well-documented final submission.

## Run resources and initial candidate

GPU evaluation limit: {{max_evaluations}}.
Total token budget for this research run: {{token_budget}}.
This is cumulative input and output usage, including provider-reported reasoning
tokens. Conversation history consumes input tokens again on subsequent calls.
Use get_budget to check remaining resources; evaluation feedback also includes them.
Reserve tokens for submission and final explanation; another full call may not fit
even when some tokens remain.

Initial candidate:
```python
{{initial_candidate}}
```
