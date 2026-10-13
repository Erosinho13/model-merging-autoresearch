# Research workspace

Implement `merge(base_model, task_vectors, lora_adapters)` in `candidate.py`. Its initial body is
`pass`. You may add helper functions within this file. Return a `torch.nn.Module`.

```python
def merge(
    base_model: peft.PeftModel,
    task_vectors: list[dict[str, torch.Tensor]],
    lora_adapters: list[dict[str, LoRAFactors]],
) -> torch.nn.Module:
    pass
```

Type annotations document the contract; Python does not enforce them by itself.
The worker checks the returned model type and the evaluator checks exported tensors.

`base_model` is a CPU PEFT sequence-classification model with the reference LoRA
adapter installed. `task_vectors` contains three dictionaries, in SNLI, MNLI,
SciTail order, with the repository's dense B@A task updates minus reference updates.
The dictionary keys identify the corresponding backbone weights. Use
`base_model.state_dict()` to read and update those weights. The supplied reference
adapter remains installed; preserve its structure.

The task-vector dictionaries have identical keys and CPU float32 tensor shapes
matching their backbone weights. `lora_adapters` is a list of three dictionaries
in the same task order and using the same keys. Each entry is a `LoRAFactors`
TypedDict from the public `merging_types.py` module:

- `A: torch.Tensor`, shape `[rank, input_features]`.
- `B: torch.Tensor`, shape `[output_features, rank]`.
- `scaling: float`, the adapter's per-layer effective multiplier (normally alpha/r).

These are copies of the ORIGINAL task adapter factors, not an SVD refactorization
or reference-subtracted factors. The effective task update is `scaling * (B @ A)`.
The supplied base model contains the reference adapter and its own factors.
Dense task vectors preserve the existing repository's `B@A_task - B@A_reference`
construction. Its helper does not multiply by scaling; the released configuration
has alpha=rank=16, so scaling is one. Choose merging hyperparameters inside your code.

Only the weights identified by the task-vector keys are exported from the returned
model. Other weight changes, architecture changes, and classification-head changes
are not transferred. The evaluator installs its own trained task heads.
Factor-based methods should reconstruct their merged update and apply it to these
backbone weights; edits only to the returned model's LoRA factors are not exported.
All exported weights must have their original shapes and finite floating values.

The public repository includes the original Task Arithmetic, TIES, DARE, vector
merging and parameter-handler implementations. `baseline_configs.py` lists their
settings. Averaging uses the sum of the three updates scaled by 1/3.
Candidate code can import these public modules from the sandbox's Python path.
They are references; the candidate cannot change the operator's source files.

Use `read_repository` to list/read repository files and the current candidate.
Use `read_reference` for paper Markdown, `write_candidate` to replace candidate
code, `get_budget` to inspect resources, and `evaluate_candidate` for validation
scores on val or test. Finish with `submit_candidate` after successfully evaluating that exact code.
Each evaluation requires an `explanation` argument containing Markdown about the
algorithm, equations or pseudocode, hyperparameters, rationale and limitations.
It is saved as `algorithm_<attempt>.md` with that candidate. Submission writes
`algorithm.md` alongside `submission.py`; an optional submission explanation can
add validation evidence. Fallback submission uses the selected candidate's saved explanation.
Read baseline_results.md for named baselines and anonymous reference results.
The evaluator returns SNLI, MNLI, SciTail and their arithmetic Avg. Test scores
are reference-relative percentages; validation scores are raw accuracy percentages,
because no validation reference is available. Feedback identifies the score type.
Hyperparameter tuning is optional; if you tune, evaluate choices on val.
Both splits are available and successful evaluation on either allows submission.
Fallback uses the best test Avg, or the best raw validation Avg if no test candidate
succeeded. Test is visible; extra unseen-dataset checks are future work.
