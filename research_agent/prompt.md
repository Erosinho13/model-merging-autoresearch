Research a new method for merging three released Llama-3-8B LoRA task models:
SNLI, MNLI and SciTail. Maximize mean validation accuracy across the three tasks.
This is the initial integration objective; a fixed KnOTS normalization reference
will be added once it has been measured on the same splits.

Use the supplied reference documents where available. Implement your method with
write_candidate, experiment with evaluate_candidate, then submit_candidate.
Only validation scores are available. Do not seek test data or change the grader.
Evaluate the initial averaging candidate first to verify the integration.

Interface: merge(base_weights, task_vectors) -> dict[str, torch.Tensor].
base_weights contains CPU weights for LoRA-targeted backbone layers only.
task_vectors is a list of three dictionaries in SNLI, MNLI, SciTail order,
with identical keys and shapes. They use the existing repository's dense B@A
task-vector construction and reference subtraction, exactly as the baselines do.
Return absolute merged weights for exactly these keys, with unchanged shapes and
finite values. Classification heads are excluded and installed by the grader.
The grader constructs the final model; the function does not load a model.
Python, PyTorch 2.2.2, NumPy 1.26.4 and safetensors 0.4.1 are available.
CPU execution only, eight CPUs, 64 GiB RAM, ten-minute merging timeout.
You may define helper functions and import the installed libraries.
The candidate executes without network, datasets, evaluator source or secrets.

Each evaluation consumes an experiment, including failed attempts. Code snapshots
and results are saved. Syntax/signature checks do not consume GPU experiments.
Submit the exact code of a successfully evaluated candidate, then explain the
method and evidence briefly. Do not claim test performance or novelty as proven.
