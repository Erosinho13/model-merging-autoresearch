"""Trusted bridge: reuse baseline task vectors, heads, loaders and metrics."""
import gc
from pathlib import Path
from time import monotonic


def evaluate(source, output_dir, execute_candidate, split="val"):
    import torch
    from ft_handlers import LoRAHandler
    from task_merger import TaskMerger
    from utils import prepare_llama, set_seed
    from eval_scripts.nli_fixed_test import fixed_config, evaluate_tasks, progress
    from configs.nlp_benchmark import TASKS

    started = monotonic()
    set_seed(420)
    raw = fixed_config("average")
    progress("Loading baseline models for candidate evaluation")
    models = prepare_llama(raw["model"], "cpu")
    heads = {}
    for task, model in zip(TASKS, models["bases"]):
        matches = [p.detach().cpu().clone() for n, p in model.named_parameters()
                   if "score.modules_to_save.default.weight" in n]
        if len(matches) != 1 or matches[0].shape != (3, 4096):
            raise RuntimeError(f"Invalid head for {task}")
        heads[task] = matches[0]
    reference = LoRAHandler(models["new"]).get_ft_parameters()
    parameters = [LoRAHandler(model).get_ft_parameters() for model in models["bases"]]
    helper = TaskMerger(models["bases"], models["new"], LoRAHandler, device="cpu")
    vectors = helper.get_task_directions(reference, parameters)
    model = models["new"]
    state = model.state_dict()
    base = {key: state[key].detach().cpu().clone() for key in reference}
    del helper, parameters, reference, models
    gc.collect()
    progress("Running candidate in isolated CPU sandbox")
    merged = execute_candidate(source, base, vectors)
    if set(merged) != set(base):
        raise ValueError("Candidate must return exactly the supplied weight keys")
    with torch.no_grad():
        for key, original in base.items():
            value = merged[key]
            if value.shape != original.shape or not value.is_floating_point():
                raise ValueError(f"Invalid shape or dtype: {key}")
            value = value.to(original.dtype)
            if not torch.isfinite(value).all():
                raise ValueError(f"Invalid shape or nonfinite weights: {key}")
            state[key].copy_(value)
    del base, vectors, merged, state
    gc.collect()
    model.config.pad_token_id = 128001
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.to("cuda").eval()
    report = {"method": "candidate", "split": split, "seed": 420, "tasks": {},
              "head_source": "released task adapters", "merged": True,
              "dtype": str(next(model.parameters()).dtype)}
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return evaluate_tasks(model, heads, raw, report, "candidate", output_dir, 420, started)
