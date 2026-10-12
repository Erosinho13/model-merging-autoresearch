"""Shared task selection for the reduced NLP benchmark."""

TASKS = ("snli", "mnli", "scitail")
ADAPTERS = tuple(f"hoffman-lab/KnOTS-Llama3_8B_lora_R16_{task}" for task in TASKS)
DATASETS = (
    {"name": "snli", "mask_class": None},
    {"name": "mnli", "val_fraction": 0.2, "mask_class": None},
    {"name": "scitail", "mask_class": 2},
)


def select_benchmark(config):
    """Select matching datasets and adapters before any data/model loading."""
    from copy import deepcopy

    selected = deepcopy(config)
    selected["dataset"] = deepcopy(list(DATASETS))
    selected["model"]["bases"] = list(ADAPTERS)
    return selected
