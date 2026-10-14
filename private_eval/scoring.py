"""Trusted reference-relative scoring; no reference identity in feedback."""
import json
import math
import os
from pathlib import Path

TASKS = ("snli", "mnli", "scitail")
ROOT = Path(os.environ.get("LORA_REPOSITORY_ROOT", str(Path(__file__).resolve().parent.parent)))
REFERENCES = ROOT / "private_eval" / "reference_scores.json"


def load_reference(split):
    if split not in {"val", "test"}:
        raise ValueError("split must be val or test")
    reference = json.loads(REFERENCES.read_text()).get(split)
    if not reference:
        raise ValueError(f"Reference scores for {split} are not configured; supply anonymous-reference accuracies before evaluation")
    values = reference["reference_accuracy_percent"]
    if set(values) != set(TASKS) or any(not math.isfinite(v) or not 0 < v <= 100 for v in values.values()):
        raise ValueError(f"Invalid {split} reference calibration")
    return reference


def relative_scores(report, reference):
    """Reference-relative percentages using the requested split's calibration."""
    if reference is None:
        raise ValueError("Reference calibration is required; raw-accuracy feedback is disabled")
    accuracies = {task: report["tasks"][task]["accuracy_percent"] for task in TASKS}
    if any(not math.isfinite(v) or not 0 <= v <= 100 for v in accuracies.values()):
        raise ValueError("Invalid candidate accuracies")
    scores = {
        task: 100 * (accuracies[task] / reference["reference_accuracy_percent"][task])
        for task in TASKS
    }
    scores["avg"] = sum(scores.values()) / len(TASKS)
    return scores
