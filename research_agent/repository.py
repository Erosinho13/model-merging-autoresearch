"""Construct an explicit public source view; never expose the operator tree."""
from pathlib import Path
import os

ROOT = Path(os.environ.get("LORA_REPOSITORY_ROOT", str(Path(__file__).resolve().parent.parent)))


# These Python files are exposed in their entirety. No class/function extraction.
PUBLIC_FILES = (
    "merging_functions.py", "masking_ops.py", "ft_handlers.py",
    "task_merger.py", "utils.py", "baseline_configs.py", "merging_types.py",
)
# Documentation of the boundary; access is enforced by PUBLIC_FILES, not this list.
PRIVATE_PATHS = (
    "private_eval/", "eval_scripts/", "dataset/", "configs/",
    "modal_app.py", "modal_diagnostics.py", "research_agent/task.py",
    "research_agent/bridge.py", "research_agent/modal_runner.py",
    "research_agent/worker.py", "research_agent/inputs_cache.py", "tests/",
    "README.md", "requirements-modal.txt",
    "conda_environment.yaml", "research_agent/repository.py",
    "research_agent/prompt.md", "research_agent/examples/",
)


def public_repository():
    files = {name: (ROOT / name).read_text() for name in PUBLIC_FILES}
    files["baseline_results.md"] = (ROOT / "research_agent/baseline_results.md").read_text()
    # Refuse accidental inclusion of the withheld method or checkpoint names.
    if any("knots" in source.lower() for source in files.values()):
        raise ValueError("Private method name found in public source view")
    return files


def reference_files():
    root = ROOT / "research_agent/references"
    papers = {}
    for path in root.rglob("*.md"):
        if path.name.lower() == "readme.md" or path.is_symlink():
            continue
        if not path.resolve().is_relative_to(root.resolve()):
            continue
        papers[path.relative_to(root).as_posix()] = path.read_text()
    return papers


def reference_figures():
    """Allow only local raster images beneath the supplied reference folder."""
    root = ROOT / "research_agent/references"
    return {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        and path.is_file() and not path.is_symlink()
        and path.resolve().is_relative_to(root.resolve())
    }
