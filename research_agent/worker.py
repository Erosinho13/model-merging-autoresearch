"""Runs only inside an isolated candidate sandbox; never in the grader."""
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main(folder):
    folder = Path(folder)
    tensors = load_file(str(folder / "inputs.safetensors"))
    base = {k[5:]: v for k, v in tensors.items() if k.startswith("base/")}
    vectors = [{k[len(f"task{i}/"):]: v for k, v in tensors.items()
                if k.startswith(f"task{i}/")} for i in range(3)]
    scope = {"torch": torch}
    exec(compile((folder / "candidate.py").read_text(), "candidate.py", "exec"), scope)
    with torch.no_grad():
        merged = scope["merge"](base, vectors)
    save_file({k: v.detach().cpu().contiguous().clone() for k, v in merged.items()},
              str(folder / "output.safetensors"))


if __name__ == "__main__":
    main(sys.argv[1])
