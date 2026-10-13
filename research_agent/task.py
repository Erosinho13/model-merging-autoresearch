"""Inspect tools orchestrate research; the existing evaluator supplies scores."""
import ast
import asyncio
import hashlib
import json
import uuid
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate, use_tools
from inspect_ai.tool import tool
from inspect_ai.model import GenerateConfig

ROOT = Path(__file__).resolve().parent


def check_source(source):
    if len(source.encode()) > 100_000:
        raise ValueError("Candidate exceeds 100 KB")
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "merge"]
    if len(functions) != 1:
        raise ValueError("Define exactly one merge(base_weights, task_vectors) function")
    args = functions[0].args
    if ([a.arg for a in args.args] != ["base_weights", "task_vectors"]
            or args.vararg or args.kwarg or args.kwonlyargs or args.posonlyargs or args.defaults):
        raise ValueError("Required signature: merge(base_weights, task_vectors)")
    return hashlib.sha256(source.encode()).hexdigest()


@task
def research(max_evaluations: int = 3, run_dir: str = "agent-runs",
             token_budget: int = 30000):
    if max_evaluations < 1 or token_budget < 1:
        raise ValueError("Budgets must be positive")
    folder = Path(run_dir) / uuid.uuid4().hex
    folder.mkdir(parents=True, exist_ok=False)
    source = (ROOT / "average.py").read_text()
    attempts = 0
    submitted = False
    lock = asyncio.Lock()
    successful = {}

    @tool
    def read_reference():
        async def execute(name: str = "", start: int = 1, lines: int = 200) -> str:
            """List or read supplied Markdown research references.

            Args:
                name: Filename from the reference list; empty lists references.
                start: First line to read, starting at 1.
                lines: Number of lines, at most 300.
            """
            papers = {p.name: p for p in (ROOT / "references").glob("*.md") if not p.is_symlink()}
            if not name:
                return json.dumps(sorted(papers))
            if name not in papers or start < 1 or not 1 <= lines <= 300:
                return "Invalid reference or line range"
            return "\n".join(papers[name].read_text().splitlines()[start - 1:start - 1 + lines])
        return execute

    @tool
    def write_candidate():
        async def execute(code: str) -> str:
            """Replace the candidate file and check its syntax and signature.

            Args:
                code: Complete Python source defining merge(base_weights, task_vectors).
            """
            nonlocal source
            async with lock:
                if submitted:
                    return "Submission is frozen"
                try:
                    digest = check_source(code)
                except (ValueError, SyntaxError) as error:
                    return str(error)
                source = code
                (folder / "candidate.py").write_text(code)
                return f"Saved candidate {digest}; syntax and signature valid. Tensor checks happen remotely."
        return execute

    @tool
    def evaluate_candidate():
        async def execute() -> str:
            """Evaluate the current candidate on validation only; consumes one experiment."""
            nonlocal attempts
            async with lock:
                if submitted or attempts >= max_evaluations:
                    return "No experiments available; submit an evaluated candidate"
                check_source(source)
                attempts += 1
                snapshot = source
                (folder / f"candidate_{attempts:03}.py").write_text(snapshot)
                try:
                    def remote():
                        from research_agent.modal_runner import app, evaluate_candidate as evaluator
                        with app.run():
                            return evaluator.remote(snapshot, "val")
                    report = await asyncio.to_thread(remote)
                    successful[check_source(snapshot)] = report
                except Exception as error:
                    report = {"error": str(error), "attempt": attempts}
                (folder / f"result_{attempts:03}.json").write_text(json.dumps(report, indent=2))
                return json.dumps(report)
        return execute

    @tool
    def submit_candidate():
        async def execute() -> str:
            """Freeze the current successfully evaluated candidate as the final submission."""
            nonlocal submitted
            async with lock:
                digest = check_source(source)
                if digest not in successful:
                    return "Evaluate this exact candidate successfully before submitting"
                submitted = True
                (folder / "submission.py").write_text(source)
                (folder / "submission.json").write_text(json.dumps(successful[digest], indent=2))
                return "Submission frozen. End your response with a brief explanation of your method."
        return execute

    prompt = (ROOT / "prompt.md").read_text() + f"\nGPU evaluation limit: {max_evaluations}.\nInitial candidate:\n```python\n{source}\n```"
    (folder / "prompt.txt").write_text(prompt)
    return Task(dataset=[Sample(input=prompt)],
                solver=[use_tools([read_reference(), write_candidate(), evaluate_candidate(), submit_candidate()]),
                        generate(tool_calls="loop")],
                config=GenerateConfig(parallel_tool_calls=False),
                message_limit=60, token_limit=token_budget)
