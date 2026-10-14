"""Inspect research tools with an explicit public repository boundary."""
import ast
import asyncio
import base64
import hashlib
import json
import math
import uuid
from pathlib import Path
from time import monotonic

import anyio
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.event import ErrorEvent, SampleLimitEvent
from inspect_ai.log import transcript
from inspect_ai.solver import Generate, TaskState, generate, solver, use_tools
from inspect_ai.tool import tool, ToolResult
from inspect_ai.model import ContentImage, ContentText
from inspect_ai.model import GenerateConfig
from inspect_ai.util import LimitExceededError, sample_limits
from research_agent.repository import public_repository, reference_files, reference_figures
from research_agent.runtime import PausableTimer
from research_agent.human_review import request_review
from private_eval.scoring import load_reference

ROOT = Path(__file__).resolve().parent


# Source, file-reading and termination helpers

def termination_reason(events):
    """Use Inspect's recorded stop event rather than guessing from usage."""
    reasons = {"token": "token budget exceeded", "time": "time limit exceeded",
               "message": "message limit exceeded", "working": "working time limit exceeded",
               "turn": "turn limit exceeded", "cost": "cost limit exceeded",
               "operator": "stopped by operator", "custom": "custom limit reached"}
    for event in reversed(events):
        if isinstance(event, SampleLimitEvent):
            return reasons[event.type]
        if isinstance(event, ErrorEvent):
            return "execution error (see Inspect log)"
    return "completed normally"

def check_source(source, require_implementation=False):
    if len(source.encode()) > 100_000:
        raise ValueError("Candidate exceeds 100 KB")
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "merge"]
    if len(functions) != 1:
        raise ValueError("Define exactly one merge(base_model, task_vectors, lora_adapters) function")
    function = functions[0]
    args = function.args
    if ([a.arg for a in args.args] != ["base_model", "task_vectors", "lora_adapters"]
            or args.vararg or args.kwarg or args.kwonlyargs or args.posonlyargs or args.defaults):
        raise ValueError("Required signature: merge(base_model, task_vectors, lora_adapters)")
    body = function.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    if require_implementation and (not body or all(isinstance(n, ast.Pass) for n in body)):
        raise ValueError("Implement the empty merge function before evaluating")
    return hashlib.sha256(source.encode()).hexdigest()


def read_slice(files, name, start, lines, max_chars=20000):
    if not name:
        return json.dumps(sorted(files))
    if name not in files or start < 1 or lines < 1 or max_chars < 1:
        return "Invalid file or line range"
    content = files[name].splitlines()
    chunk = "\n".join(f"{i + 1}: {content[i]}" for i in range(start - 1, min(len(content), start - 1 + lines)))
    notice = "\n[Character limit reached; increase max_chars to read the full requested range.]" if len(chunk) > max_chars else ""
    return f"{name}: {len(content)} total lines\n" + chunk[:max_chars] + notice


@task
def research(max_evaluations: int = 3, run_dir: str = "agent-runs", token_budget: int = 30000,
             time_limit: int = 3600, max_human_reviews: int = 0):
    # Run helpers and cleanup

    def budget_status():
        used = active_state.token_usage if active_state is not None else 0
        limit = active_state.token_limit if active_state is not None else token_budget
        try:
            clock = sample_limits().time
            seconds_limit, seconds_used, seconds_remaining = clock.limit, clock.usage, clock.remaining
        except RuntimeError:
            # No running Inspect sample during direct local tool tests/setup.
            seconds_limit = time_limit
            seconds_used = max(0, monotonic() - started_at) if started_at is not None else 0
            seconds_remaining = max(0, seconds_limit - seconds_used)
        if max_human_reviews:
            seconds_limit, seconds_used, seconds_remaining = timer.limit, timer.elapsed, timer.remaining
        return {"token_limit": limit, "tokens_used": used,
                "tokens_remaining": max(0, limit - used) if limit is not None else None,
                "evaluation_limit": max_evaluations, "evaluations_used": attempts,
                "evaluations_remaining": max(0, max_evaluations - attempts),
                "time_limit_seconds": seconds_limit, "time_used_seconds": seconds_used,
                "time_remaining_seconds": seconds_remaining,
                "human_review_limit": max_human_reviews, "human_reviews_used": review_attempts,
                "human_reviews_remaining": max(0, max_human_reviews - review_attempts),
                "human_wait_seconds": timer.human_wait_seconds,
                "waiting_for_human": timer.paused_at is not None,
                "submitted": submitted}

    def freeze(code, report, explanation, selection):
        nonlocal submitted
        submitted = True
        (folder / "submission.py").write_text(code)
        (folder / "algorithm.md").write_text(explanation.strip() + "\n")
        (folder / "submission.json").write_text(json.dumps(
            {**report, "selection": selection, "candidate_sha256": check_source(code),
             "algorithm_file": "algorithm.md",
             "algorithm_sha256": hashlib.sha256((explanation.strip() + "\n").encode()).hexdigest(),
             "human_reviews": [r for r in human_reviews if r["candidate_sha256"] == check_source(code)],
             "human_review_requests": review_attempts,
             "human_assisted": any(r["status"] == "accepted" for r in human_reviews)}, indent=2))

    async def finalize(state: TaskState):
        # Inspect task cleanup also runs on token/message/time-limit termination.
        try:
            async with lock:
                if submitted:
                    return
                if successful:
                    tests = [item for item in successful.values() if item["report"]["split"] == "test"]
                    best = max(tests or list(successful.values()), key=lambda item: item["report"]["scores"]["avg"])
                    selection = "fallback_best_test" if tests else "fallback_best_validation"
                    freeze(best["source"], best["report"], best["explanation"], selection)
                else:
                    (folder / "submission.json").write_text(json.dumps(
                        {"selection": "none", "reason": "No successfully evaluated candidate"}, indent=2))
        finally:
            try:
                events = transcript().events
            except RuntimeError:
                events = []  # Direct local tool tests have no Inspect transcript.
            reason = "time limit exceeded" if timer.expired else termination_reason(events)
            print(f"[Research run ended] {reason}", flush=True)

    # Agent tools

    @tool
    def get_budget():
        async def execute() -> str:
            """Report token usage, remaining evaluation attempts and remaining wall-clock seconds.

            Usage covers completed calls; future context and output also consume
            tokens. Checking does not consume an evaluation.
            """
            return json.dumps(budget_status())
        return execute

    @tool
    def read_repository():
        async def execute(path: str = "", start: int = 1, lines: int = 200, max_chars: int = 20000) -> str:
            """List or read approved repository files, without write access.

            Args:
                path: Exact listed filename; empty lists the public repository.
                start: First line, starting at 1.
                lines: Positive number of lines to request; choose based on your token budget.
                max_chars: Positive character limit for the excerpt; characters are not tokens.
            """
            return read_slice({**repository, "candidate.py": source}, path, start, lines, max_chars)
        return execute

    @tool
    def read_reference():
        async def execute(name: str = "", start: int = 1, lines: int = 200, max_chars: int = 20000) -> str:
            """List or read supplied research papers, including paper subfolders.

            Args:
                name: Exact path from the reference list; empty lists references.
                start: First line, starting at 1.
                lines: Positive number of lines to request; choose based on your token budget.
                max_chars: Positive character limit for the excerpt; characters are not tokens.
            """
            return read_slice(papers, name, start, lines, max_chars)
        return execute

    @tool
    def read_figure():
        async def execute(name: str = "") -> ToolResult:
            """List reference figures or return one image for visual analysis.

            Args:
                name: Exact reference-relative image path; empty lists available images.
                    Resolve Markdown image links relative to the paper's folder.
            """
            if not name:
                return json.dumps(sorted(figures))
            path = figures.get(name)
            if (path is None or path.is_symlink() or not path.is_file()
                    or not path.resolve().is_relative_to((ROOT / "references").resolve())):
                return "Invalid reference figure"
            mime = {".png": "image/png", ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg", ".webp": "image/webp"}[path.suffix.lower()]
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            return [ContentText(text=f"Reference figure: {name}"),
                    ContentImage(image=f"data:{mime};base64,{encoded}")]
        return execute

    @tool
    def write_candidate():
        async def execute(code: str) -> str:
            """Replace only candidate.py; check syntax and the fixed signature.

            Args:
                code: Complete Python source defining merge(base_model, task_vectors, lora_adapters).
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
                return f"Saved candidate {digest}; syntax and signature valid. Model checks happen remotely."
        return execute

    @tool
    def evaluate_candidate():
        async def execute(explanation: str, split: str = "val") -> str:
            """Evaluate candidate.py on val or test, returning four scores labeled by metric.

            Args:
                explanation: Markdown describing this candidate's algorithm, equations
                    or pseudocode, hyperparameters, rationale and limitations.
                split: val for hyperparameter selection, or test for benchmark feedback.
            """
            nonlocal attempts
            async with lock:
                if submitted or attempts >= max_evaluations:
                    return "No experiments available; submit an evaluated candidate"
                remaining = budget_status()["time_remaining_seconds"]
                if remaining is not None and remaining <= 0:
                    return "The agent's time limit has been reached; no new evaluation can start"
                if split not in {"val", "test"}:
                    return "split must be val or test"
                try:
                    digest = check_source(source, require_implementation=True)
                except (ValueError, SyntaxError) as error:
                    return str(error)
                if not explanation.strip() or len(explanation.encode()) > 20000:
                    return "Provide a nonempty Markdown algorithm explanation of at most 20 KB"
                try:
                    load_reference(split)
                except ValueError as error:
                    return str(error)
                attempts += 1
                snapshot = source
                (folder / f"candidate_{attempts:03}.py").write_text(snapshot)
                algorithm_file = f"algorithm_{attempts:03}.md"
                (folder / algorithm_file).write_text(explanation.strip() + "\n")
                try:
                    def remote():
                        # Initialize the Modal app only when an evaluation is requested.
                        from research_agent.modal_runner import app, prepare_evaluation, evaluate_candidate as evaluator
                        with app.run():
                            prepared = prepare_evaluation()
                            return evaluator.remote(snapshot, split,
                                                    prepared["cache_key"], prepared["sandbox_image_id"])
                    report = await asyncio.to_thread(remote)
                    if set(report["scores"]) != {"snli", "mnli", "scitail", "avg"} or not all(math.isfinite(v) for v in report["scores"].values()):
                        raise ValueError("Evaluator returned a nonfinite score")
                    report["split"] = split
                    report["score_type"] = "reference_relative_percent"
                    key = digest + ":" + split
                    successful[key] = {"source": snapshot, "report": report, "explanation": explanation}
                except Exception as error:
                    report = {"error": str(error), "attempt": attempts}
                report["budget"] = budget_status()
                report["split"] = split
                report["algorithm_file"] = algorithm_file
                (folder / f"result_{attempts:03}.json").write_text(json.dumps(report, indent=2))
                feedback = {k: report[k] for k in
                            ("split", "score_type", "scores", "error", "experiment_id", "budget", "algorithm_file")
                            if k in report}
                return json.dumps(feedback)
        return execute

    @tool
    def submit_candidate():
        async def execute(explanation: str = "") -> str:
            """Freeze the evaluated candidate and save algorithm.md alongside its code.

            Args:
                explanation: Optional revised Markdown explanation including observed
                    validation evidence. Empty reuses the explanation saved at evaluation.
            """
            async with lock:
                if submitted:
                    return "Submission is already frozen"
                digest = check_source(source)
                selected = successful.get(digest + ":test") or successful.get(digest + ":val")
                if selected is None:
                    return "Evaluate this exact candidate successfully before submitting"
                if explanation and (not explanation.strip() or len(explanation.encode()) > 20000):
                    return "Provide a nonempty Markdown explanation of at most 20 KB, or omit it"
                explanation = explanation or selected["explanation"]
                freeze(source, selected["report"], explanation, "agent_submission")
                return "Submission frozen. End with a brief explanation of your method."
        return execute

    @tool
    def human_scorer():
        async def execute(explanation: str) -> str:
            """Ask a human to score the current candidate's innovativeness and reasoning, each 0-10.

            Args:
                explanation: Markdown describing the method, claimed contribution,
                    rationale, assumptions, limitations and available evidence.
            """
            nonlocal review_attempts
            async with lock:
                if submitted or review_attempts >= max_human_reviews:
                    return "No human reviews available"
                if timer.remaining <= 0:
                    return "The agent's time limit has been reached"
                try:
                    digest = check_source(source, require_implementation=True)
                except (ValueError, SyntaxError) as error:
                    return str(error)
                if not explanation.strip() or len(explanation.encode()) > 20000:
                    return "Provide a nonempty Markdown explanation of at most 20 KB"
                review_attempts += 1
                evidence = [item["report"] for key, item in successful.items() if key.startswith(digest + ":")]
                record = await request_review(folder, review_attempts, max_human_reviews,
                                              source, digest, explanation, evidence, timer)
                human_reviews.append(record)
                feedback = {k: record[k] for k in ("review", "candidate_sha256", "status", "scores", "comments", "error") if k in record}
                return json.dumps(feedback | {"budget": budget_status()})
        return execute

    # Custom solvers

    @solver
    def run_agent_with_clock():
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            if sample_limits().time.limit is not None:
                raise ValueError("With human reviews, use -T time_limit=... rather than --time-limit; an external wall-clock limit cannot pause")
            with anyio.CancelScope() as scope:
                timer.attach(scope)
                try:
                    result = await agent_loop(state, generate)
                finally:
                    timer.finish()
            if scope.cancelled_caught:
                timer.expired = True
                transcript().info({"research_stop_reason": "time limit exceeded"})
                raise LimitExceededError("time", value=timer.elapsed, limit=time_limit,
                                         message="Agent runtime budget exceeded (human waiting excluded)")
            return result
        return solve

    @solver
    def track_budget():
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            nonlocal active_state, started_at
            active_state = state
            started_at = monotonic()
            if max_human_reviews:
                timer.start()
            return state
        return solve

    # Run setup and shared state
    if max_evaluations < 1 or token_budget < 1 or time_limit < 1:
        raise ValueError("Budgets must be positive")
    if max_human_reviews < 0:
        raise ValueError("max_human_reviews must be nonnegative")
    
    # Inspect changes the working directory during task loading and execution.
    run_root = Path(run_dir).expanduser()
    if not run_root.is_absolute():
        run_root = ROOT.parent / run_root
    folder = run_root.resolve() / uuid.uuid4().hex
    folder.mkdir(parents=True, exist_ok=False)
    source = (ROOT / "candidate.py").read_text()

    repository = public_repository()
    papers = reference_files()
    figures = reference_figures()
    
    for name, content in repository.items():
        path = folder / "repository" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    (folder / "candidate.py").write_text(source)

    attempts = 0
    review_attempts = 0
    human_reviews = []
    timer = PausableTimer(time_limit)
    submitted = False
    lock = asyncio.Lock()
    successful = {}
    active_state = None
    started_at = None

    # Prompt construction

    prompt = ((ROOT / "prompt.md").read_text()
              .replace("{{max_evaluations}}", str(max_evaluations))
              .replace("{{token_budget}}", str(token_budget))
              .replace("{{time_limit}}", str(time_limit))
              .replace("{{max_human_reviews}}", str(max_human_reviews))
              .replace("{{initial_candidate}}", source))
    (folder / "prompt.txt").write_text(prompt)
    # Solver sequence

    agent_loop = generate(tool_calls="loop")
    solvers = [
        track_budget(),
        use_tools([
            get_budget(),
            read_repository(),
            read_reference(),
            write_candidate(),
            evaluate_candidate(),
            submit_candidate(),
            read_figure(),
        ] + ([human_scorer()] if max_human_reviews else [])),
        run_agent_with_clock() if max_human_reviews else agent_loop,
    ]

    # Inspect task assembly

    return Task(
        dataset=[Sample(input=prompt)],
        solver=solvers,
        cleanup=finalize,
        config=GenerateConfig(parallel_tool_calls=False),
        message_limit=60, token_limit=token_budget,
        time_limit=None if max_human_reviews else time_limit
    )
