"""Exercise the real human-review form using scripted, free model responses."""
from pathlib import Path

from inspect_ai import eval
from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ModelOutput
from inspect_ai.tool import ToolCall

from research_agent.task import research


def tool_response(name: str, arguments: dict, identifier: str) -> ModelOutput:
    return ModelOutput(model="mockllm/model", choices=[ChatCompletionChoice(
        message=ChatMessageAssistant(content="", tool_calls=[
            ToolCall(id=identifier, function=name, arguments=arguments)]),
        stop_reason="tool_calls")])


def main():
    code = (Path(__file__).resolve().parent / "examples" / "average.py").read_text()
    explanation = (
        "# Human-review interface check\n\n"
        "This candidate adds the mean of the three dense task vectors to the base model. "
        "It is the existing averaging baseline, with no claim of innovation. "
        "No evaluation is run in this check. Please enter two ratings and optional "
        "comments to verify that human feedback reaches the agent."
    )
    responses = [
        tool_response("write_candidate", {"code": code}, "write"),
        tool_response("human_scorer", {"explanation": explanation}, "review"),
        tool_response("get_budget", {}, "budget"),
        ModelOutput.from_content(model="mockllm/model", content="Human-review check complete."),
    ]
    task = research(run_dir="agent-runs/human-review-check", token_budget=100000,
                    time_limit=60, max_human_reviews=1)
    print("Local check: no API calls or GPU jobs. Enter ratings when the form opens.")
    eval(task, model="mockllm/model", model_args={"custom_outputs": responses},
         log_dir="agent-logs/human-review-check", acp_server=False)
    print("Review snapshots and ratings: agent-runs/human-review-check/<run-id>/human_review_001.*")
    print("This check does not evaluate or submit a research candidate.")


if __name__ == "__main__":
    main()
