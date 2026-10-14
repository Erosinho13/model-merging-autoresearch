"""Local human judgements associated with immutable candidate snapshots."""
import json
import math

from acp.schema import ElicitationSchema, ElicitationNumberPropertySchema, ElicitationStringPropertySchema
from inspect_ai.util import request_input


async def request_review(folder, index, limit, code, digest, explanation, evidence, timer):
    stem = f"human_review_{index:03}"
    code_path, explanation_path = folder / f"{stem}.py", folder / f"{stem}.md"
    code_path.write_text(code)
    explanation_path.write_text(explanation.strip() + "\n")
    record = {"review": index, "candidate_sha256": digest, "code_file": code_path.name,
              "explanation_file": explanation_path.name, "status": "pending", "evaluations": evidence}
    review_path = folder / f"{stem}.json"
    review_path.write_text(json.dumps(record, indent=2))
    schema = ElicitationSchema(description=explanation, properties={
        "innovation": ElicitationNumberPropertySchema(type="number", minimum=0, maximum=10,
            title="Innovativeness (0-10)", description="Originality of the proposed mechanism relative to prior work."),
        "reasoning": ElicitationNumberPropertySchema(type="number", minimum=0, maximum=10,
            title="Logical reasoning (0-10)", description="Coherence and support for the explanation, including assumptions and limitations."),
        "comments": ElicitationStringPropertySchema(type="string", title="Feedback",
            description="Optional reasons for the scores and suggestions for the next attempt."),
    }, required=["innovation", "reasoning"])
    # Inspect renders this message as an unscrollable header. Keep the full
    # proposal in the scrollable form and its Markdown file instead.
    message = (f"Human review {index}/{limit}. Agent timer paused.\n"
               f"Review the code: {code_path}\nProposal: {explanation_path}\n"
               f"Evidence and metadata: {review_path}\n"
               "Scroll through the proposal below; use Tab to reach the ratings and feedback.")
    waited_before = timer.human_wait_seconds
    try:
        with timer.pause_for_human():
            while True:
                response = await request_input(message=message, schema=schema)
                if response.outcome != "accepted":
                    record["status"] = response.outcome
                    break
                values = response.content or {}
                if all(isinstance(values.get(k), (int, float))
                       and not isinstance(values[k], bool) and math.isfinite(values[k])
                       and 0 <= values[k] <= 10 for k in ("innovation", "reasoning")):
                    record.update(status="accepted", scores={k: values[k] for k in ("innovation", "reasoning")},
                                  comments=values.get("comments", ""))
                    break
                message = "Both scores must be numbers between 0 and 10.\n" + message.removeprefix(
                    "Both scores must be numbers between 0 and 10.\n")
    except Exception as error:
        record.update(status="error", error=str(error))
    finally:
        record["human_wait_seconds"] = timer.human_wait_seconds - waited_before
        if record["status"] == "pending":
            record["status"] = "cancelled"
        review_path.write_text(json.dumps(record, indent=2))
    return record
