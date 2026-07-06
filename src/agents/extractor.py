"""
src/agents/extractor.py

Agent 1: reads retrieved SOP chunks and extracts discrete
WorkflowRule objects.

"""
from typing import List

from pydantic import ValidationError

from src.llm.base import LLMClient
from src.vectorstore.base import SearchResult
from src.agents.schemas import WorkflowRule, ExtractedRules


EXTRACTOR_SYSTEM = """You are a regulatory workflow analyst.

Your job is to read excerpts from a Standard Operating Procedure and
extract discrete, actionable rules. Each rule should be structured
so a workflow automation system could execute it.

RULES FOR EXTRACTION:
1. Only extract rules explicitly stated in the excerpts. Do not infer
   or fabricate.
2. Each rule must have a clear trigger (what event activates it) and
   action (what must be done).
3. Include the deadline if the excerpt specifies one (e.g. "within 24
   hours", "within 10 working days").
4. Include the responsible party if the excerpt names one (e.g.
   "Principal Investigator", "study coordinator").
5. Attribute each rule to the page number of the excerpt it came from.
6. If an excerpt does not contain any extractable rule, ignore it —
   do not invent one.

Respond with a JSON object of the form:
{
  "rules": [
    {
      "rule_id": "R-001",
      "trigger": "...",
      "action": "...",
      "deadline": "..." or null,
      "responsible_party": "..." or null,
      "source_page": <integer>
    },
    ...
  ]
}"""


def _format_chunks(chunks: List[SearchResult]) -> str:
    """Label each chunk with its page so the model can attribute rules."""
    parts = []
    for c in chunks:
        parts.append(f"[Excerpt from page {c.page}]\n{c.text}")
    return "\n\n---\n\n".join(parts)


def extract_rules(chunks: List[SearchResult], llm: LLMClient) -> List[WorkflowRule]:
    """
    Run the extractor over a batch of retrieved chunks.

    Returns a list of validated WorkflowRule objects. If the LLM
    produces malformed JSON structurally (missing fields, wrong
    types), Pydantic raises — we surface it rather than silently
    dropping rules.
    """
    if not chunks:
        return []

    prompt = (
        "Extract workflow rules from the following SOP excerpts. "
        "Respond in JSON with the schema described in the system message.\n\n"
        + _format_chunks(chunks)
    )

    raw = llm.generate_json(
        prompt=prompt,
        system=EXTRACTOR_SYSTEM,
        temperature=0.0,
        max_tokens=2048,
    )

    try:
        parsed = ExtractedRules.model_validate(raw)
    except ValidationError as e:
        # Surface the LLM's actual output alongside the validation
        # error — invaluable when debugging prompt drift.
        raise ValueError(
            f"Extractor produced invalid schema. "
            f"Raw output: {raw}\nValidation error: {e}"
        ) from e

    return parsed.rules
