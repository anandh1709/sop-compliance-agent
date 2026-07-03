"""
src/agents/checker.py

Agent 2: takes the WorkflowRule list from Agent 1 and evaluates it
against a compliance checklist, producing a CheckResult per
requirement.
"""
import time
from typing import List
from typing import List

from pydantic import ValidationError, BaseModel

from src.llm.base import LLMClient
from src.agents.schemas import WorkflowRule, CheckResult


CHECKER_SYSTEM = """You are a regulatory compliance auditor.

You will be given:
  1. A list of workflow rules extracted from an SOP.
  2. A single compliance requirement.

Your job is to determine whether the SOP's rules satisfy the
requirement, and cite the specific rule_ids that support your
verdict.

Verdict rules:
  - "satisfied": one or more extracted rules clearly cover the
    requirement.
  - "partial": the SOP addresses the requirement but is incomplete
    (e.g. requirement mentions a deadline but the SOP doesn't
    specify one).
  - "missing": no extracted rule addresses the requirement.

Respond with a JSON object of the form:
{
  "requirement": "<echo the requirement verbatim>",
  "status": "satisfied" | "partial" | "missing",
  "evidence_rule_ids": ["R-001", "R-003"],
  "reasoning": "<one sentence>"
}"""


class _CheckerResponse(BaseModel):
    """Internal wrapper matching the LLM's JSON response shape."""
    requirement: str
    status: str
    evidence_rule_ids: List[str] = []
    reasoning: str


def _format_rules(rules: List[WorkflowRule]) -> str:
    """Compact table-like format easy for the LLM to reason over."""
    if not rules:
        return "(No rules were extracted.)"
    lines = []
    for r in rules:
        lines.append(
            f"- {r.rule_id}: trigger='{r.trigger}' | action='{r.action}' | "
            f"deadline='{r.deadline}' | responsible='{r.responsible_party}' | "
            f"page={r.source_page}"
        )
    return "\n".join(lines)


def check_requirement(
    requirement: str,
    rules: List[WorkflowRule],
    llm: LLMClient,
) -> CheckResult:
    """Evaluate one checklist item against the extracted rules."""
    prompt = (
        f"EXTRACTED RULES:\n{_format_rules(rules)}\n\n"
        f"COMPLIANCE REQUIREMENT:\n{requirement}\n\n"
        f"Evaluate whether the extracted rules satisfy the requirement. "
        f"Respond in JSON as specified in the system message."
    )

    raw = llm.generate_json(
        prompt=prompt,
        system=CHECKER_SYSTEM,
        temperature=0.0,
        max_tokens=512,
    )

    try:
        parsed = _CheckerResponse.model_validate(raw)
    except ValidationError as e:
        raise ValueError(
            f"Checker produced invalid schema. "
            f"Raw output: {raw}\nValidation error: {e}"
        ) from e

    # Coerce the string status to the Literal type on CheckResult.
    # If the LLM returns an out-of-vocabulary value, default to
    # 'missing' — safer than crashing mid-report.
    status = parsed.status if parsed.status in {"satisfied", "partial", "missing"} else "missing"

    return CheckResult(
        requirement=parsed.requirement,
        status=status,  # type: ignore[arg-type]
        evidence_rule_ids=parsed.evidence_rule_ids,
        reasoning=parsed.reasoning,
    )


def check_all(
    requirements: List[str],
    rules: List[WorkflowRule],
    llm: LLMClient,
    pace_seconds: float = 1.0,
) -> List[CheckResult]:
    """
    Run the checker over every requirement in the checklist.

    Sequential with a small sleep between calls to stay comfortably
    inside Groq's free-tier tokens-per-minute limit. Prod would
    parallelise with asyncio.gather and rely on a paid rate tier.
    """
    results: List[CheckResult] = []
    for i, req in enumerate(requirements):
        if i > 0:
            time.sleep(pace_seconds)
        results.append(check_requirement(req, rules, llm))
    return results