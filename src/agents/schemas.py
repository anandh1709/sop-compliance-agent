"""
src/agents/schemas.py

Pydantic models used by both agents. Centralised so the extractor's
output type is exactly what the checker consumes — no drift.
"""
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class WorkflowRule(BaseModel):
    """
    A discrete, actionable rule extracted from an SOP.

    The field names deliberately echo the language of workflow
    automation systems (trigger, action, deadline, responsible party)
    because that's what a downstream Power Automate flow or Dataverse
    table row would consume.
    """
    rule_id: str = Field(
        description="Short unique identifier, e.g. R-001, R-002"
    )
    trigger: str = Field(
        description="The event or condition that activates this rule, "
                    "e.g. 'a serious adverse event occurs'"
    )
    action: str = Field(
        description="What must be done, e.g. 'notify the IRB'"
    )
    deadline: Optional[str] = Field(
        default=None,
        description="Time constraint if specified, e.g. 'within 24 hours'"
    )
    responsible_party: Optional[str] = Field(
        default=None,
        description="Who is responsible, e.g. 'Principal Investigator'"
    )
    source_page: Optional[int] = Field(
        default=None,
        description="Page number in the source SOP for auditor traceability"
    )


class ExtractedRules(BaseModel):
    """LLM response wrapper — JSON mode expects a top-level object."""
    rules: List[WorkflowRule]


class CheckResult(BaseModel):
    """One line item from the compliance checker."""
    requirement: str = Field(
        description="The compliance checklist item being evaluated"
    )
    status: Literal["satisfied", "missing", "partial"] = Field(
        description="Whether the SOP covers this requirement"
    )
    evidence_rule_ids: List[str] = Field(
        default_factory=list,
        description="rule_ids from the extracted rules that support "
                    "the status. Empty for 'missing'."
    )
    reasoning: str = Field(
        description="One-sentence justification for the status"
    )


class ComplianceReport(BaseModel):
    """Final output of the graph. This is what the API returns."""
    extracted_rules: List[WorkflowRule]
    checks: List[CheckResult]