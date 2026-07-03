"""
src/checklist.py

Hardcoded compliance checklist for the demo. Six requirements drawn
from ICH-GCP E6(R2) and 21 CFR Part 312 — the regulatory frameworks
the Johns Hopkins IRB SOP is written against.

Kept as a plain list of strings (not a database table) for the demo
because the checker only needs the wording. In production this would
be a Dataverse table with columns (id, market, framework, requirement,
severity, ...) and the checker would pull the relevant subset by
SOP type at runtime.
"""

DEFAULT_CHECKLIST = [
    "Adverse events must be reported to the IRB within a defined timeframe.",
    "Informed consent must be documented in writing before study procedures begin.",
    "Protocol deviations must be reported to the IRB.",
    "Study drug or device accountability records must be maintained.",
    "Investigator responsibilities and delegated tasks must be documented.",
    "Changes to the research protocol require IRB approval before implementation.",
]