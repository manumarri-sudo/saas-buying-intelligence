"""
Strict decision-narrative gate for SaaS buying intelligence.

A row is ONLY eligible if decision_context contains:
  A) At least one DECISION VERB (whole word, case-insensitive)
  B) At least one REASONING MARKER (case-insensitive)

If A and B are not both present, the row is DROPPED. No exceptions.
This gate is applied AFTER text cleaning and BEFORE writing rows.
It cannot be bypassed by confidence scoring or any other signal.
"""

import re
from enum import Enum

# ── Drop reasons (for debug artifact) ─────────────────────────────────

class DropReason(str, Enum):
    PASSED = "passed"
    MISSING_DECISION_VERB = "dropped_missing_decision_verb"
    MISSING_REASON_MARKER = "dropped_missing_reason_marker"
    NAVIGATION_NOISE = "dropped_navigation_noise"
    BIO_OR_JOB_DESCRIPTION = "dropped_bio_or_job_description"


# ── Decision verbs (strict list, whole-word boundaries) ───────────────

_DECISION_VERBS = [
    "chose", "choose",
    "selected", "select",
    "evaluated", "evaluate",
    "shortlisted", "shortlist",
    "compared", "compare",
    "trialed", "trial",
    "piloted", "pilot",
    "adopted", "adopt",
    "purchased", "purchase",
    "procured", "procurement",
    "rejected", "reject",
    "replaced", "replace",
    "switched", "switch",
    "migrated", "migrate",
]

DECISION_VERB_RE = re.compile(
    r'\b(' + '|'.join(re.escape(v) for v in _DECISION_VERBS) + r')\b',
    re.IGNORECASE,
)


# ── Reasoning markers (strict list, case-insensitive) ─────────────────

_REASONING_MARKERS = [
    # Causal connectors
    "because",
    "due to",
    "since",
    "so that",
    "therefore",
    "as a result",
    # Need/requirement language
    "needed",
    "required",
    "requirement",
    "must have",
    "deciding factor",
    "dealbreaker",
    # Concern/issue/challenge language
    "concern",
    "concerns",
    "issue",
    "issues",
    "challenge",
    "challenges",
    "risk",
    "risks",
    # Specific evaluation reasoning
    "too expensive",
    "pricing",
    "security review",
    "compliance",
    "integration",
    "vendor lock-in",
    "onboarding",
    "implementation",
    "support",
]

REASONING_MARKER_RE = re.compile(
    r'\b(' + '|'.join(re.escape(m) for m in _REASONING_MARKERS) + r')\b',
    re.IGNORECASE,
)


def has_decision_verb(text: str) -> bool:
    """Check if text contains at least one decision verb."""
    return bool(DECISION_VERB_RE.search(text))


def has_reasoning_marker(text: str) -> bool:
    """Check if text contains at least one reasoning marker."""
    return bool(REASONING_MARKER_RE.search(text))


def passes_decision_narrative_gate(text: str) -> bool:
    """
    Hard gate: returns True ONLY if text contains both a decision verb
    AND a reasoning marker. No exceptions, no bypass via scoring.
    """
    return has_decision_verb(text) and has_reasoning_marker(text)


def classify_drop_reason(text: str) -> DropReason:
    """
    Classify why a row would be dropped by the gate.
    Returns DropReason.PASSED if it passes.
    """
    has_verb = has_decision_verb(text)
    has_reason = has_reasoning_marker(text)

    if has_verb and has_reason:
        return DropReason.PASSED
    elif not has_verb:
        return DropReason.MISSING_DECISION_VERB
    else:
        return DropReason.MISSING_REASON_MARKER
