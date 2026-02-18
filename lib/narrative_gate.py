"""
Strict decision-narrative gate for SaaS buying intelligence.

A row is ONLY eligible if decision_context contains:
  A) At least one DECISION VERB (whole word, case-insensitive)
  B) At least one REASONING MARKER (case-insensitive)
  C) Evidence of a real actor making the decision (we/team/company/client/etc.)

If A, B, and C are not all present, the row is DROPPED. No exceptions.
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
    MISSING_ACTOR = "dropped_missing_actor"
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
    # Friction indicators
    "limitation",
    "limitations",
    "friction",
    "tradeoff",
    "trade-off",
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
    # Lifecycle / switching
    "moved from",
    "switched from",
    "migrated away",
    "transitioned",
    "deprecated",
    "lacked",
    "scalability",
]

REASONING_MARKER_RE = re.compile(
    r'\b(' + '|'.join(re.escape(m) for m in _REASONING_MARKERS) + r')\b',
    re.IGNORECASE,
)


# ── Actor presence (real org/team/person making the decision) ──────────
# Filters out third-person product descriptions ("CRM allows you to...")
# Requires evidence that a real subject made the decision.

ACTOR_RE = re.compile(
    r'\b('
    # ── First person plural + decision verb (tightest signal) ───────
    r'we\s+(?:chose|selected|evaluated|adopted|rejected|replaced|switched|'
    r'migrated|purchased|procured|trialed|piloted|decided|moved\s+(?:from|to|away)|'
    r'opted\s+(?:for|to)|ended\s+up\s+(?:with|choosing|selecting|going)|'
    r'went\s+with|settled\s+on|shortlisted|narrowed\s+down|picked|'
    r'signed\s+(?:up\s+with|a\s+contract|on\s+with))|'
    # "our team/company/org + decided/chose/selected/evaluated..."
    r'our\s+(?:team|company|org|organization|engineering|engineers|'
    r'developers|devs|security|infra|infrastructure|ops|management|'
    r'procurement|finance|it\s+(?:department|team)|cto|vp|ciso|'
    r'procurement\s+team|buying\s+committee|leadership)\s+'
    r'(?:\w+\s+){0,3}'
    r'(?:chose|selected|evaluated|adopted|rejected|replaced|switched|'
    r'migrated|purchased|procured|trialed|piloted|decided|opted)|'
    # ── Third person org subject + decision verb ─────────────────────
    r'(?:the\s+)?(?:company|organization|team|firm|enterprise|client|'
    r'customer|vendor|agency|startup|corporation)\s+'
    r'(?:chose|selected|evaluated|adopted|rejected|replaced|switched|'
    r'migrated|purchased|procured|trialed|piloted|decided|moved|opted)|'
    # ── Named actor + decision verb ──────────────────────────────────
    r'(?:they|the\s+team|management|engineering|leadership|the\s+client|'
    r'the\s+vendor|the\s+customer)\s+'
    r'(?:chose|selected|evaluated|adopted|rejected|replaced|switched|'
    r'migrated|purchased|procured|decided|opted|needed|required)|'
    # ── Passive voice with agent marker ─────────────────────────────
    r'(?:was|were)\s+(?:chosen|selected|evaluated|adopted|rejected|'
    r'replaced|switched|migrated|purchased|procured|shortlisted)\s+'
    r'(?:by|after|because|due\s+to)|'
    # ── First-person switch/migration phrasing ────────────────────────
    r'(?:we|our\s+team|our\s+company)\s+'
    r'(?:moved|switched|migrated|transitioned)\s+(?:from|away\s+from|to)|'
    # ── "why we ..." or "how we ..." decision phrases ─────────────────
    r'why\s+we\s+(?:chose|switched|moved|use|replaced|selected|went\s+with)|'
    r'how\s+we\s+(?:chose|evaluated|selected|decided|migrated|switched)|'
    # ── After-evaluation actor ────────────────────────────────────────
    r'after\s+(?:evaluating|comparing|testing|trialing|piloting|assessing|'
    r'reviewing|shortlisting|benchmarking)\s+(?:\w+\s+){0,4}'
    r'(?:we|the\s+team|our\s+team|management)|'
    # ── Decision made by / decision to ───────────────────────────────
    r'decision\s+(?:was\s+made|to\s+(?:choose|select|adopt|reject|replace|'
    r'switch|migrate|purchase|procure))'
    r')\b',
    re.IGNORECASE,
)


def has_decision_verb(text: str) -> bool:
    """Check if text contains at least one decision verb."""
    return bool(DECISION_VERB_RE.search(text))


def has_reasoning_marker(text: str) -> bool:
    """Check if text contains at least one reasoning marker."""
    return bool(REASONING_MARKER_RE.search(text))


def has_actor(text: str) -> bool:
    """Check if text contains evidence of a real actor making the decision."""
    return bool(ACTOR_RE.search(text))


def passes_decision_narrative_gate(text: str) -> bool:
    """
    Hard gate: returns True ONLY if text contains:
      - a decision verb
      - a reasoning marker
      - an actor (real org/team/person making the decision)
    No exceptions, no bypass via scoring.
    """
    return has_decision_verb(text) and has_reasoning_marker(text) and has_actor(text)


def classify_drop_reason(text: str) -> DropReason:
    """
    Classify why a row would be dropped by the gate.
    Returns DropReason.PASSED if it passes.
    """
    has_verb = has_decision_verb(text)
    has_reason = has_reasoning_marker(text)
    has_act = has_actor(text)

    if has_verb and has_reason and has_act:
        return DropReason.PASSED
    elif not has_verb:
        return DropReason.MISSING_DECISION_VERB
    elif not has_reason:
        return DropReason.MISSING_REASON_MARKER
    else:
        return DropReason.MISSING_ACTOR
