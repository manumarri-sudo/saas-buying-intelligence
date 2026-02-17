"""
Structured row extraction from scored passages.

Transforms filtered text into rows with fields:
  decision_context, criteria, objection, workflow_step,
  industry_hint, confidence

Uses rule-based extraction first.
LLM fallback is available for low-confidence rows.
"""

import re
import logging
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ExtractedRow:
    decision_context: str
    criteria: str
    objection: str
    workflow_step: str
    industry_hint: str
    confidence: float
    source_url: str
    crawl_date: str
    matched_keywords: str  # comma-separated

    def to_dict(self) -> dict:
        return asdict(self)


# --- Workflow step patterns ---
WORKFLOW_PATTERNS = {
    "discovery": [
        r'\b(discover\w*|research\w*|look(?:ing)?\s+(?:for|into)|explor\w*|search(?:ing)?)',
    ],
    "evaluation": [
        r'\b(evaluat\w*|compar\w*|assess\w*|review\w*|analyz\w*|benchmark\w*|test(?:ing)?)',
    ],
    "shortlisting": [
        r'\b(shortlist\w*|narrow(?:ed)?\s+down|top\s+\d|finalist|candidate)',
    ],
    "trial": [
        r'\b(trial|pilot|proof\s+of\s+concept|POC|demo|free\s+tier|sandbox)',
    ],
    "negotiation": [
        r'\b(negotiat\w*|pricing|contract\w*|discount\w*|terms|proposal|quote)',
    ],
    "procurement": [
        r'\b(procurement|RFP|RFI|RFQ|purchas\w*|buy(?:ing)?|acquisition)',
    ],
    "implementation": [
        r'\b(implement\w*|deploy\w*|migrat\w*|onboard\w*|rollout|integrat\w*|launch)',
    ],
    "renewal": [
        r'\b(renew\w*|churn\w*|switch(?:ing)?|replac\w*|cancel\w*|retent\w*)',
    ],
}

# --- Criteria patterns ---
# NOTE: Use \b only at the start for prefix stems (e.g. pric matches pricing)
CRITERIA_PATTERNS = {
    "pricing": r'\b(pric\w*|cost\w*|budget|expense|ROI|TCO|total\s+cost)',
    "security": r'\b(secur\w*|complian\w*|SOC\s*2|GDPR|HIPAA|encrypt\w*|audit\w*)',
    "scalability": r'\b(scal\w*|performance|uptime|SLA|reliab\w*|availab\w*)',
    "integration": r'\b(integrat\w*|API|connect\w*|plugin|ecosystem|interoper\w*)',
    "usability": r'\b(usab\w*|UX|user\s+experience|intuitive|ease\s+of\s+use|learning\s+curve)',
    "support": r'\b(support\w*|customer\s+success|documentation|training|onboard\w*)',
    "features": r'\b(feature\w*|capabilit\w*|function\w*|workflow\w*|automat\w*)',
}

# --- Objection patterns ---
OBJECTION_PATTERNS = {
    "too_expensive": r'\b(too\s+expensive|over\s*budget|cost\s+prohibitive|pric(?:ey|y))',
    "security_concern": r'\b(security\s+concern|not\s+compliant|failed\s+audit|data\s+breach)',
    "poor_integration": r'\b(doesn.t\s+integrat\w*|no\s+API|lack.*integrat\w*|incompatible)',
    "complexity": r'\b(too\s+complex|steep\s+learning|hard\s+to\s+use|complicated)',
    "vendor_lock_in": r'\b(vendor\s+lock|lock.?in|proprietary|switching\s+cost)',
    "missing_feature": r'\b(missing\s+feature|doesn.t\s+support|no\s+support\s+for|lack)',
}

# --- Industry patterns ---
INDUSTRY_PATTERNS = {
    "healthcare": r'\b(health\w*|medical|hospital|pharma\w*|HIPAA|patient|clinical)',
    "finance": r'\b(financ\w*|bank\w*|fintech|trading|payment\w*|insur(?:ance)?)',
    "education": r'\b(educ\w*|university|school|learning|LMS|academic)',
    "ecommerce": r'\b(ecommerce|e-commerce|retail|shop\w*|marketplace|cart)',
    "technology": r'\b(tech\w*|software|SaaS|startup|developer|engineering)',
    "manufacturing": r'\b(manufactur\w*|supply\s+chain|logistics|warehouse|inventory)',
    "government": r'\b(government|federal|public\s+sector|FedRAMP|agency)',
    "media": r'\b(media|publish\w*|content|broadcast\w*|entertainment|streaming)',
}


def _match_patterns(text: str, pattern_dict: dict) -> list[str]:
    """Return all pattern keys that match in text."""
    text_lower = text.lower()
    matches = []
    for key, patterns in pattern_dict.items():
        if isinstance(patterns, list):
            for p in patterns:
                if re.search(p, text_lower, re.IGNORECASE):
                    matches.append(key)
                    break
        else:
            if re.search(patterns, text_lower, re.IGNORECASE):
                matches.append(key)
    return matches


def _compute_confidence(
    workflow_matches: list[str],
    criteria_matches: list[str],
    objection_matches: list[str],
    industry_matches: list[str],
    text_length: int,
) -> float:
    """
    Heuristic confidence score 0.0–1.0.
    Rewards: multiple signal types found, reasonable text length.
    Penalizes: very short text, no structured signals.

    Note: All rows reaching this point have already passed precision gates
    (decision verb + software noun + reasoning phrase), so the baseline
    is higher than a naive scorer would assign.
    """
    score = 0.0

    # Signal type count — how many distinct signal categories are present
    signal_types = sum([
        bool(workflow_matches),
        bool(criteria_matches),
        bool(objection_matches),
        bool(industry_matches),
    ])

    # Workflow signal — primary indicator
    if workflow_matches:
        score += 0.32
        if len(workflow_matches) > 1:
            score += 0.05
    # Criteria signal — strong buying indicator
    if criteria_matches:
        score += 0.26
        if len(criteria_matches) > 1:
            score += 0.08
    # Objection signal (strong buying discussion indicator)
    if objection_matches:
        score += 0.12
    # Industry signal
    if industry_matches:
        score += 0.07

    # Multi-signal bonus: rows with 2+ signal types are high-quality
    # (all rows already passed precision gates: verb + noun + reasoning)
    if signal_types >= 3:
        score += 0.12
    elif signal_types >= 2:
        score += 0.10

    # Text length bonus
    if 80 < text_length < 240:
        score += 0.05
    elif text_length >= 50:
        score += 0.02

    return min(round(score, 2), 1.0)


def extract_row(
    text: str,
    source_url: str,
    crawl_date: str,
    matched_keywords: list[str],
    max_text_length: int = 240,
) -> ExtractedRow | None:
    """
    Extract a structured row from a scored passage.
    Returns None if no meaningful signals are found.
    """
    if not text or len(text.strip()) < 30:
        return None

    workflow_matches = _match_patterns(text, WORKFLOW_PATTERNS)
    criteria_matches = _match_patterns(text, CRITERIA_PATTERNS)
    objection_matches = _match_patterns(text, OBJECTION_PATTERNS)
    industry_matches = _match_patterns(text, INDUSTRY_PATTERNS)

    # Must have at least one structured signal
    if not (workflow_matches or criteria_matches or objection_matches):
        return None

    # Constraint 4: require at least one of criteria or workflow_step
    if not criteria_matches and not workflow_matches:
        return None

    confidence = _compute_confidence(
        workflow_matches, criteria_matches, objection_matches,
        industry_matches, len(text),
    )

    # Truncate text fields to max length
    decision_context = text[:max_text_length]
    if len(text) > max_text_length:
        decision_context = text[:max_text_length - 3] + "..."

    return ExtractedRow(
        decision_context=decision_context,
        criteria=", ".join(criteria_matches) if criteria_matches else "",
        objection=", ".join(objection_matches) if objection_matches else "",
        workflow_step=", ".join(workflow_matches) if workflow_matches else "",
        industry_hint=", ".join(industry_matches) if industry_matches else "",
        confidence=confidence,
        source_url=source_url,
        crawl_date=crawl_date,
        matched_keywords=", ".join(matched_keywords),
    )


def extract_batch(
    passages: list[dict],
    max_text_length: int = 240,
) -> list[ExtractedRow]:
    """
    Extract structured rows from a batch of scored passages.
    Each passage dict must have: text, source_url, crawl_date, matched_keywords.
    """
    rows = []
    for p in passages:
        row = extract_row(
            text=p["text"],
            source_url=p.get("source_url", ""),
            crawl_date=p.get("crawl_date", ""),
            matched_keywords=p.get("matched_keywords", []),
            max_text_length=max_text_length,
        )
        if row is not None:
            rows.append(row)

    logger.info(f"Extracted {len(rows)} rows from {len(passages)} passages")
    return rows
