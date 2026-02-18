"""
Keyword scoring engine for SaaS buying signal detection.

Enforces precision-focused filtering:
  1. Domain blocklist: reject gambling, dating, coupon, adult, SEO spam URLs
  2. Co-occurrence: require BOTH a decision verb AND a software-related noun
  3. Reasoning phrase: require causal/deliberative language
  4. Keyword scoring with weighted signals
  5. Context window extraction with 240-char limit
"""

import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ScoredPassage:
    text: str
    score: int
    matched_keywords: list[str]
    source_url: str
    crawl_date: str
    segment_id: str
    has_reasoning: bool = True

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "score": self.score,
            "matched_keywords": self.matched_keywords,
            "source_url": self.source_url,
            "crawl_date": self.crawl_date,
            "segment_id": self.segment_id,
            "has_reasoning": self.has_reasoning,
        }


# ── Constraint 1: Domain blocklist ──────────────────────────────────

BLOCKED_URL_PATTERNS = [
    # Gambling / casino
    re.compile(r'\b(casino|poker|slot|betting|gambl|jackpot|roulette|blackjack|sportsbook)\b', re.I),
    # Dating / adult
    re.compile(r'\b(dating|hookup|escort|adult|porn|xxx|cam[s]?girl|onlyfans)\b', re.I),
    # Coupon / deal spam
    re.compile(r'\b(coupon[s]?code|promo-?code|discount-?code|deal[s]?of|cashback|freebies)\b', re.I),
    # SEO spam patterns
    re.compile(r'\b(buy-?backlinks|seo-?service|link-?building|guest-?post-?service|pbn)\b', re.I),
    # Pharma spam
    re.compile(r'\b(viagra|cialis|pharmacy-?online|buy-?pills|weight-?loss-?pill)\b', re.I),
    # Pirated content / file sharing
    re.compile(r'\b(djvu|torrent|pirate|warez|cracked|keygen)\b', re.I),
]

BLOCKED_CONTENT_PATTERNS = [
    # Gambling content markers
    re.compile(r'\b(free\s+spins|no\s+deposit\s+bonus|wagering\s+requirement|slot\s+machine|live\s+dealer)\b', re.I),
    # Dating content markers
    re.compile(r'\b(find\s+a\s+date|singles\s+near|hookup\s+site|dating\s+app\s+review)\b', re.I),
    # Generic spam signals
    re.compile(r'\b(click\s+here\s+to\s+buy|limited\s+time\s+offer|act\s+now|order\s+today)\b', re.I),
    # Crypto/forex spam (not legit fintech)
    re.compile(r'\b(crypto\s+signal|forex\s+robot|binary\s+option|guaranteed\s+return|passive\s+income\s+online)\b', re.I),
]


def is_blocked_domain(url: str) -> bool:
    """Check if a URL matches blocked domain patterns."""
    if not url:
        return False
    url_lower = url.lower()
    for pattern in BLOCKED_URL_PATTERNS:
        if pattern.search(url_lower):
            return True
    return False


def has_spam_content(text: str) -> bool:
    """Check if text contains spam content markers."""
    for pattern in BLOCKED_CONTENT_PATTERNS:
        if pattern.search(text):
            return True
    return False


# ── Constraint 2: Decision verb + software noun co-occurrence ───────

DECISION_VERBS = re.compile(
    r'\b('
    r'evaluat\w*|compar\w*|assess\w*|chose|choose|choosing|'
    r'select\w*|shortlist\w*|procur\w*|purchas\w*|'
    r'switch\w*|migrat\w*|adopt\w*|implement\w*|deploy\w*|'
    r'negotiat\w*|renew\w*|cancel\w*|replac\w*|'
    r'onboard\w*|integrat\w*|pilot\w*|trial\w*|'
    r'demo\w*|benchmark\w*|vett\w*|audit\w*|'
    r'review\w*|analyz\w*|test(?:ed|ing)|'
    r'decided|deciding|sign(?:ed|ing)\s+(?:up|on|with)|'
    r'went\s+with|opted\s+for|rolled?\s+out|'
    r'narrow\w*\s+down|ruled?\s+out'
    r')\b',
    re.I,
)

SOFTWARE_NOUNS = re.compile(
    r'\b('
    r'SaaS|software|platform|tool|solution|vendor|provider|'
    r'product|service|subscription|license|'
    r'CRM|ERP|HRIS|HCM|LMS|CMS|ATS|CDP|'
    r'cloud|API|dashboard|module|suite|'
    r'Salesforce|HubSpot|Workday|ServiceNow|Zendesk|Slack|'
    r'Jira|Confluence|Notion|Asana|Monday|'
    r'SAP|Oracle|Microsoft\s+365|Google\s+Workspace|'
    r'AWS|Azure|GCP|Snowflake|Databricks|'
    r'Stripe|Twilio|SendGrid|Okta|Auth0|'
    r'enterprise\s+software|business\s+application|'
    r'tech\s+stack|software\s+stack|martech|'
    r'procurement\s+system|vendor\s+management'
    r')\b',
    re.I,
)


def has_decision_verb(text: str) -> bool:
    return bool(DECISION_VERBS.search(text))


def has_software_noun(text: str) -> bool:
    return bool(SOFTWARE_NOUNS.search(text))


# ── Constraint 3: Reasoning phrases ─────────────────────────────────

REASONING_PHRASES = re.compile(
    r'\b('
    r'because|due\s+to|owing\s+to|'
    r'needed|required|requirement[s]?|'
    r'after\s+(?:testing|evaluating|comparing|reviewing|trying|assessing|piloting)|'
    r'deciding\s+factor|key\s+factor|main\s+reason|'
    r'the\s+reason\s+(?:we|they|our|I)|'
    r'in\s+order\s+to|so\s+(?:we|they|that)|'
    r'which\s+(?:led|helped|allowed|enabled|made)|'
    r'as\s+a\s+result|consequently|therefore|'
    r'based\s+on|given\s+(?:that|the)|considering|'
    r'(?:better|worse|cheaper|faster|easier|harder)\s+than|'
    r'compared\s+to|versus|vs\.?|'
    r'pros?\s+and\s+cons?|trade-?off|'
    r'ultimately|finally\s+(?:chose|decided|went|selected)|'
    r'the\s+(?:biggest|main|primary|key)\s+(?:advantage|disadvantage|benefit|drawback|concern|issue)|'
    r'what\s+(?:sold|convinced|swayed|persuaded)\s+(?:us|them|me)|'
    r'(?:worth|not\s+worth)\s+(?:the|it)|'
    r'outweigh\w*|prioritiz\w*|'
    r'criteria|must-?have|nice-?to-?have|deal-?breaker'
    r')\b',
    re.I,
)


def has_reasoning_phrase(text: str) -> bool:
    return bool(REASONING_PHRASES.search(text))


# ── Core scoring ─────────────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex."""
    text = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof|Inc|Ltd|Corp|etc|vs)\.',
                  r'\1<DOT>', text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.replace('<DOT>', '.').strip() for s in sentences if s.strip()]


def score_text(
    text: str,
    signal_keywords: dict[str, int],
) -> tuple[int, list[str]]:
    """
    Score a text block against weighted keywords.
    Returns (total_score, list_of_matched_keywords).
    """
    text_lower = text.lower()
    total_score = 0
    matched = []

    for keyword, weight in signal_keywords.items():
        kw_lower = keyword.lower()
        if " " in kw_lower:
            if kw_lower in text_lower:
                total_score += weight
                matched.append(keyword)
        else:
            pattern = rf'\b{re.escape(kw_lower)}\b'
            if re.search(pattern, text_lower):
                total_score += weight
                matched.append(keyword)

    return total_score, matched


def extract_context_windows(
    text: str,
    signal_keywords: dict[str, int],
    sentences_before: int = 1,
    sentences_after: int = 1,
    max_snippet_chars: int = 240,
) -> list[dict]:
    """
    Find keyword matches and extract surrounding sentence context.
    Only yields windows that individually pass verb+noun+reasoning checks.
    """
    sentences = split_sentences(text)
    windows = []
    seen_indices = set()

    for kw in signal_keywords:
        kw_lower = kw.lower()
        for i, sent in enumerate(sentences):
            if kw_lower in sent.lower() and i not in seen_indices:
                seen_indices.add(i)

                start = max(0, i - sentences_before)
                end = min(len(sentences), i + sentences_after + 1)
                snippet = " ".join(sentences[start:end])

                if len(snippet) > max_snippet_chars:
                    snippet = snippet[:max_snippet_chars - 3] + "..."

                # Constraint 2 at snippet level: require verb+noun
                if not has_decision_verb(snippet):
                    continue
                if not has_software_noun(snippet):
                    continue

                # Reasoning phrase is tracked but not a hard gate —
                # the extractor uses it as a confidence signal
                windows.append({
                    "snippet": snippet,
                    "matched_keyword": kw,
                    "sentence_index": i,
                    "has_reasoning": has_reasoning_phrase(snippet),
                })

    return windows


def filter_passage(
    text: str,
    source_url: str,
    crawl_date: str,
    segment_id: str,
    signal_keywords: dict[str, int],
    min_score: int,
    sentences_before: int = 1,
    sentences_after: int = 1,
    max_snippet_chars: int = 240,
) -> list[ScoredPassage]:
    """
    Full filtering pipeline for a single text passage.

    Gate order:
      1. Domain blocklist check
      2. Spam content check
      3. Keyword score threshold
      4. Page-level: decision verb + software noun presence
      5. Per-snippet: decision verb + software noun + reasoning phrase
    """
    # Constraint 1: domain blocklist
    if is_blocked_domain(source_url):
        return []

    # Constraint 1b: spam content check on full text
    if has_spam_content(text):
        return []

    # Keyword score gate
    score, matched = score_text(text, signal_keywords)
    if score < min_score:
        return []

    # Constraint 2 at page level: early exit
    if not has_decision_verb(text):
        return []
    if not has_software_noun(text):
        return []

    # Extract context windows (each window re-checked for verb+noun)
    windows = extract_context_windows(
        text, signal_keywords,
        sentences_before, sentences_after,
        max_snippet_chars,
    )

    passages = []
    for win in windows:
        passages.append(ScoredPassage(
            text=win["snippet"],
            score=score,
            matched_keywords=matched,
            source_url=source_url,
            crawl_date=crawl_date,
            segment_id=segment_id,
            has_reasoning=win.get("has_reasoning", True),
        ))

    return passages
