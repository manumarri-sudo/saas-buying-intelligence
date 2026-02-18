"""
Keyword scoring engine for SaaS buying signal detection.

Enforces precision-focused filtering:
  1. Domain blocklist: reject gambling, dating, coupon, adult, SEO spam URLs
  2. Co-occurrence: require BOTH a decision verb AND a software-related noun
  3. Narrative reasoning markers: prioritize deliberative content
  4. Navigation text cleaning: strip menus, headers, nav artifacts
  5. Docs/help center detection: flag descriptive-only pages
  6. Keyword scoring with weighted signals
  7. Context window extraction with 240-char limit
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
    has_narrative: bool = False
    is_docs_page: bool = False

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "score": self.score,
            "matched_keywords": self.matched_keywords,
            "source_url": self.source_url,
            "crawl_date": self.crawl_date,
            "segment_id": self.segment_id,
            "has_reasoning": self.has_reasoning,
            "has_narrative": self.has_narrative,
            "is_docs_page": self.is_docs_page,
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


# ── Narrative reasoning markers ──────────────────────────────────────
# Lightweight markers that signal the text is a decision narrative,
# not just a product feature description.

NARRATIVE_MARKERS = re.compile(
    r'\b('
    r'because|after\s+testing|decided\s+to|evaluated|'
    r'chose|rejected|due\s+to|needed|'
    r'challenge|concern|issue|'
    r'struggled|switched\s+(?:from|to)|'
    r'we\s+(?:chose|picked|went|opted|decided|needed|found|tried|tested)|'
    r'our\s+(?:team|company|org|experience)|'
    r'pain\s+point|deal\s*breaker|lesson\s+learned|'
    r'in\s+hindsight|looking\s+back|turned\s+out|'
    r'pros?\s+and\s+cons?|trade-?off|downside|upside'
    r')\b',
    re.I,
)


def has_narrative_marker(text: str) -> bool:
    """Check for first-person or deliberative reasoning markers."""
    return bool(NARRATIVE_MARKERS.search(text))


# ── Navigation / boilerplate cleaner ─────────────────────────────────
# Removes site-chrome artifacts that leak into Common Crawl text.

_NAV_PATTERNS = [
    # Menu-like fragments: "Home | About | Pricing | Contact"
    re.compile(r'^(?:[A-Z][a-z]+\s*[|/•·]\s*){3,}[A-Z][a-z]+\s*$', re.M),
    # Breadcrumbs: "Home > Products > CRM > Pricing"
    re.compile(r'^(?:[A-Za-z ]+\s*>\s*){2,}[A-Za-z ]+\s*$', re.M),
    # Repeated short lines that look like nav (e.g. "Login\nSign Up\nPricing\nBlog")
    re.compile(r'(?:^.{1,20}\n){4,}', re.M),
    # Footer boilerplate
    re.compile(
        r'(?:©|copyright|\bAll\s+Rights\s+Reserved\b|Privacy\s+Policy|'
        r'Terms\s+of\s+(?:Service|Use)|Cookie\s+(?:Policy|Settings)|'
        r'Sitemap|Unsubscribe).*$',
        re.I | re.M,
    ),
    # Social media link clusters
    re.compile(
        r'(?:Follow\s+us|Share\s+(?:on|this)|Tweet|Pin\s+it|'
        r'Facebook|Twitter|LinkedIn|Instagram|YouTube)\s*[|/•·\s]*'
        r'(?:Facebook|Twitter|LinkedIn|Instagram|YouTube)',
        re.I,
    ),
    # "Skip to content" / "Skip to main" accessibility links
    re.compile(r'^Skip\s+to\s+(?:content|main|navigation)\s*$', re.I | re.M),
    # Cookie consent banners
    re.compile(
        r'(?:We\s+use\s+cookies|This\s+(?:site|website)\s+uses\s+cookies|'
        r'Accept\s+(?:all\s+)?cookies|Cookie\s+preferences).*?(?:\.|$)',
        re.I,
    ),
]


def clean_nav_artifacts(text: str) -> str:
    """Strip navigation fragments, headers, footers, and boilerplate."""
    cleaned = text
    for pattern in _NAV_PATTERNS:
        cleaned = pattern.sub('', cleaned)
    # Collapse resulting blank lines
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


# ── Docs / help center detection ─────────────────────────────────────
# Flag pages that are product docs or help centers — these describe
# features but rarely contain buying decision reasoning.

DOCS_URL_PATTERNS = re.compile(
    r'(?:'
    r'/docs?/|/help/|/support/|/kb/|/knowledge-?base/|'
    r'/api-?(?:reference|docs?)/|/developer[s]?/|'
    r'/guide[s]?/|/tutorial[s]?/|/how-?to/|/faq/|'
    r'/changelog/|/release-?notes?/|/reference/|'
    r'docs\.|help\.|support\.|wiki\.|learn\.|'
    r'developer[s]?\.'
    r')',
    re.I,
)


def is_docs_or_help_page(url: str) -> bool:
    """Detect documentation, help center, and API reference URLs."""
    if not url:
        return False
    return bool(DOCS_URL_PATTERNS.search(url))


# ── Source targeting: high-signal page types ──────────────────────────
# Pages likely to contain real decision narratives get a score boost.
# This shifts filtering to surface case studies, migration writeups,
# and engineering blogs over generic product descriptions.

HIGH_SIGNAL_URL_PATTERNS = re.compile(
    r'(?:'
    # Case studies / success stories
    r'/case-?stud(?:y|ies)|/success-?stor(?:y|ies)|/customer-?stor(?:y|ies)|'
    r'/customer-?case|/client-?stor(?:y|ies)|'
    # Migration / switching content
    r'/migrat(?:e|ion|ing)|/switch(?:ing|ed)|/mov(?:ing|ed)-?(?:from|to|away)|'
    r'/from-\w+-to-\w+|/why-we-(?:chose|switched|moved|use|replaced)|'
    # Comparison / evaluation content
    r'/compar(?:e|ison|ing)|/vs-?|/versus|/alternatives?|/evaluation|'
    r'/vendor-?(?:compar|select|assess|review)|/tool-?compar|'
    # RFP / procurement guidance
    r'/rfp|/procurement|/vendor-?select|/buying-?guide|/software-?select|'
    # Engineering / retrospective blogs
    r'/engineering-?blog|/tech-?blog|/retrospect|/post-?mortem|'
    r'/lessons-?learned|/how-we-|/why-we-|'
    # Security / compliance evaluations
    r'/security-?(?:review|assessment|evaluat)|/compliance-?(?:review|check)|'
    # Reviews / analysis
    r'/review|/analysis|/deep-?dive|/breakdown'
    r')',
    re.I,
)

HIGH_SIGNAL_DOMAIN_PATTERNS = re.compile(
    r'(?:'
    r'g2\.com|capterra\.com|trustradius\.com|gartner\.com|'
    r'forrester\.com|idc\.com|infoq\.com|'
    r'engineering\.(?:\w+\.)?(?:com|io)|'
    r'blog\.(?:\w+\.)?(?:com|io)|'
    r'medium\.com|dev\.to|hashnode\.(?:com|dev)|'
    r'hbr\.org|mckinsey\.com|deloitte\.com'
    r')',
    re.I,
)

# Low-signal URL patterns to downweight (not block — just lower effective score)
LOW_SIGNAL_URL_PATTERNS = re.compile(
    r'(?:'
    r'/product[s]?/?$|/feature[s]?/?$|/pricing/?$|/plans/?$|'
    r'/about/?$|/about-us/?$|/contact/?$|/login/?$|/signup/?$|'
    r'/home/?$|/index|/404|/error|'
    r'job|career|hire|recruit|position|opening'
    r')',
    re.I,
)


def get_source_score_modifier(url: str) -> int:
    """
    Return a score modifier based on URL type.
    High-signal pages: +4 bonus
    Low-signal pages: -4 penalty
    Docs pages: already handled separately
    """
    if not url:
        return 0
    if HIGH_SIGNAL_URL_PATTERNS.search(url) or HIGH_SIGNAL_DOMAIN_PATTERNS.search(url):
        return 4
    if LOW_SIGNAL_URL_PATTERNS.search(url):
        return -4
    return 0


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
    Only yields windows that pass verb+noun checks.
    Tracks narrative markers and reasoning phrases as quality signals.
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

                # Track quality signals (not hard gates)
                windows.append({
                    "snippet": snippet,
                    "matched_keyword": kw,
                    "sentence_index": i,
                    "has_reasoning": has_reasoning_phrase(snippet),
                    "has_narrative": has_narrative_marker(snippet),
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
      3. Navigation artifact cleaning
      4. Keyword score threshold
      5. Page-level: decision verb + software noun presence
      6. Per-snippet: decision verb + software noun (reasoning tracked)
      7. Docs/help center detection (tracked, not gated)
    """
    # Constraint 1: domain blocklist
    if is_blocked_domain(source_url):
        return []

    # Constraint 1b: spam content check on full text
    if has_spam_content(text):
        return []

    # Clean navigation artifacts before scoring
    text = clean_nav_artifacts(text)
    if len(text) < 80:
        return []

    # Keyword score gate with source-type modifier
    score, matched = score_text(text, signal_keywords)
    source_mod = get_source_score_modifier(source_url)
    effective_score = score + source_mod
    if effective_score < min_score:
        return []

    # Constraint 2 at page level: early exit
    if not has_decision_verb(text):
        return []
    if not has_software_noun(text):
        return []

    # Detect docs/help pages (tracked as metadata, not hard-gated)
    is_docs = is_docs_or_help_page(source_url)

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
            has_narrative=win.get("has_narrative", False),
            is_docs_page=is_docs,
        ))

    return passages
