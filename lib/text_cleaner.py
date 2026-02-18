"""
Text cleaning for SaaS buying intelligence extraction.

Removes:
  1. Common navigation menu terms and fragments
  2. Repeated lines/headings
  3. Long sequences of single-word navigation items
  4. Biography/job description content (unless it has procurement language)
  5. Collapses excessive whitespace

Applied BEFORE the decision-narrative gate to ensure the gate
evaluates clean, meaningful text.
"""

import re
from typing import Optional

# ── Navigation / menu terms to strip ──────────────────────────────────

_NAV_TERMS = re.compile(
    r'\b('
    r'skip to content|skip to main|skip to navigation|'
    r'home|about us|about|contact us|contact|services|'
    r'our services|products|solutions|pricing|blog|'
    r'resources|careers|faq|faqs|terms of service|'
    r'terms of use|privacy policy|privacy|cookie policy|'
    r'sitemap|site map|sign in|sign up|log in|log out|'
    r'login|logout|register|subscribe|unsubscribe|'
    r'follow us|share this|tweet|pin it|'
    r'all rights reserved|copyright|'
    r'read more|learn more|see more|view all|'
    r'get started|get a demo|request demo|book a demo|'
    r'free trial|start free|try free|try it free|'
    r'menu|navigation|search|close|back to top|'
    r'next|previous|page \d+|'
    r'accept cookies|cookie preferences|we use cookies|'
    r'this site uses cookies|accept all cookies'
    r')\b',
    re.IGNORECASE,
)

# Menu-like fragments: "Home | About | Pricing | Contact"
_PIPE_MENU_RE = re.compile(
    r'^(?:[A-Za-z][A-Za-z ]{0,20}\s*[|/\u2022\u00b7]\s*){2,}[A-Za-z][A-Za-z ]{0,20}\s*$',
    re.MULTILINE,
)

# Breadcrumbs: "Home > Products > CRM > Pricing"
_BREADCRUMB_RE = re.compile(
    r'^(?:[A-Za-z ]+\s*>\s*){2,}[A-Za-z ]+\s*$',
    re.MULTILINE,
)

# Repeated short lines (nav-like): "Login\nSign Up\nPricing\nBlog"
_SHORT_LINE_CLUSTER_RE = re.compile(
    r'(?:^.{1,20}\n){4,}',
    re.MULTILINE,
)

# Footer boilerplate line
_FOOTER_RE = re.compile(
    r'(?:\u00a9|copyright|\bAll\s+Rights\s+Reserved\b|Privacy\s+Policy|'
    r'Terms\s+of\s+(?:Service|Use)|Cookie\s+(?:Policy|Settings)|'
    r'Sitemap|Unsubscribe).*$',
    re.IGNORECASE | re.MULTILINE,
)

# Social media link clusters
_SOCIAL_RE = re.compile(
    r'(?:Follow\s+us|Share\s+(?:on|this)|Tweet|Pin\s+it|'
    r'Facebook|Twitter|LinkedIn|Instagram|YouTube)\s*[|/\u2022\u00b7\s]*'
    r'(?:Facebook|Twitter|LinkedIn|Instagram|YouTube)',
    re.IGNORECASE,
)


# ── Bio / job description detection ───────────────────────────────────

_BIO_PATTERNS = re.compile(
    r'\b('
    # Job title patterns
    r'(?:senior|junior|lead|staff|principal|chief|head of|vp of|director of|'
    r'manager of|associate|intern)\s+'
    r'(?:software|data|product|project|program|marketing|sales|'
    r'account|customer|business|operations|engineering|design|'
    r'devops|cloud|security|compliance|hr|finance|support)\s*'
    r'(?:engineer|developer|architect|analyst|scientist|manager|'
    r'director|officer|specialist|consultant|coordinator|lead|'
    r'representative|executive|strategist|designer)?|'
    # Resume/bio language
    r'years?\s+of\s+experience|'
    r'responsible\s+for|'
    r'job\s+(?:description|responsibilities|duties|requirements|qualifications)|'
    r'key\s+responsibilities|'
    r'(?:bachelor|master|phd|mba|degree)\s+in|'
    r'graduated\s+from|'
    r'resume|curriculum\s+vitae|'
    r'looking\s+for\s+(?:a|an)\s+(?:job|position|role|opportunity)|'
    r'apply\s+(?:now|today|here)|'
    r'(?:full|part)\s*-?\s*time\s+(?:position|role|opportunity|job)|'
    r'salary\s+range|compensation\s+package|'
    r'we\s+are\s+(?:looking|hiring|seeking)'
    r')\b',
    re.IGNORECASE,
)

# Procurement language that overrides bio detection
_PROCUREMENT_OVERRIDE = re.compile(
    r'\b('
    r'chose|selected|evaluated|adopted|purchased|procured|'
    r'rejected|replaced|switched|migrated|'
    r'vendor\s+selection|procurement\s+(?:process|decision)|'
    r'buying\s+decision|evaluation\s+criteria|'
    r'shortlisted|compared\s+(?:vendors|solutions|platforms|tools)'
    r')\b',
    re.IGNORECASE,
)


def is_bio_or_job_content(text: str) -> bool:
    """
    Detect biography/job description content.
    Returns False if text also contains procurement language
    (indicating it's about vendor selection IN a hiring context).
    """
    if not _BIO_PATTERNS.search(text):
        return False
    # Override: if the text also has procurement language, keep it
    if _PROCUREMENT_OVERRIDE.search(text):
        return False
    return True


def clean_navigation_text(text: str) -> str:
    """
    Remove navigation artifacts from text.

    Steps:
      1. Remove pipe-separated menu bars
      2. Remove breadcrumbs
      3. Remove clusters of short lines (nav items)
      4. Remove footer boilerplate
      5. Remove social media clusters
      6. Strip common nav terms
      7. Collapse excessive whitespace
    """
    cleaned = text

    # Structural nav patterns
    cleaned = _PIPE_MENU_RE.sub('', cleaned)
    cleaned = _BREADCRUMB_RE.sub('', cleaned)
    cleaned = _SHORT_LINE_CLUSTER_RE.sub('', cleaned)
    cleaned = _FOOTER_RE.sub('', cleaned)
    cleaned = _SOCIAL_RE.sub('', cleaned)

    # Remove isolated nav terms (only when they appear as standalone
    # short phrases, not when embedded in longer sentences)
    lines = cleaned.split('\n')
    filtered_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip very short lines that are just nav terms
        if len(stripped) < 40 and _NAV_TERMS.search(stripped):
            # Check if the line is MOSTLY nav terms
            nav_match = _NAV_TERMS.findall(stripped)
            nav_chars = sum(len(m) for m in nav_match)
            if nav_chars > len(stripped) * 0.5:
                continue
        filtered_lines.append(line)

    cleaned = '\n'.join(filtered_lines)

    # Collapse whitespace
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)

    return cleaned.strip()


def is_navigation_noise(text: str, threshold: float = 0.5) -> bool:
    """
    Check if text is predominantly navigation/menu content.
    Returns True if more than `threshold` fraction of words are nav terms.
    """
    words = text.split()
    if len(words) < 5:
        return True  # Very short text is noise

    nav_hits = len(_NAV_TERMS.findall(text))
    return nav_hits / len(words) > threshold


# ── Promotional / CTA content detection ───────────────────────────────
# Catches marketing copy that happens to contain decision verbs
# (e.g., "Sign up for a free trial", "Choose your plan")

_PROMO_PATTERNS = re.compile(
    r'\b('
    r'sign\s+up\s+(?:for|now|today|here|free)|'
    r'create\s+(?:an?\s+)?account|'
    r'(?:start|begin)\s+(?:your\s+)?free\s+trial|'
    r'try\s+(?:it\s+)?free|'
    r'get\s+(?:started|a\s+demo)|'
    r'book\s+(?:a\s+)?demo|'
    r'request\s+(?:a\s+)?demo|'
    r'schedule\s+(?:a\s+)?demo|'
    r'no\s+credit\s+card\s+required|'
    r'limited\s+time\s+offer|'
    r'(?:\d+)\s*%\s*off|'
    r'money\s*-?\s*back\s+guarantee|'
    r'cancel\s+(?:any\s*time|at\s+any\s+time)|'
    r'subscribe\s+to\s+(?:our|the)\s+newsletter|'
    r'join\s+(?:\d+[\+k]?\s+)?(?:companies|teams|users|customers)|'
    r'trusted\s+by\s+(?:\d+[\+k]?\s+)?(?:companies|teams|businesses)'
    r')\b',
    re.IGNORECASE,
)


def is_promotional_content(text: str) -> bool:
    """
    Detect promotional/CTA content that uses decision language
    in a marketing context rather than a decision-narrative context.
    """
    promo_count = len(_PROMO_PATTERNS.findall(text))
    # If 2+ promo signals, it's marketing copy
    return promo_count >= 2


def full_clean(text: str, source_url: Optional[str] = None) -> tuple[str, Optional[str]]:
    """
    Full text cleaning pipeline.

    Returns:
      (cleaned_text, drop_reason)
      drop_reason is None if text is clean, or a reason string if it should be dropped.
    """
    # Step 1: Clean navigation artifacts
    cleaned = clean_navigation_text(text)

    # Step 2: Check if remaining text is just nav noise
    if is_navigation_noise(cleaned):
        return cleaned, "dropped_navigation_noise"

    # Step 3: Check for bio/job content
    if is_bio_or_job_content(cleaned):
        return cleaned, "dropped_bio_or_job_description"

    # Step 4: Check for promotional / CTA content
    if is_promotional_content(cleaned):
        return cleaned, "dropped_navigation_noise"

    return cleaned, None
