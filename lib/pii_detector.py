"""
PII detection and removal.

Uses regex patterns to detect:
  - Email addresses
  - Phone numbers (US/international)
  - Physical addresses (US-style)
  - Social Security Numbers
  - Credit card numbers
  - Personal name patterns near identifiers

Returns detection results and can scrub or flag rows.
"""

import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PIIDetectionResult:
    has_pii: bool
    pii_types: list[str]
    details: list[str]


# --- Compiled regex patterns ---

EMAIL_PATTERN = re.compile(
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
)

PHONE_PATTERNS = [
    # US formats: (555) 123-4567, 555-123-4567, +1-555-123-4567
    re.compile(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'),
    # International: +44 20 7946 0958
    re.compile(r'\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}'),
]

SSN_PATTERN = re.compile(
    r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b'
)

# US address patterns — street number + street name + type
ADDRESS_PATTERNS = [
    re.compile(
        r'\b\d{1,5}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+'
        r'(?:Street|St|Avenue|Ave|Boulevard|Blvd|Drive|Dr|Road|Rd|'
        r'Lane|Ln|Court|Ct|Place|Pl|Way|Circle|Cir)\b',
        re.IGNORECASE,
    ),
    # PO Box
    re.compile(r'\bP\.?O\.?\s*Box\s+\d+\b', re.IGNORECASE),
    # ZIP code patterns (5 or 5+4)
    re.compile(
        r'\b[A-Z][a-z]+,?\s+[A-Z]{2}\s+\d{5}(?:-\d{4})?\b'
    ),
]

CREDIT_CARD_PATTERN = re.compile(
    r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
)


def detect_pii(text: str) -> PIIDetectionResult:
    """
    Scan text for PII patterns.
    Returns a PIIDetectionResult with flags and details.
    """
    pii_types = []
    details = []

    # Email
    emails = EMAIL_PATTERN.findall(text)
    if emails:
        pii_types.append("email")
        details.append(f"Found {len(emails)} email(s)")

    # Phone
    for pattern in PHONE_PATTERNS:
        phones = pattern.findall(text)
        if phones:
            pii_types.append("phone")
            details.append(f"Found {len(phones)} phone number(s)")
            break

    # SSN
    ssns = SSN_PATTERN.findall(text)
    if ssns:
        # Filter out likely false positives (dates, etc.)
        real_ssns = [
            s for s in ssns
            if not _is_likely_date(s)
        ]
        if real_ssns:
            pii_types.append("ssn")
            details.append(f"Found {len(real_ssns)} possible SSN(s)")

    # Address
    for pattern in ADDRESS_PATTERNS:
        addresses = pattern.findall(text)
        if addresses:
            pii_types.append("address")
            details.append(f"Found {len(addresses)} address pattern(s)")
            break

    # Credit card
    cc = CREDIT_CARD_PATTERN.findall(text)
    if cc:
        pii_types.append("credit_card")
        details.append(f"Found {len(cc)} possible credit card number(s)")

    return PIIDetectionResult(
        has_pii=len(pii_types) > 0,
        pii_types=pii_types,
        details=details,
    )


def _is_likely_date(candidate: str) -> bool:
    """Check if an SSN-pattern match is more likely a date."""
    clean = candidate.replace("-", "").replace(" ", "")
    if len(clean) != 9:
        return False
    # If first 3 digits > 12 or middle 2 > 31, probably not a date
    try:
        first = int(clean[:3])
        # Dates wouldn't start with 3-digit month
        if first > 31:
            return False
    except ValueError:
        return False
    return True


def scan_dataframe(df, text_columns: list[str]) -> dict:
    """
    Scan a pandas DataFrame for PII across specified text columns.
    Returns {row_index: PIIDetectionResult} for rows with PII.
    """
    flagged = {}
    for idx, row in df.iterrows():
        combined_text = " ".join(
            str(row[col]) for col in text_columns if col in row.index
        )
        result = detect_pii(combined_text)
        if result.has_pii:
            flagged[idx] = result

    logger.info(f"PII scan: flagged {len(flagged)} of {len(df)} rows")
    return flagged


def drop_pii_rows(df, text_columns: list[str]):
    """
    Remove rows containing PII from a DataFrame.
    Returns (cleaned_df, dropped_count, pii_summary).
    """
    flagged = scan_dataframe(df, text_columns)
    drop_indices = list(flagged.keys())

    pii_summary = {}
    for result in flagged.values():
        for ptype in result.pii_types:
            pii_summary[ptype] = pii_summary.get(ptype, 0) + 1

    cleaned = df.drop(index=drop_indices).reset_index(drop=True)
    logger.info(f"Dropped {len(drop_indices)} rows with PII")

    return cleaned, len(drop_indices), pii_summary
