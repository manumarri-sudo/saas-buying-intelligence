"""
Fuzzy deduplication using rapidfuzz.

Compares rows on a specified text field and drops near-duplicates
above a similarity threshold.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def deduplicate_dataframe(
    df,
    field: str = "decision_context",
    threshold: int = 85,
    batch_size: int = 1000,
):
    """
    Remove near-duplicate rows from a DataFrame based on fuzzy string matching.

    Args:
        df: pandas DataFrame
        field: column name to compare
        threshold: similarity threshold (0-100), rows above this are duplicates
        batch_size: process in batches to manage memory

    Returns:
        (deduped_df, duplicate_count)
    """
    try:
        from rapidfuzz import fuzz
    except ImportError:
        logger.warning(
            "rapidfuzz not installed — falling back to exact dedup only"
        )
        return _exact_dedup(df, field)

    if field not in df.columns:
        logger.error(f"Field '{field}' not in DataFrame columns")
        return df, 0

    texts = df[field].fillna("").astype(str).tolist()
    n = len(texts)
    keep_mask = [True] * n

    # Process in manageable chunks
    for i in range(n):
        if not keep_mask[i]:
            continue

        text_i = texts[i]
        if not text_i:
            continue

        # Compare against subsequent rows
        for j in range(i + 1, min(i + batch_size, n)):
            if not keep_mask[j]:
                continue

            text_j = texts[j]
            if not text_j:
                continue

            similarity = fuzz.ratio(text_i, text_j)
            if similarity >= threshold:
                keep_mask[j] = False

    duplicate_count = keep_mask.count(False)
    deduped = df[keep_mask].reset_index(drop=True)

    logger.info(
        f"Deduplication: {duplicate_count} duplicates removed, "
        f"{len(deduped)} rows retained"
    )
    return deduped, duplicate_count


def _exact_dedup(df, field: str):
    """Fallback: exact string deduplication."""
    before = len(df)
    deduped = df.drop_duplicates(subset=[field]).reset_index(drop=True)
    duplicate_count = before - len(deduped)
    logger.info(f"Exact dedup: removed {duplicate_count} duplicates")
    return deduped, duplicate_count
