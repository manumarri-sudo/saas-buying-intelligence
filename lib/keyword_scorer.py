"""
Keyword scoring engine for SaaS buying signal detection.

Scores text passages against a weighted keyword dictionary.
Extracts context windows (surrounding sentences) around matches.
Enforces the 240-character snippet limit.
"""

import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ScoredPassage:
    text: str
    score: int
    matched_keywords: list[str]
    source_url: str
    crawl_date: str
    segment_id: str

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "score": self.score,
            "matched_keywords": self.matched_keywords,
            "source_url": self.source_url,
            "crawl_date": self.crawl_date,
            "segment_id": self.segment_id,
        }


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex."""
    # Handle common abbreviations to avoid false splits
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
        # Use word boundary matching for single words,
        # substring matching for multi-word phrases
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
    Returns list of {snippet, matched_keyword, sentence_index}.
    Enforces max_snippet_chars limit.
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

                # Enforce character limit
                if len(snippet) > max_snippet_chars:
                    snippet = snippet[:max_snippet_chars - 3] + "..."

                windows.append({
                    "snippet": snippet,
                    "matched_keyword": kw,
                    "sentence_index": i,
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
    Returns list of ScoredPassage (one per context window that meets threshold).
    """
    score, matched = score_text(text, signal_keywords)

    if score < min_score:
        return []

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
        ))

    return passages
