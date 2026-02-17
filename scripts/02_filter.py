#!/usr/bin/env python3
"""
Stage 2: Filtering

Reads ingested records, scores them against SaaS buying signal keywords,
extracts context windows, and writes scored passages to data/filtered/.
"""

import gzip
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config_loader import get_config, resolve_path
from lib.keyword_scorer import filter_passage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("02_filter")


def load_ingested_records(raw_dir: Path) -> list[dict]:
    """Load records from the ingestion stage."""
    input_path = raw_dir / "ingested_records.json.gz"
    if not input_path.exists():
        logger.error(f"Ingested records not found at {input_path}")
        logger.error("Run 01_ingest.py first.")
        return []

    logger.info(f"Loading records from {input_path}")
    with gzip.open(input_path, "rt", encoding="utf-8") as f:
        records = json.load(f)

    logger.info(f"Loaded {len(records)} records")
    return records


def main():
    cfg = get_config()
    filter_cfg = cfg["filtering"]

    signal_keywords = filter_cfg["signal_keywords"]
    min_score = filter_cfg["min_score"]
    ctx_before = filter_cfg["context_sentences_before"]
    ctx_after = filter_cfg["context_sentences_after"]
    max_snippet = filter_cfg["max_snippet_chars"]

    raw_dir = resolve_path("data/raw")
    filtered_dir = resolve_path("data/filtered")
    filtered_dir.mkdir(parents=True, exist_ok=True)

    records = load_ingested_records(raw_dir)
    if not records:
        logger.error("No records to filter. Exiting.")
        sys.exit(1)

    all_passages = []
    stats = {
        "records_processed": 0,
        "records_with_matches": 0,
        "total_passages": 0,
        "keyword_hit_counts": {},
    }

    for i, record in enumerate(records):
        if i % 500 == 0 and i > 0:
            logger.info(f"  Processed {i}/{len(records)} records...")

        text = record.get("text", "")
        if not text or len(text) < 50:
            continue

        stats["records_processed"] += 1

        passages = filter_passage(
            text=text,
            source_url=record.get("url", ""),
            crawl_date=record.get("crawl_date", ""),
            segment_id=record.get("segment_id", ""),
            signal_keywords=signal_keywords,
            min_score=min_score,
            sentences_before=ctx_before,
            sentences_after=ctx_after,
            max_snippet_chars=max_snippet,
        )

        if passages:
            stats["records_with_matches"] += 1
            for p in passages:
                all_passages.append(p.to_dict())
                for kw in p.matched_keywords:
                    stats["keyword_hit_counts"][kw] = (
                        stats["keyword_hit_counts"].get(kw, 0) + 1
                    )

    stats["total_passages"] = len(all_passages)

    # Save filtered passages
    output_path = filtered_dir / "scored_passages.json.gz"
    logger.info(f"Saving {len(all_passages)} passages to {output_path}")

    with gzip.open(output_path, "wt", encoding="utf-8") as f:
        json.dump(all_passages, f, ensure_ascii=False)

    # Save filter stats
    stats_path = filtered_dir / "filter_stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            "stage": "filtering",
            **stats,
            "config": {
                "min_score": min_score,
                "context_window": f"{ctx_before}+1+{ctx_after} sentences",
                "max_snippet_chars": max_snippet,
            },
            "pipeline_timestamp": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
        }, f, indent=2)

    logger.info(f"Filtering complete:")
    logger.info(f"  Records processed: {stats['records_processed']}")
    logger.info(f"  Records with matches: {stats['records_with_matches']}")
    logger.info(f"  Total passages extracted: {stats['total_passages']}")
    logger.info(f"  Stats saved to {stats_path}")

    return len(all_passages)


if __name__ == "__main__":
    count = main()
    sys.exit(0 if count > 0 else 1)
