#!/usr/bin/env python3
"""
Stage 2: Filtering — parallel, streaming from combined file.

Reads ingested_records.json.gz (single source of truth), splits into
chunks, filters each chunk in parallel worker processes, then merges
results. Never reads from shards (those are deleted after merge).

After this script succeeds the shards/ directory is removed to
reclaim disk.
"""

import gzip
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config_loader import get_config, resolve_path
from lib.keyword_scorer import filter_passage
from lib.perf_logger import PerfLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("02_filter")


# ── Worker (runs in subprocess) ──────────────────────────────────────

def _filter_chunk(args: tuple) -> tuple[list[dict], dict]:
    """Filter a list of records; return (passages, keyword_hit_counts)."""
    records, signal_keywords, min_score, ctx_before, ctx_after, max_snippet = args

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from lib.keyword_scorer import filter_passage

    passages = []
    kw_hits: dict[str, int] = {}

    for record in records:
        text = record.get("text", "")
        if not text or len(text) < 50:
            continue
        hits = filter_passage(
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
        for p in hits:
            d = p.to_dict()
            # Store full_text (up to 8000 chars) for gate evaluation in stage 3
            d["full_text"] = text[:8000]
            passages.append(d)
            for kw in p.matched_keywords:
                kw_hits[kw] = kw_hits.get(kw, 0) + 1

    return passages, kw_hits


# ── Main ─────────────────────────────────────────────────────────────

def main():
    cfg = get_config()
    filter_cfg = cfg["filtering"]

    signal_keywords = filter_cfg["signal_keywords"]
    min_score       = filter_cfg["min_score"]
    ctx_before      = filter_cfg["context_sentences_before"]
    ctx_after       = filter_cfg["context_sentences_after"]
    max_snippet     = filter_cfg["max_snippet_chars"]

    raw_dir      = resolve_path("data/raw")
    filtered_dir = resolve_path("data/filtered")
    filtered_dir.mkdir(parents=True, exist_ok=True)

    perf = PerfLogger()
    perf.start_stage("filtering")

    # ── Load records from combined file (single source of truth) ────
    input_path = raw_dir / "ingested_records.json.gz"
    if not input_path.exists():
        logger.error(f"Ingested records not found at {input_path}")
        sys.exit(1)

    logger.info(f"Loading records from {input_path} ...")
    t0 = time.time()
    with gzip.open(input_path, "rt", encoding="utf-8") as f:
        records = json.load(f)
    logger.info(f"Loaded {len(records):,} records in {time.time()-t0:.1f}s")

    # ── Parallel filtering ───────────────────────────────────────────
    workers = max(1, min(os.cpu_count() - 1, 8))
    chunk_size = max(500, len(records) // (workers * 4))
    chunks = [records[i:i+chunk_size] for i in range(0, len(records), chunk_size)]
    logger.info(f"Filtering {len(records):,} records in {len(chunks)} chunks "
                f"using {workers} workers (chunk_size={chunk_size})")

    args_list = [
        (chunk, signal_keywords, min_score, ctx_before, ctx_after, max_snippet)
        for chunk in chunks
    ]

    all_passages: list[dict] = []
    kw_hit_counts: dict[str, int] = {}
    records_with_matches = 0
    done = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_filter_chunk, args): i
                   for i, args in enumerate(args_list)}
        for future in as_completed(futures):
            chunk_passages, chunk_hits = future.result()
            all_passages.extend(chunk_passages)
            for kw, cnt in chunk_hits.items():
                kw_hit_counts[kw] = kw_hit_counts.get(kw, 0) + cnt
            if chunk_passages:
                records_with_matches += 1
            done += 1
            logger.info(f"  Chunk {done}/{len(chunks)} done — "
                        f"{len(chunk_passages)} passages "
                        f"(running total: {len(all_passages):,})")

    logger.info(f"Filtering complete: {len(all_passages):,} passages from "
                f"{len(records):,} records")

    # ── Save filtered passages ───────────────────────────────────────
    output_path = filtered_dir / "scored_passages.json.gz"
    logger.info(f"Saving passages to {output_path} ...")
    with gzip.open(output_path, "wt", encoding="utf-8") as f:
        json.dump(all_passages, f, ensure_ascii=False)
    logger.info(f"Saved ({output_path.stat().st_size / 1_048_576:.1f} MB)")

    # ── Save filter stats ────────────────────────────────────────────
    stats = {
        "stage": "filtering",
        "records_processed": len(records),
        "records_with_matches": records_with_matches,
        "total_passages": len(all_passages),
        "keyword_hit_counts": kw_hit_counts,
        "config": {
            "min_score": min_score,
            "context_window": f"{ctx_before}+1+{ctx_after} sentences",
            "max_snippet_chars": max_snippet,
            "workers": workers,
        },
        "pipeline_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    stats_path = filtered_dir / "filter_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # ── Remove shards directory to reclaim disk ──────────────────────
    shard_dir = raw_dir / "shards"
    if shard_dir.exists():
        import shutil
        shard_size_mb = sum(
            sf.stat().st_size for sf in shard_dir.glob("*.json.gz")
        ) / 1_048_576
        shutil.rmtree(shard_dir)
        logger.info(f"Deleted shards/ directory ({shard_size_mb:.0f} MB freed)")

    perf.end_stage()
    perf.record("records_processed", len(records))
    perf.record("passages_extracted", len(all_passages))
    perf.save_report(resolve_path("data/manifests/filter_perf.json"))

    logger.info("=" * 50)
    logger.info("FILTERING COMPLETE")
    logger.info(f"  Records processed : {len(records):,}")
    logger.info(f"  Passages extracted: {len(all_passages):,}")
    logger.info("=" * 50)

    return len(all_passages)


if __name__ == "__main__":
    count = main()
    sys.exit(0 if count > 0 else 1)
