#!/usr/bin/env python3
"""
Stage 1b: Targeted Ingestion for Decision Narratives

Ingests from NEW crawl indexes (CC-MAIN-2024-10, CC-MAIN-2024-18,
CC-MAIN-2023-50, CC-MAIN-2023-40) with keywords focused on:
  - case studies / migration writeups
  - vendor comparisons / engineering retrospectives
  - security/compliance evaluations
  - "why we switched" articles

Uses the existing manifest to skip already-processed WET files.
Appends new records to existing ingested_records.json.gz.
Saves shards to data/raw/shards/ as before.
"""

import gzip
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tempfile import NamedTemporaryFile

import requests
from warcio.archiveiterator import ArchiveIterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.config_loader import resolve_path
from lib.manifest import WETManifest
from lib.perf_logger import PerfLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("01_ingest_targeted")

CC_S3_BASE = "https://data.commoncrawl.org"

# ── High-signal prefilter keywords ─────────────────────────────────────
# Much tighter than the generic prefilter — targets decision narratives
TARGETED_KEYWORDS = [
    # Decision verb phrases
    "we chose",
    "we selected",
    "we evaluated",
    "we switched",
    "we migrated",
    "we replaced",
    "we adopted",
    "we rejected",
    "we piloted",
    "we trialed",
    "we procured",
    "we purchased",
    # "why we switched"
    "why we switched",
    "why we chose",
    "why we moved",
    "why we use",
    "why we replaced",
    # Migration language
    "moved from",
    "migrated from",
    "switched from",
    "migrated away from",
    "transitioned from",
    "replaced with",
    "moved away from",
    # Evaluation language
    "vendor evaluation",
    "vendor selection",
    "software evaluation",
    "evaluated several",
    "evaluated multiple",
    "compared vendors",
    "shortlisted vendors",
    "proof of concept",
    "security review",
    "compliance review",
    # Procurement / RFP
    "procurement process",
    "RFP process",
    "request for proposal",
    "vendor assessment",
    "due diligence",
    # Rejection / friction
    "rejected because",
    "decided against",
    "too expensive for",
    "vendor lock-in",
    "lacked support",
    "integration issues",
    "scalability issues",
    "security concerns",
    # Case study / retrospective markers
    "case study",
    "lessons learned",
    "retrospective",
    "post-mortem",
    "engineering blog",
    "tech stack",
    "tool evaluation",
]


def fetch_wet_paths(cc_index: str) -> list[str]:
    url = f"{CC_S3_BASE}/crawl-data/{cc_index}/wet.paths.gz"
    logger.info(f"Fetching WET paths for {cc_index}...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    paths = gzip.decompress(resp.content).decode("utf-8").strip().split("\n")
    logger.info(f"Found {len(paths):,} WET segments available")
    return paths


def _process_single_wet(args: tuple) -> dict:
    """Worker: process one WET file, return retained records."""
    wet_path, cc_index, prefilter_keywords, max_records_scan, shard_dir = args
    segment_id = wet_path.split("/")[-1].replace(".warc.wet.gz", "")
    url = f"{CC_S3_BASE}/{wet_path}"
    records = []
    scanned = 0
    downloaded = 0
    start_time = time.time()

    result = {
        "wet_path": wet_path,
        "segment_id": segment_id,
        "cc_index": cc_index,
        "records_scanned": 0,
        "records_retained": 0,
        "bytes_downloaded": 0,
        "elapsed_seconds": 0,
        "shard_path": "",
        "error": None,
    }

    try:
        resp = requests.get(url, stream=True, timeout=300)
        resp.raise_for_status()

        with NamedTemporaryFile(suffix=".warc.wet.gz", delete=True) as tmp:
            for chunk in resp.iter_content(chunk_size=2_097_152):
                tmp.write(chunk)
                downloaded += len(chunk)
                if downloaded >= 80_000_000:  # 80MB per file
                    break
            tmp.flush()
            tmp.seek(0)

            for record in ArchiveIterator(tmp):
                if record.rec_type != "conversion":
                    continue

                scanned += 1
                if scanned > max_records_scan:
                    break

                target_uri = record.rec_headers.get_header("WARC-Target-URI") or ""
                warc_date = record.rec_headers.get_header("WARC-Date") or ""

                content = record.content_stream().read()
                try:
                    text = content.decode("utf-8", errors="replace")
                except Exception:
                    continue

                if len(text) < 200:
                    continue

                text_lower = text.lower()
                # Higher bar: require at least one targeted keyword
                if any(kw in text_lower for kw in prefilter_keywords):
                    records.append({
                        "url": target_uri,
                        "text": text,
                        "crawl_date": warc_date,
                        "segment_id": segment_id,
                        "cc_index": cc_index,
                        "fetch_timestamp": time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                        ),
                    })

        # Write per-file shard
        shard_path = ""
        if records:
            shard_dir_path = Path(shard_dir)
            shard_dir_path.mkdir(parents=True, exist_ok=True)
            shard_file = shard_dir_path / f"{segment_id}.json.gz"
            with gzip.open(shard_file, "wt", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False)
            shard_path = str(shard_file)

        elapsed = time.time() - start_time
        result.update({
            "records_scanned": scanned,
            "records_retained": len(records),
            "bytes_downloaded": downloaded,
            "elapsed_seconds": round(elapsed, 1),
            "shard_path": shard_path,
        })

    except Exception as e:
        result["error"] = str(e)[:200]
        result["elapsed_seconds"] = round(time.time() - start_time, 1)

    return result


def main():
    # Target: bring total raw records to ~80k (from 40k)
    # Process new crawl indexes not yet seen
    NEW_CRAWL_INDEXES = [
        "CC-MAIN-2024-10",   # Mar 2024
        "CC-MAIN-2024-18",   # Apr-May 2024
        "CC-MAIN-2023-50",   # Dec 2023
        "CC-MAIN-2023-40",   # Sep-Oct 2023
    ]
    WET_FILES_PER_INDEX = 50   # 50 × 4 indexes = 200 new WET files max
    MAX_RECORDS_SCAN = 30000   # Records to scan per WET file
    TARGET_NEW_RECORDS = 40000 # Stop when we've collected this many new records

    raw_dir = resolve_path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = raw_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = resolve_path("data/manifests")

    manifest = WETManifest(manifest_dir)
    perf = PerfLogger()
    perf.start_stage("targeted_ingestion")

    already_done = manifest.get_completed_count()
    logger.info(f"Manifest: {already_done} files already processed")
    logger.info(f"Targeting {len(NEW_CRAWL_INDEXES)} new crawl indexes")
    logger.info(f"Keywords: {len(TARGETED_KEYWORDS)} high-signal phrases")

    prefilter_kws = [kw.lower() for kw in TARGETED_KEYWORDS]

    all_wet_entries = []
    for idx in NEW_CRAWL_INDEXES:
        try:
            paths = fetch_wet_paths(idx)
            # Evenly sample across the index
            step = max(1, len(paths) // WET_FILES_PER_INDEX)
            selected = [paths[i * step] for i in range(WET_FILES_PER_INDEX)
                       if i * step < len(paths)]
            all_wet_entries.extend([(p, idx) for p in selected])
            logger.info(f"  Selected {len(selected)} paths from {idx}")
        except Exception as e:
            logger.warning(f"  Failed to fetch paths for {idx}: {e}")

    # Filter out already-processed files
    pending = [(p, idx) for p, idx in all_wet_entries
               if not manifest.is_processed(p)]
    skipped = len(all_wet_entries) - len(pending)
    logger.info(f"Pending: {len(pending)}, Already processed: {skipped}")

    if not pending:
        logger.info("All selected files already processed.")
        _merge_and_append(shard_dir, raw_dir)
        perf.end_stage()
        return 0

    workers = max(1, min(os.cpu_count() - 1, 4))
    logger.info(f"Processing with {workers} workers")

    work_args = [
        (path, idx, prefilter_kws, MAX_RECORDS_SCAN, str(shard_dir))
        for path, idx in pending
    ]

    new_records_total = 0
    files_done = 0
    files_failed = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_process_single_wet, args): args[0]
            for args in work_args
        }

        for future in as_completed(futures):
            wet_path = futures[future]
            try:
                result = future.result()
            except Exception as e:
                logger.error(f"Worker crashed for {wet_path}: {e}")
                manifest.mark_failed(wet_path, "", str(e))
                files_failed += 1
                continue

            if result["error"]:
                logger.error(f"  Failed: {result['segment_id']}: {result['error']}")
                manifest.mark_failed(
                    result["wet_path"], result["cc_index"], result["error"]
                )
                files_failed += 1
            else:
                manifest.mark_complete(
                    wet_path=result["wet_path"],
                    crawl_index=result["cc_index"],
                    records_retained=result["records_retained"],
                    records_scanned=result["records_scanned"],
                    bytes_downloaded=result["bytes_downloaded"],
                    elapsed_seconds=result["elapsed_seconds"],
                    shard_path=result["shard_path"],
                )
                new_records_total += result["records_retained"]
                files_done += 1

                logger.info(
                    f"  [{files_done}/{len(pending)}] "
                    f"{result['segment_id']}: "
                    f"scanned {result['records_scanned']:,}, "
                    f"retained {result['records_retained']:,} "
                    f"in {result['elapsed_seconds']:.0f}s "
                    f"(new total: {new_records_total:,})"
                )

            if new_records_total >= TARGET_NEW_RECORDS:
                logger.info(f"Reached target of {TARGET_NEW_RECORDS} new records. Stopping.")
                executor.shutdown(wait=False, cancel_futures=True)
                break

    # Merge new shards into combined file (appending to existing)
    _merge_and_append(shard_dir, raw_dir)

    perf.end_stage()
    perf.record("files_processed", files_done)
    perf.record("files_failed", files_failed)
    perf.record("files_skipped", skipped)
    perf.record("new_records_retained", new_records_total)
    perf.save_report(resolve_path("data/manifests/targeted_ingestion_perf.json"))

    logger.info("=" * 50)
    logger.info("TARGETED INGESTION COMPLETE")
    logger.info(f"  Files processed: {files_done}")
    logger.info(f"  Files failed:    {files_failed}")
    logger.info(f"  New records:     {new_records_total:,}")
    logger.info("=" * 50)

    return new_records_total


def _merge_and_append(shard_dir: Path, raw_dir: Path):
    """
    Load all shard files and merge into ingested_records.json.gz.
    Existing records are preserved; new shards are appended.
    Uses URL deduplication to avoid re-processing same pages.
    """
    existing_path = raw_dir / "ingested_records.json.gz"
    seen_urls = set()
    existing_records = []

    # Load existing
    if existing_path.exists():
        logger.info("Loading existing records for dedup...")
        with gzip.open(existing_path, "rt", encoding="utf-8") as f:
            existing_records = json.load(f)
        seen_urls = {r.get("url", "") for r in existing_records}
        logger.info(f"Existing records: {len(existing_records):,} (unique URLs: {len(seen_urls):,})")

    # Load all shards
    shard_files = sorted(shard_dir.glob("*.json.gz"))
    new_added = 0
    all_records = list(existing_records)

    for sf in shard_files:
        try:
            with gzip.open(sf, "rt", encoding="utf-8") as f:
                shard_recs = json.load(f)
            for r in shard_recs:
                url = r.get("url", "")
                if url not in seen_urls:
                    all_records.append(r)
                    seen_urls.add(url)
                    new_added += 1
        except Exception as e:
            logger.warning(f"Failed to read shard {sf.name}: {e}")

    logger.info(f"Merged: {len(all_records):,} total records ({new_added:,} new)")

    with gzip.open(existing_path, "wt", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False)

    logger.info(f"Saved merged records to {existing_path}")


if __name__ == "__main__":
    count = main()
    sys.exit(0 if count >= 0 else 1)
