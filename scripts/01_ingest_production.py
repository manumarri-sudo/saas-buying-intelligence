#!/usr/bin/env python3
"""
Stage 1: Production Ingestion

Downloads multiple Common Crawl WET segments, parses them in-process,
and filters by SaaS buying keywords. Designed for substantial output.

Each WET file is ~150-250MB compressed, containing ~40-60k text records.
With 10 files and keyword filtering, we expect 2000-10000+ relevant passages.
"""

import gzip
import io
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.config_loader import get_config, resolve_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("01_ingest_production")

CC_S3_BASE = "https://data.commoncrawl.org"


def fetch_wet_paths(cc_index: str) -> list[str]:
    """Fetch all WET file paths for this crawl."""
    url = f"{CC_S3_BASE}/crawl-data/{cc_index}/wet.paths.gz"
    logger.info(f"Fetching WET paths from {url}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    paths = gzip.decompress(resp.content).decode("utf-8").strip().split("\n")
    logger.info(f"Found {len(paths):,} WET segments")
    return paths


def download_and_process_wet(
    wet_path: str,
    cc_index: str,
    prefilter_keywords: list[str],
    max_records: int,
    max_bytes: int = 150_000_000,  # 150MB cap per file
) -> tuple[list[dict], dict]:
    """
    Download a WET file, decompress, parse WARC records,
    and pre-filter by keywords. All in streaming fashion.
    """
    url = f"{CC_S3_BASE}/{wet_path}"
    segment_id = wet_path.split("/")[-1].replace(".warc.wet.gz", "")
    records = []
    stats = {
        "wet_path": wet_path,
        "segment_id": segment_id,
        "bytes_downloaded": 0,
        "records_parsed": 0,
        "records_retained": 0,
    }

    try:
        logger.info(f"  Downloading: {segment_id}")
        resp = requests.get(url, stream=True, timeout=180)
        resp.raise_for_status()

        # Download to memory (streamed with size cap)
        chunks = []
        downloaded = 0
        for chunk in resp.iter_content(chunk_size=2_097_152):  # 2MB
            chunks.append(chunk)
            downloaded += len(chunk)
            if downloaded >= max_bytes:
                break
        raw_gz = b"".join(chunks)
        stats["bytes_downloaded"] = downloaded

        logger.info(f"  Downloaded {downloaded / 1_000_000:.1f}MB for {segment_id}")

        # Decompress
        try:
            raw_text = gzip.decompress(raw_gz).decode("utf-8", errors="replace")
        except (gzip.BadGzipFile, OSError) as e:
            logger.warning(f"  Decompression failed for {segment_id}: {e}")
            return records, stats

        # Parse WARC records
        warc_records = raw_text.split("WARC/1.0")
        parsed_count = 0

        for warc_rec in warc_records:
            if parsed_count >= max_records:
                break

            url_found = ""
            crawl_date = ""
            body_lines = []
            header_done = False

            for line in warc_rec.split("\n"):
                if not header_done:
                    if line.startswith("WARC-Target-URI:"):
                        url_found = line.split(":", 1)[1].strip()
                    elif line.startswith("WARC-Date:"):
                        crawl_date = line.split(":", 1)[1].strip()
                    elif line.strip() == "" and (url_found or crawl_date):
                        header_done = True
                else:
                    body_lines.append(line)

            body = "\n".join(body_lines).strip()

            if not url_found or len(body) < 100:
                continue

            parsed_count += 1

            # Keyword pre-filter
            body_lower = body.lower()
            if any(kw in body_lower for kw in prefilter_keywords):
                records.append({
                    "url": url_found,
                    "text": body,
                    "crawl_date": crawl_date,
                    "segment_id": segment_id,
                    "cc_index": cc_index,
                    "fetch_timestamp": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    ),
                })

        stats["records_parsed"] = parsed_count
        stats["records_retained"] = len(records)

        logger.info(
            f"  {segment_id}: parsed {parsed_count:,} records, "
            f"retained {len(records):,} after keyword filter"
        )

    except requests.RequestException as e:
        logger.error(f"  Download failed for {segment_id}: {e}")
    except Exception as e:
        logger.error(f"  Processing failed for {segment_id}: {e}")

    return records, stats


def main():
    cfg = get_config()
    cc_cfg = cfg["ingestion"]["common_crawl"]
    cc_index = cc_cfg["index"]
    max_wet = cc_cfg["max_wet_files"]
    max_records = cc_cfg["max_records_per_file"]
    prefilter_kws = [kw.lower() for kw in cc_cfg["keyword_prefilter"]]

    raw_dir = resolve_path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Get WET paths
    wet_paths = fetch_wet_paths(cc_index)

    # Spread selections across the crawl for diversity
    # Pick every Nth file instead of just the first N
    step = max(1, len(wet_paths) // max_wet)
    selected = [wet_paths[i * step] for i in range(max_wet) if i * step < len(wet_paths)]
    logger.info(f"Selected {len(selected)} WET files (step={step})")

    all_records = []
    all_stats = []

    # Process files with limited parallelism (2 at a time to avoid memory issues)
    logger.info("=== Downloading and processing WET files ===")
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(
                download_and_process_wet,
                path, cc_index, prefilter_kws, max_records,
            ): path
            for path in selected
        }

        for future in as_completed(futures):
            path = futures[future]
            try:
                records, stats = future.result()
                all_records.extend(records)
                all_stats.append(stats)
                logger.info(
                    f"  Running total: {len(all_records):,} records"
                )
            except Exception as e:
                logger.error(f"  Worker failed for {path}: {e}")

    # Also ingest licensed data if present
    licensed_dir = resolve_path(cfg["ingestion"]["licensed_data"]["input_dir"])
    if licensed_dir.exists():
        import pandas as pd
        accepted = set(cfg["ingestion"]["licensed_data"]["accepted_formats"])
        for fp in sorted(licensed_dir.iterdir()):
            if fp.suffix not in accepted:
                continue
            logger.info(f"Ingesting licensed dataset: {fp.name}")
            try:
                if fp.suffix == ".csv":
                    df = pd.read_csv(fp)
                elif fp.suffix in (".json", ".jsonl"):
                    df = pd.read_json(fp, lines=fp.suffix == ".jsonl")
                else:
                    continue
                if "text" not in df.columns:
                    continue
                for _, row in df.iterrows():
                    all_records.append({
                        "url": row.get("url", f"licensed:{fp.name}"),
                        "text": str(row["text"]),
                        "crawl_date": str(row.get("date", "")),
                        "segment_id": f"licensed_{fp.stem}",
                        "cc_index": "licensed",
                        "fetch_timestamp": time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                        ),
                    })
            except Exception as e:
                logger.error(f"Licensed ingest failed: {e}")

    # Save
    logger.info(f"=== Saving {len(all_records):,} total records ===")
    output_path = raw_dir / "ingested_records.json.gz"
    with gzip.open(output_path, "wt", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False)

    # Provenance
    total_bytes = sum(s.get("bytes_downloaded", 0) for s in all_stats)
    total_parsed = sum(s.get("records_parsed", 0) for s in all_stats)
    provenance = {
        "stage": "ingestion",
        "total_records": len(all_records),
        "total_bytes_downloaded": total_bytes,
        "total_records_parsed": total_parsed,
        "sources": all_stats,
        "pipeline_timestamp": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
    }
    prov_path = raw_dir / "ingestion_provenance.json"
    with open(prov_path, "w") as f:
        json.dump(provenance, f, indent=2, default=str)

    logger.info(f"Ingestion complete:")
    logger.info(f"  WET files processed: {len(all_stats)}")
    logger.info(f"  Total bytes downloaded: {total_bytes / 1_000_000:.1f}MB")
    logger.info(f"  Total records parsed: {total_parsed:,}")
    logger.info(f"  Records retained: {len(all_records):,}")
    logger.info(f"  Output: {output_path}")

    return len(all_records)


if __name__ == "__main__":
    count = main()
    sys.exit(0 if count > 0 else 1)
