#!/usr/bin/env python3
"""
Stage 1: Fast Production Ingestion

Uses warcio for streaming WARC/WET parsing (much faster than string splitting).
Downloads WET segments, streams through records, and keyword-filters in one pass.
Each segment is capped at max_records to bound runtime.

Target: 1000-5000+ retained records across 8 WET segments.
Expected runtime: ~15-25 minutes total.
"""

import gzip
import io
import json
import logging
import sys
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

import requests
from warcio.archiveiterator import ArchiveIterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.config_loader import get_config, resolve_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("01_ingest_fast")

CC_S3_BASE = "https://data.commoncrawl.org"


def fetch_wet_paths(cc_index: str) -> list[str]:
    url = f"{CC_S3_BASE}/crawl-data/{cc_index}/wet.paths.gz"
    logger.info(f"Fetching WET paths...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    paths = gzip.decompress(resp.content).decode("utf-8").strip().split("\n")
    logger.info(f"Found {len(paths):,} WET segments available")
    return paths


def stream_wet_file(
    wet_path: str,
    cc_index: str,
    prefilter_keywords: list[str],
    max_records_scan: int = 20000,
) -> tuple[list[dict], dict]:
    """
    Stream a WET file through warcio, filtering on the fly.
    Caps at max_records_scan to bound time.
    """
    url = f"{CC_S3_BASE}/{wet_path}"
    segment_id = wet_path.split("/")[-1].replace(".warc.wet.gz", "")
    records = []
    scanned = 0
    start_time = time.time()

    stats = {
        "wet_path": wet_path,
        "segment_id": segment_id,
        "records_scanned": 0,
        "records_retained": 0,
        "elapsed_seconds": 0,
    }

    try:
        logger.info(f"  Streaming: {segment_id}")
        resp = requests.get(url, stream=True, timeout=300)
        resp.raise_for_status()

        # Stream through warcio — no need to download entire file first
        # We need to save to a temp file because warcio needs seekable input
        with NamedTemporaryFile(suffix=".warc.wet.gz", delete=True) as tmp:
            downloaded = 0
            for chunk in resp.iter_content(chunk_size=2_097_152):
                tmp.write(chunk)
                downloaded += len(chunk)
                # Cap download at 100MB to keep things manageable
                if downloaded >= 100_000_000:
                    break
            tmp.flush()
            tmp.seek(0)

            logger.info(f"  Downloaded {downloaded / 1_000_000:.1f}MB, parsing...")

            for record in ArchiveIterator(tmp):
                if record.rec_type != "conversion":
                    continue

                scanned += 1
                if scanned > max_records_scan:
                    break

                target_uri = record.rec_headers.get_header("WARC-Target-URI") or ""
                warc_date = record.rec_headers.get_header("WARC-Date") or ""

                # Read content
                content = record.content_stream().read()
                try:
                    text = content.decode("utf-8", errors="replace")
                except Exception:
                    continue

                if len(text) < 100:
                    continue

                # Keyword pre-filter
                text_lower = text.lower()
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

        elapsed = time.time() - start_time
        stats["records_scanned"] = scanned
        stats["records_retained"] = len(records)
        stats["elapsed_seconds"] = round(elapsed, 1)
        stats["bytes_downloaded"] = downloaded

        logger.info(
            f"  {segment_id}: scanned {scanned:,}, "
            f"retained {len(records):,} in {elapsed:.0f}s"
        )

    except requests.RequestException as e:
        logger.error(f"  Failed: {segment_id}: {e}")
    except Exception as e:
        logger.error(f"  Error processing {segment_id}: {e}")

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

    wet_paths = fetch_wet_paths(cc_index)

    # Spread selections for diversity
    step = max(1, len(wet_paths) // max_wet)
    selected = [wet_paths[i * step] for i in range(max_wet) if i * step < len(wet_paths)]
    logger.info(f"Selected {len(selected)} WET files (every {step}th)")

    all_records = []
    all_stats = []

    for i, path in enumerate(selected):
        logger.info(f"=== File {i+1}/{len(selected)} ===")
        records, stats = stream_wet_file(
            path, cc_index, prefilter_kws, max_records,
        )
        all_records.extend(records)
        all_stats.append(stats)
        logger.info(f"  Running total: {len(all_records):,} records")

        # If we already have plenty of data, stop early
        if len(all_records) >= 5000:
            logger.info("  Reached 5000 record target, stopping early")
            break

    # Save
    output_path = raw_dir / "ingested_records.json.gz"
    logger.info(f"Saving {len(all_records):,} records to {output_path}")
    with gzip.open(output_path, "wt", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False)

    total_bytes = sum(s.get("bytes_downloaded", 0) for s in all_stats)
    total_scanned = sum(s.get("records_scanned", 0) for s in all_stats)
    total_time = sum(s.get("elapsed_seconds", 0) for s in all_stats)

    provenance = {
        "stage": "ingestion",
        "total_records": len(all_records),
        "total_bytes_downloaded": total_bytes,
        "total_records_scanned": total_scanned,
        "total_elapsed_seconds": round(total_time, 1),
        "sources": all_stats,
        "pipeline_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    prov_path = raw_dir / "ingestion_provenance.json"
    with open(prov_path, "w") as f:
        json.dump(provenance, f, indent=2, default=str)

    logger.info(f"")
    logger.info(f"========== INGESTION COMPLETE ==========")
    logger.info(f"  Files processed:    {len(all_stats)}")
    logger.info(f"  Total downloaded:   {total_bytes / 1_000_000:.0f}MB")
    logger.info(f"  Records scanned:    {total_scanned:,}")
    logger.info(f"  Records retained:   {len(all_records):,}")
    logger.info(f"  Total time:         {total_time:.0f}s")
    logger.info(f"  Output:             {output_path}")

    return len(all_records)


if __name__ == "__main__":
    count = main()
    sys.exit(0 if count > 0 else 1)
