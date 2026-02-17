#!/usr/bin/env python3
"""
Stage 1: Ingestion

Downloads Common Crawl WET files filtered by SaaS keywords.
Also ingests any licensed CSV/JSON datasets from data/licensed/.
Stores raw text with metadata in data/raw/.
"""

import gzip
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config_loader import get_config, resolve_path
from lib.cc_fetcher import (
    fetch_wet_paths,
    download_full_wet_file,
    parse_wet_records,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("01_ingest")


def ingest_common_crawl(cfg: dict) -> list[dict]:
    """
    Fetch WET files from Common Crawl, parse records,
    and pre-filter by keyword presence.
    """
    cc_cfg = cfg["ingestion"]["common_crawl"]
    cc_index = cc_cfg["index"]
    max_wet = cc_cfg["max_wet_files"]
    max_records = cc_cfg["max_records_per_file"]
    prefilter_kws = [kw.lower() for kw in cc_cfg["keyword_prefilter"]]

    raw_dir = resolve_path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Get list of WET file paths
    logger.info(f"Fetching WET paths for index {cc_index}...")
    wet_paths = fetch_wet_paths(cc_index)

    if not wet_paths:
        logger.error("No WET paths found. Check index name or connectivity.")
        return []

    # Limit to configured max
    wet_paths = wet_paths[:max_wet]
    logger.info(f"Will process {len(wet_paths)} WET file(s)")

    all_records = []
    provenance_entries = []

    for i, wet_path in enumerate(wet_paths):
        logger.info(f"Downloading WET file {i+1}/{len(wet_paths)}: {wet_path}")

        local_path = raw_dir / f"wet_{i:04d}.gz"
        success = download_full_wet_file(wet_path, local_path, max_bytes=50_000_000)

        if not success:
            logger.warning(f"Skipping failed download: {wet_path}")
            continue

        # Parse and pre-filter
        logger.info(f"Parsing WET file: {local_path.name}")
        try:
            with open(local_path, "rb") as f:
                raw_data = f.read()

            segment_id = wet_path.split("/")[-1].replace(".warc.wet.gz", "")
            records = list(parse_wet_records(
                raw_data, cc_index, segment_id, max_records,
            ))

            # Pre-filter: text must contain at least one keyword
            filtered = []
            for rec in records:
                text_lower = rec.text.lower()
                if any(kw in text_lower for kw in prefilter_kws):
                    filtered.append(rec.to_dict())

            logger.info(
                f"  Parsed {len(records)} records, "
                f"{len(filtered)} passed keyword pre-filter"
            )
            all_records.extend(filtered)

            provenance_entries.append({
                "source": "common_crawl",
                "wet_path": wet_path,
                "cc_index": cc_index,
                "records_parsed": len(records),
                "records_retained": len(filtered),
                "download_timestamp": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
            })

        except Exception as e:
            logger.error(f"Failed to parse {local_path}: {e}")
            continue

    return all_records, provenance_entries


def ingest_licensed_data(cfg: dict) -> tuple[list[dict], list[dict]]:
    """
    Ingest optional CSV/JSON datasets from data/licensed/.
    """
    lic_cfg = cfg["ingestion"]["licensed_data"]
    input_dir = resolve_path(lic_cfg["input_dir"])
    accepted = set(lic_cfg["accepted_formats"])

    records = []
    provenance = []

    if not input_dir.exists():
        logger.info(f"No licensed data directory at {input_dir}")
        return records, provenance

    import pandas as pd

    for fp in sorted(input_dir.iterdir()):
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

            # Expect at minimum a 'text' column
            if "text" not in df.columns:
                logger.warning(
                    f"  Skipping {fp.name}: no 'text' column found. "
                    f"Columns: {list(df.columns)}"
                )
                continue

            for _, row in df.iterrows():
                records.append({
                    "url": row.get("url", f"licensed:{fp.name}"),
                    "text": str(row["text"]),
                    "crawl_date": str(row.get("date", "")),
                    "segment_id": f"licensed_{fp.stem}",
                    "cc_index": "licensed",
                    "fetch_timestamp": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    ),
                })

            provenance.append({
                "source": "licensed_dataset",
                "filename": fp.name,
                "records_ingested": len(df),
                "ingest_timestamp": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
            })

            logger.info(f"  Ingested {len(df)} records from {fp.name}")

        except Exception as e:
            logger.error(f"Failed to ingest {fp.name}: {e}")
            continue

    return records, provenance


def main():
    cfg = get_config()
    raw_dir = resolve_path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    all_provenance = []

    # Common Crawl ingestion
    logger.info("=== Common Crawl Ingestion ===")
    cc_records, cc_provenance = ingest_common_crawl(cfg)
    all_records.extend(cc_records)
    all_provenance.extend(cc_provenance)

    # Licensed data ingestion
    logger.info("=== Licensed Data Ingestion ===")
    lic_records, lic_provenance = ingest_licensed_data(cfg)
    all_records.extend(lic_records)
    all_provenance.extend(lic_provenance)

    # Save raw records
    output_path = raw_dir / "ingested_records.json.gz"
    logger.info(f"Saving {len(all_records)} records to {output_path}")

    import gzip as gz
    with gz.open(output_path, "wt", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False)

    # Save provenance
    provenance_path = raw_dir / "ingestion_provenance.json"
    with open(provenance_path, "w") as f:
        json.dump({
            "stage": "ingestion",
            "total_records": len(all_records),
            "sources": all_provenance,
            "pipeline_timestamp": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
        }, f, indent=2)

    logger.info(f"Ingestion complete. {len(all_records)} total records.")
    logger.info(f"Provenance saved to {provenance_path}")

    return len(all_records)


if __name__ == "__main__":
    count = main()
    sys.exit(0 if count > 0 else 1)
