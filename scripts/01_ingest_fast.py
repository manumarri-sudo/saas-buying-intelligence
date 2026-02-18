#!/usr/bin/env python3
"""
Stage 1: Fast Production Ingestion (with resume + parallelism)

Features:
  - Manifest-based checkpointing: skips already-processed WET files
  - Parallel download+parse with concurrent.futures
  - Per-file shard output: each WET file → data/raw/shards/<segment_id>.json.gz
  - Early stopping when target record count is reached
  - Performance metrics logging

Uses warcio for streaming WARC/WET parsing.
Target: 1000-5000+ retained records across WET segments.
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
from lib.config_loader import get_config, resolve_path
from lib.manifest import WETManifest
from lib.perf_logger import PerfLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("01_ingest_fast")

CC_S3_BASE = "https://data.commoncrawl.org"


def fetch_wet_paths(cc_index: str) -> list[str]:
    url = f"{CC_S3_BASE}/crawl-data/{cc_index}/wet.paths.gz"
    logger.info(f"Fetching WET paths for {cc_index}...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    paths = gzip.decompress(resp.content).decode("utf-8").strip().split("\n")
    logger.info(f"Found {len(paths):,} WET segments available")
    return paths


def _process_single_wet(args: tuple) -> dict:
    """
    Process a single WET file. Runs in a subprocess.
    Returns a result dict with records and stats.
    """
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
                if downloaded >= 100_000_000:
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

                if len(text) < 100:
                    continue

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
    cfg = get_config()
    cc_cfg = cfg["ingestion"]["common_crawl"]
    cc_index = cc_cfg["index"]
    max_wet = cc_cfg["max_wet_files"]
    max_records = cc_cfg["max_records_per_file"]
    prefilter_kws = [kw.lower() for kw in cc_cfg["keyword_prefilter"]]
    target_records = 40000

    # Use multiple crawl indexes for diverse data
    crawl_indexes = [cc_index, "CC-MAIN-2024-51"]

    raw_dir = resolve_path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = raw_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = resolve_path("data/manifests")

    # Initialize manifest for resume support
    manifest = WETManifest(manifest_dir)
    perf = PerfLogger()
    perf.start_stage("ingestion")

    logger.info(f"Manifest: {manifest.get_completed_count()} files already processed, "
                f"{manifest.get_total_records()} records from previous runs")

    # Gather WET paths from all crawl indexes
    all_wet_entries = []
    for idx in crawl_indexes:
        try:
            paths = fetch_wet_paths(idx)
            wet_per_index = max_wet // len(crawl_indexes)
            step = max(1, len(paths) // wet_per_index)
            selected_paths = [
                paths[i * step] for i in range(wet_per_index)
                if i * step < len(paths)
            ]
            all_wet_entries.extend([(p, idx) for p in selected_paths])
            logger.info(f"  Selected {len(selected_paths)} from {idx}")
        except Exception as e:
            logger.warning(f"  Failed to fetch paths for {idx}: {e}")

    # Filter out already-processed files
    pending = [
        (path, idx) for path, idx in all_wet_entries
        if not manifest.is_processed(path)
    ]
    skipped = len(all_wet_entries) - len(pending)

    logger.info(f"Total selected: {len(all_wet_entries)} WET files")
    logger.info(f"  Already processed (skipping): {skipped}")
    logger.info(f"  Pending: {len(pending)}")

    if not pending:
        logger.info("All WET files already processed. Nothing to do.")
        # Still merge shards into combined output
        _merge_shards(shard_dir, raw_dir, manifest)
        perf.end_stage()
        perf.record("files_skipped", skipped)
        perf.record("files_processed", 0)
        perf.save_report(resolve_path("data/manifests/performance_report.json"))
        return manifest.get_total_records()

    # Determine concurrency
    workers = max(1, min(os.cpu_count() - 1, 4))  # cap at 4 to be polite to CC
    logger.info(f"Processing with {workers} parallel workers")

    # Build args for parallel execution
    work_args = [
        (path, idx, prefilter_kws, max_records, str(shard_dir))
        for path, idx in pending
    ]

    total_retained = manifest.get_total_records()
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
                total_retained += result["records_retained"]
                files_done += 1

                logger.info(
                    f"  [{manifest.get_completed_count()}/{len(all_wet_entries)}] "
                    f"{result['segment_id']}: scanned {result['records_scanned']:,}, "
                    f"retained {result['records_retained']:,} "
                    f"in {result['elapsed_seconds']:.0f}s "
                    f"(total: {total_retained:,})"
                )

            # Early stopping if we have enough data
            if total_retained >= target_records:
                logger.info(f"Reached {target_records} record target, stopping early")
                executor.shutdown(wait=False, cancel_futures=True)
                break

    # Merge shards into combined output
    _merge_shards(shard_dir, raw_dir, manifest)

    # Performance logging
    perf.end_stage()
    summary = manifest.summary()
    perf.record("files_processed", files_done)
    perf.record("files_failed", files_failed)
    perf.record("files_skipped", skipped)
    perf.record("records_retained", summary["total_records"])
    perf.record("records_scanned", summary["total_scanned"])
    perf.record("bytes_downloaded", summary["total_bytes"])

    perf.save_report(resolve_path("data/manifests/performance_report.json"))

    # Save provenance
    provenance = {
        "stage": "ingestion",
        "total_records": summary["total_records"],
        "total_bytes_downloaded": summary["total_bytes"],
        "total_records_scanned": summary["total_scanned"],
        "total_elapsed_seconds": summary["total_elapsed"],
        "files_processed_this_run": files_done,
        "files_skipped_cached": skipped,
        "files_failed": files_failed,
        "pipeline_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    prov_path = raw_dir / "ingestion_provenance.json"
    with open(prov_path, "w") as f:
        json.dump(provenance, f, indent=2, default=str)

    logger.info("")
    logger.info("========== INGESTION COMPLETE ==========")
    logger.info(f"  Files this run:     {files_done}")
    logger.info(f"  Files skipped:      {skipped}")
    logger.info(f"  Files failed:       {files_failed}")
    logger.info(f"  Total downloaded:   {summary['total_bytes'] / 1_000_000:.0f}MB")
    logger.info(f"  Records scanned:    {summary['total_scanned']:,}")
    logger.info(f"  Records retained:   {summary['total_records']:,}")

    return summary["total_records"]


def _merge_shards(shard_dir: Path, raw_dir: Path, manifest: WETManifest):
    """Merge all completed shard files into a single ingested_records.json.gz."""
    shard_files = sorted(shard_dir.glob("*.json.gz"))
    if not shard_files:
        logger.warning("No shard files found to merge")
        return

    all_records = []
    for sf in shard_files:
        try:
            with gzip.open(sf, "rt", encoding="utf-8") as f:
                records = json.load(f)
                all_records.extend(records)
        except Exception as e:
            logger.warning(f"Failed to read shard {sf.name}: {e}")

    output_path = raw_dir / "ingested_records.json.gz"
    logger.info(f"Merging {len(shard_files)} shards → {len(all_records):,} records")
    with gzip.open(output_path, "wt", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False)

    logger.info(f"Merged output: {output_path}")


if __name__ == "__main__":
    count = main()
    sys.exit(0 if count > 0 else 1)
