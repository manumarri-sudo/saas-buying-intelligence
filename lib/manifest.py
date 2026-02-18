"""
Manifest-based resume and checkpointing for pipeline stages.

Tracks which WET files have been processed, enabling:
  - Resume from interruption without reprocessing
  - Incremental runs with new crawl data
  - Per-file provenance tracking

Manifest format: JSONL file at data/manifests/processed_wet_files.jsonl
Each line: {"wet_path": "...", "crawl_index": "...", "records_retained": N,
            "timestamp": "...", "status": "complete"|"failed", ...}
"""

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class WETManifest:
    """Tracks processed WET files to enable resume and incremental runs."""

    def __init__(self, manifest_dir: Path):
        self.manifest_dir = manifest_dir
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = manifest_dir / "processed_wet_files.jsonl"
        self._processed: dict[str, dict] = {}
        self._load()

    def _load(self):
        """Load existing manifest entries."""
        if not self.manifest_path.exists():
            logger.info("No existing manifest found — starting fresh")
            return

        count = 0
        with open(self.manifest_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    wet_path = entry.get("wet_path", "")
                    if wet_path and entry.get("status") == "complete":
                        self._processed[wet_path] = entry
                        count += 1
                except json.JSONDecodeError:
                    continue

        logger.info(f"Manifest loaded: {count} previously completed WET files")

    def is_processed(self, wet_path: str) -> bool:
        """Check if a WET file has already been successfully processed."""
        return wet_path in self._processed

    def mark_complete(self, wet_path: str, crawl_index: str,
                      records_retained: int, records_scanned: int,
                      bytes_downloaded: int, elapsed_seconds: float,
                      shard_path: str = ""):
        """Record a successfully processed WET file."""
        entry = {
            "wet_path": wet_path,
            "crawl_index": crawl_index,
            "records_retained": records_retained,
            "records_scanned": records_scanned,
            "bytes_downloaded": bytes_downloaded,
            "elapsed_seconds": round(elapsed_seconds, 1),
            "shard_path": shard_path,
            "status": "complete",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._processed[wet_path] = entry
        # Append to JSONL
        with open(self.manifest_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def mark_failed(self, wet_path: str, crawl_index: str, error: str):
        """Record a failed WET file (will be retried on next run)."""
        entry = {
            "wet_path": wet_path,
            "crawl_index": crawl_index,
            "status": "failed",
            "error": str(error)[:200],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        # Append to JSONL but don't add to _processed (so it gets retried)
        with open(self.manifest_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_completed_count(self) -> int:
        return len(self._processed)

    def get_total_records(self) -> int:
        return sum(e.get("records_retained", 0) for e in self._processed.values())

    def get_completed_paths(self) -> set[str]:
        return set(self._processed.keys())

    def get_shard_paths(self) -> list[str]:
        """Return paths to all completed shard files."""
        return [
            e["shard_path"] for e in self._processed.values()
            if e.get("shard_path")
        ]

    def summary(self) -> dict:
        """Return summary statistics."""
        entries = list(self._processed.values())
        return {
            "completed_files": len(entries),
            "total_records": sum(e.get("records_retained", 0) for e in entries),
            "total_scanned": sum(e.get("records_scanned", 0) for e in entries),
            "total_bytes": sum(e.get("bytes_downloaded", 0) for e in entries),
            "total_elapsed": round(sum(e.get("elapsed_seconds", 0) for e in entries), 1),
        }


class StageManifest:
    """Tracks which pipeline stages have been completed for incremental runs."""

    def __init__(self, manifest_dir: Path):
        self.manifest_dir = manifest_dir
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = manifest_dir / "stage_manifest.json"
        self._stages: dict[str, dict] = {}
        self._load()

    def _load(self):
        if self.manifest_path.exists():
            with open(self.manifest_path, "r") as f:
                self._stages = json.load(f)

    def _save(self):
        with open(self.manifest_path, "w") as f:
            json.dump(self._stages, f, indent=2)

    def mark_stage_complete(self, stage: str, metadata: dict = None):
        self._stages[stage] = {
            "status": "complete",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metadata": metadata or {},
        }
        self._save()

    def is_stage_complete(self, stage: str) -> bool:
        return self._stages.get(stage, {}).get("status") == "complete"

    def invalidate_from(self, stage: str):
        """Invalidate this stage and all downstream stages."""
        stage_order = ["ingestion", "filtering", "extraction", "validation", "packaging"]
        try:
            idx = stage_order.index(stage)
        except ValueError:
            return
        for s in stage_order[idx:]:
            if s in self._stages:
                del self._stages[s]
        self._save()
