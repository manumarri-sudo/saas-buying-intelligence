"""
Performance logging and reporting for pipeline stages.

Tracks throughput metrics, timing, cache hit rates, and generates
a performance_report.json at the end of each run.
"""

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class PerfLogger:
    """Collects performance metrics across pipeline stages."""

    def __init__(self):
        self._stage_start: float = 0
        self._stages: dict[str, dict] = {}
        self._current_stage: str = ""
        self._pipeline_start = time.time()

    def start_stage(self, stage: str):
        self._current_stage = stage
        self._stage_start = time.time()
        self._stages[stage] = {
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metrics": {},
        }

    def end_stage(self, stage: str = None):
        stage = stage or self._current_stage
        if stage in self._stages:
            elapsed = time.time() - self._stage_start
            self._stages[stage]["elapsed_seconds"] = round(elapsed, 2)
            self._stages[stage]["end_time"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            )

    def record(self, key: str, value, stage: str = None):
        """Record a metric for the current or specified stage."""
        stage = stage or self._current_stage
        if stage in self._stages:
            self._stages[stage]["metrics"][key] = value

    def increment(self, key: str, amount: int = 1, stage: str = None):
        """Increment a counter metric."""
        stage = stage or self._current_stage
        if stage in self._stages:
            metrics = self._stages[stage]["metrics"]
            metrics[key] = metrics.get(key, 0) + amount

    def save_report(self, output_path: Path):
        """Save the full performance report."""
        total_elapsed = time.time() - self._pipeline_start

        report = {
            "pipeline_total_seconds": round(total_elapsed, 2),
            "pipeline_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stages": self._stages,
        }

        # Compute throughput metrics where applicable
        for stage_name, stage_data in self._stages.items():
            metrics = stage_data.get("metrics", {})
            elapsed = stage_data.get("elapsed_seconds", 0)
            if elapsed > 0:
                if "records_processed" in metrics:
                    metrics["records_per_second"] = round(
                        metrics["records_processed"] / elapsed, 1
                    )
                if "bytes_downloaded" in metrics:
                    metrics["mb_per_second"] = round(
                        metrics["bytes_downloaded"] / (1_000_000 * elapsed), 2
                    )
                if "files_processed" in metrics:
                    metrics["seconds_per_file"] = round(
                        elapsed / metrics["files_processed"], 2
                    )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Performance report saved to {output_path}")
        logger.info(f"Total pipeline time: {total_elapsed:.1f}s")

        return report
