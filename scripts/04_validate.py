#!/usr/bin/env python3
"""
Stage 4: Validation

Runs automated quality checks on extracted rows:
  - Fuzzy deduplication
  - PII detection and removal
  - Text length enforcement
  - Low-confidence flagging
  - Quality report generation
"""

import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config_loader import get_config, resolve_path
from lib.pii_detector import drop_pii_rows
from lib.deduplicator import deduplicate_dataframe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("04_validate")


def load_extracted_data(extracted_dir: Path) -> pd.DataFrame:
    """Load extracted rows from Stage 3."""
    input_path = extracted_dir / "extracted_rows.parquet"
    if not input_path.exists():
        logger.error(f"Extracted data not found at {input_path}")
        logger.error("Run 03_extract.py first.")
        return pd.DataFrame()

    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df)} extracted rows")
    return df


def enforce_text_length(df: pd.DataFrame, max_length: int) -> tuple[pd.DataFrame, int]:
    """
    Truncate any text fields exceeding max_length.
    Drop rows where the primary field is empty after truncation.
    Returns (df, truncated_count).
    """
    text_cols = ["decision_context", "criteria", "objection",
                 "workflow_step", "industry_hint"]
    truncated = 0

    for col in text_cols:
        if col not in df.columns:
            continue
        mask = df[col].astype(str).str.len() > max_length
        truncated += mask.sum()
        df.loc[mask, col] = df.loc[mask, col].astype(str).str[:max_length - 3] + "..."

    # Drop rows with empty decision_context
    before = len(df)
    df = df[df["decision_context"].astype(str).str.strip().str.len() > 10]
    dropped_empty = before - len(df)

    logger.info(
        f"Text length enforcement: {truncated} fields truncated, "
        f"{dropped_empty} empty rows dropped"
    )
    return df.reset_index(drop=True), truncated


def flag_low_confidence(
    df: pd.DataFrame,
    min_confidence: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into passing and flagged (low-confidence) rows.
    """
    passing = df[df["confidence"] >= min_confidence].copy()
    flagged = df[df["confidence"] < min_confidence].copy()

    logger.info(
        f"Confidence filter: {len(passing)} passing, "
        f"{len(flagged)} flagged (below {min_confidence})"
    )
    return passing.reset_index(drop=True), flagged.reset_index(drop=True)


def generate_quality_report(
    df_original: pd.DataFrame,
    df_final: pd.DataFrame,
    df_flagged: pd.DataFrame,
    duplicate_count: int,
    pii_dropped: int,
    pii_summary: dict,
    truncated_count: int,
) -> dict:
    """Build comprehensive quality report."""
    total_original = len(df_original)
    total_final = len(df_final)

    report = {
        "stage": "validation",
        "row_counts": {
            "input": total_original,
            "output": total_final,
            "dropped_total": total_original - total_final,
            "flagged_low_confidence": len(df_flagged),
        },
        "duplicate_rate": round(
            duplicate_count / max(total_original, 1), 4
        ),
        "noise_rate": round(
            (total_original - total_final) / max(total_original, 1), 4
        ),
        "pii_detection": {
            "rows_dropped": pii_dropped,
            "pii_types_found": pii_summary,
        },
        "text_truncation": {
            "fields_truncated": truncated_count,
        },
        "confidence_distribution": {},
        "workflow_step_distribution": {},
        "criteria_distribution": {},
        "industry_distribution": {},
        "pipeline_timestamp": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
    }

    if not df_final.empty:
        conf = df_final["confidence"]
        report["confidence_distribution"] = {
            "mean": round(conf.mean(), 3),
            "median": round(conf.median(), 3),
            "std": round(conf.std(), 3),
            "min": round(conf.min(), 3),
            "max": round(conf.max(), 3),
            "p25": round(conf.quantile(0.25), 3),
            "p75": round(conf.quantile(0.75), 3),
        }

        # Value distributions
        for col, key in [
            ("workflow_step", "workflow_step_distribution"),
            ("criteria", "criteria_distribution"),
            ("industry_hint", "industry_distribution"),
        ]:
            if col in df_final.columns:
                counts = df_final[col].value_counts().head(20).to_dict()
                report[key] = counts

    return report


def main():
    cfg = get_config()
    val_cfg = cfg["validation"]

    extracted_dir = resolve_path("data/extracted")
    output_dir = resolve_path(cfg["packaging"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_extracted_data(extracted_dir)
    if df.empty:
        logger.error("No data to validate. Exiting.")
        sys.exit(1)

    df_original = df.copy()
    logger.info(f"Starting validation with {len(df)} rows")

    # Step 1: Deduplication
    logger.info("=== Step 1: Deduplication ===")
    dedup_cfg = val_cfg["dedup"]
    df, duplicate_count = deduplicate_dataframe(
        df,
        field=dedup_cfg["field"],
        threshold=dedup_cfg["similarity_threshold"],
    )
    logger.info(f"After dedup: {len(df)} rows ({duplicate_count} removed)")

    # Step 2: PII Detection and Removal
    logger.info("=== Step 2: PII Detection ===")
    text_columns = ["decision_context", "criteria", "objection",
                    "workflow_step", "industry_hint"]
    existing_text_cols = [c for c in text_columns if c in df.columns]

    df, pii_dropped, pii_summary = drop_pii_rows(df, existing_text_cols)
    logger.info(f"After PII removal: {len(df)} rows ({pii_dropped} removed)")

    # Step 3: Text Length Enforcement
    logger.info("=== Step 3: Text Length Enforcement ===")
    df, truncated_count = enforce_text_length(
        df, val_cfg["max_text_length"]
    )
    logger.info(f"After text enforcement: {len(df)} rows")

    # Step 4: Confidence Filtering
    logger.info("=== Step 4: Confidence Filtering ===")
    df_final, df_flagged = flag_low_confidence(
        df, val_cfg["min_confidence"]
    )

    # Save validated data
    validated_path = extracted_dir / "validated_rows.parquet"
    df_final.to_parquet(validated_path, index=False)
    logger.info(f"Saved {len(df_final)} validated rows to {validated_path}")

    # Save flagged rows separately (for review)
    if not df_flagged.empty:
        flagged_path = extracted_dir / "flagged_low_confidence.parquet"
        df_flagged.to_parquet(flagged_path, index=False)
        logger.info(f"Saved {len(df_flagged)} flagged rows to {flagged_path}")

    # Generate quality report
    logger.info("=== Generating Quality Report ===")
    report = generate_quality_report(
        df_original, df_final, df_flagged,
        duplicate_count, pii_dropped, pii_summary, truncated_count,
    )

    report_path = output_dir / cfg["packaging"]["quality_report_filename"]
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Quality report saved to {report_path}")
    logger.info(f"Validation complete: {len(df_final)} clean rows")

    # Print summary
    logger.info("--- Validation Summary ---")
    logger.info(f"  Input rows:     {report['row_counts']['input']}")
    logger.info(f"  Output rows:    {report['row_counts']['output']}")
    logger.info(f"  Duplicate rate: {report['duplicate_rate']:.1%}")
    logger.info(f"  Noise rate:     {report['noise_rate']:.1%}")
    logger.info(f"  PII dropped:    {pii_dropped}")

    return len(df_final)


if __name__ == "__main__":
    count = main()
    sys.exit(0 if count > 0 else 1)
