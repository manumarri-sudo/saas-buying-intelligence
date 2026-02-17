#!/usr/bin/env python3
"""
Stage 3: Extraction

Reads scored passages, applies rule-based extraction to produce
structured rows. Optionally uses LLM fallback for low-confidence rows.
Writes structured rows to data/extracted/.
"""

import gzip
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config_loader import get_config, resolve_path
from lib.extractor import extract_row, ExtractedRow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("03_extract")


def load_scored_passages(filtered_dir: Path) -> list[dict]:
    """Load scored passages from filtering stage."""
    input_path = filtered_dir / "scored_passages.json.gz"
    if not input_path.exists():
        logger.error(f"Scored passages not found at {input_path}")
        logger.error("Run 02_filter.py first.")
        return []

    with gzip.open(input_path, "rt", encoding="utf-8") as f:
        passages = json.load(f)

    logger.info(f"Loaded {len(passages)} scored passages")
    return passages


def rule_based_extraction(
    passages: list[dict],
    max_text_length: int,
) -> tuple[list[dict], list[dict]]:
    """
    Apply rule-based extraction to all passages.
    Returns (extracted_rows, low_confidence_passages).
    """
    rows = []
    low_conf = []

    for p in passages:
        row = extract_row(
            text=p["text"],
            source_url=p.get("source_url", ""),
            crawl_date=p.get("crawl_date", ""),
            matched_keywords=p.get("matched_keywords", []),
            max_text_length=max_text_length,
        )
        if row is not None:
            rows.append(row.to_dict())
            if row.confidence < 0.4:
                low_conf.append(p)
        # Passages that yield no row are dropped (no signals found)

    return rows, low_conf


def llm_classify_batch(
    passages: list[dict],
    cfg: dict,
) -> list[dict]:
    """
    Use LLM to classify low-confidence passages.
    Returns additional structured rows.
    """
    llm_cfg = cfg["extraction"]["llm_fallback"]
    api_key = os.environ.get(llm_cfg["api_key_env"], "")

    if not api_key:
        logger.warning(
            f"LLM fallback enabled but {llm_cfg['api_key_env']} not set. "
            "Skipping LLM classification."
        )
        return []

    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic package not installed. Skipping LLM fallback.")
        return []

    client = anthropic.Anthropic(api_key=api_key)
    model = llm_cfg["model"]
    max_calls = llm_cfg["max_calls"]

    rows = []
    calls_made = 0

    system_prompt = (
        "You are a B2B SaaS buying behavior analyst. "
        "Given a text passage about software purchasing, extract:\n"
        "1. decision_context: brief summary of the buying situation (max 200 chars)\n"
        "2. criteria: what factors are being evaluated (comma-separated)\n"
        "3. objection: any concerns or blockers mentioned\n"
        "4. workflow_step: which stage of the buying process "
        "(discovery/evaluation/shortlisting/trial/negotiation/"
        "procurement/implementation/renewal)\n"
        "5. industry_hint: what industry this relates to\n"
        "6. confidence: your confidence 0.0-1.0\n\n"
        "Return ONLY valid JSON with these fields. "
        "If the passage is not about SaaS buying, return {\"skip\": true}"
    )

    for p in passages[:max_calls]:
        try:
            response = client.messages.create(
                model=model,
                max_tokens=300,
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": f"Passage: {p['text'][:240]}",
                }],
            )

            calls_made += 1
            text = response.content[0].text.strip()

            # Parse JSON response
            parsed = json.loads(text)
            if parsed.get("skip"):
                continue

            # Validate required fields
            required = [
                "decision_context", "criteria", "objection",
                "workflow_step", "industry_hint", "confidence",
            ]
            if all(k in parsed for k in required):
                # Enforce text length limit
                dc = str(parsed["decision_context"])[:237] + "..." \
                    if len(str(parsed["decision_context"])) > 240 \
                    else str(parsed["decision_context"])

                rows.append({
                    "decision_context": dc,
                    "criteria": str(parsed["criteria"]),
                    "objection": str(parsed["objection"]),
                    "workflow_step": str(parsed["workflow_step"]),
                    "industry_hint": str(parsed["industry_hint"]),
                    "confidence": float(parsed["confidence"]),
                    "source_url": p.get("source_url", ""),
                    "crawl_date": p.get("crawl_date", ""),
                    "matched_keywords": ", ".join(
                        p.get("matched_keywords", [])
                    ),
                })

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.debug(f"LLM response parse error: {e}")
            continue
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            continue

        time.sleep(0.2)  # rate limiting

    logger.info(
        f"LLM classification: {calls_made} calls, "
        f"{len(rows)} rows extracted"
    )
    return rows


def main():
    cfg = get_config()
    extract_cfg = cfg["extraction"]
    max_text_length = cfg["filtering"]["max_snippet_chars"]

    filtered_dir = resolve_path("data/filtered")
    extracted_dir = resolve_path("data/extracted")
    extracted_dir.mkdir(parents=True, exist_ok=True)

    passages = load_scored_passages(filtered_dir)
    if not passages:
        logger.error("No passages to extract from. Exiting.")
        sys.exit(1)

    # Rule-based extraction
    logger.info("=== Rule-Based Extraction ===")
    rows, low_conf_passages = rule_based_extraction(passages, max_text_length)
    logger.info(
        f"Rule-based: {len(rows)} rows extracted, "
        f"{len(low_conf_passages)} low-confidence passages"
    )

    # LLM fallback (optional)
    llm_rows = []
    if extract_cfg["llm_fallback"]["enabled"]:
        logger.info("=== LLM Fallback Classification ===")
        llm_rows = llm_classify_batch(low_conf_passages, cfg)
        rows.extend(llm_rows)

    # Build DataFrame
    df = pd.DataFrame(rows)

    if df.empty:
        logger.warning("No rows extracted. Creating empty output.")
        df = pd.DataFrame(columns=[
            "decision_context", "criteria", "objection",
            "workflow_step", "industry_hint", "confidence",
            "source_url", "crawl_date", "matched_keywords",
        ])

    # Enforce text length on all text columns
    text_cols = ["decision_context", "criteria", "objection",
                 "workflow_step", "industry_hint"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str[:max_text_length]

    # Save extracted data
    parquet_path = extracted_dir / "extracted_rows.parquet"
    df.to_parquet(parquet_path, index=False)
    logger.info(f"Saved {len(df)} rows to {parquet_path}")

    # Save extraction stats
    stats = {
        "stage": "extraction",
        "passages_input": len(passages),
        "rows_rule_based": len(rows) - len(llm_rows),
        "rows_llm_fallback": len(llm_rows),
        "rows_total": len(df),
        "low_confidence_count": len(
            df[df["confidence"] < extract_cfg["low_confidence_threshold"]]
        ) if not df.empty else 0,
        "high_confidence_count": len(
            df[df["confidence"] >= extract_cfg["high_confidence_threshold"]]
        ) if not df.empty else 0,
        "confidence_distribution": {
            "mean": round(df["confidence"].mean(), 3) if not df.empty else 0,
            "median": round(df["confidence"].median(), 3) if not df.empty else 0,
            "min": round(df["confidence"].min(), 3) if not df.empty else 0,
            "max": round(df["confidence"].max(), 3) if not df.empty else 0,
        },
        "workflow_step_counts": df["workflow_step"].value_counts().to_dict()
            if not df.empty else {},
        "pipeline_timestamp": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
    }

    stats_path = extracted_dir / "extraction_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Extraction complete. Stats saved to {stats_path}")
    return len(df)


if __name__ == "__main__":
    count = main()
    sys.exit(0 if count > 0 else 1)
