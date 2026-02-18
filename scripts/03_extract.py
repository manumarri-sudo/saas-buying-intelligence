#!/usr/bin/env python3
"""
Stage 3: Extraction (with strict decision-narrative gate)

Reads scored passages, applies rule-based extraction to produce
structured rows. THEN applies the hard decision-narrative gate:
  - Every row MUST contain a decision verb AND a reasoning marker
  - Navigation noise and bio/job content is stripped first
  - No bypass via confidence scoring

Also produces:
  - debug_kept_vs_dropped.csv (50 kept + 50 dropped examples)
  - decision_narrative_report.json (BEFORE vs AFTER metrics)
"""

import csv
import gzip
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config_loader import get_config, resolve_path
from lib.extractor import extract_row, ExtractedRow
from lib.llm_cache import LLMCache
from lib.narrative_gate import (
    passes_decision_narrative_gate,
    classify_drop_reason,
    DropReason,
)
from lib.perf_logger import PerfLogger
from lib.text_cleaner import full_clean, clean_navigation_text, is_bio_or_job_content, is_promotional_content

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
            has_reasoning=p.get("has_reasoning", True),
            has_narrative=p.get("has_narrative", False),
            is_docs_page=p.get("is_docs_page", False),
        )
        if row is not None:
            rows.append(row.to_dict())
            if row.confidence < 0.4:
                low_conf.append(p)

    return rows, low_conf


def llm_classify_batch(
    passages: list[dict],
    cfg: dict,
    cache: LLMCache,
) -> list[dict]:
    """
    Use LLM to classify low-confidence passages.
    Checks cache first to avoid redundant API calls.
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
    cache_hits = 0

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
        snippet = p["text"][:240]

        # Check cache first
        cached = cache.get(snippet)
        if cached is not None:
            if not cached.get("skip"):
                rows.append(cached)
            cache_hits += 1
            continue

        try:
            response = client.messages.create(
                model=model,
                max_tokens=300,
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": f"Passage: {snippet}",
                }],
            )

            calls_made += 1
            text = response.content[0].text.strip()

            parsed = json.loads(text)

            # Cache the result regardless
            cache.put(snippet, parsed)

            if parsed.get("skip"):
                continue

            required = [
                "decision_context", "criteria", "objection",
                "workflow_step", "industry_hint", "confidence",
            ]
            if all(k in parsed for k in required):
                dc = str(parsed["decision_context"])[:237] + "..." \
                    if len(str(parsed["decision_context"])) > 240 \
                    else str(parsed["decision_context"])

                row_data = {
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
                }
                rows.append(row_data)

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.debug(f"LLM response parse error: {e}")
            cache.put(snippet, {"skip": True, "error": str(e)})
            continue
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            continue

        time.sleep(0.2)

    # Flush cache to disk
    cache.flush()

    logger.info(
        f"LLM classification: {calls_made} API calls, "
        f"{cache_hits} cache hits, {len(rows)} rows extracted"
    )
    return rows


def apply_decision_narrative_gate(
    rows: list[dict],
) -> tuple[list[dict], list[dict], dict]:
    """
    Apply strict decision-narrative gate to extracted rows.

    Returns:
      (kept_rows, dropped_rows, drop_stats)

    The gate is absolute: a row must have BOTH a decision verb
    AND a reasoning marker in its decision_context to survive.
    """
    kept = []
    dropped = []
    drop_counts = Counter()

    for row in rows:
        dc = str(row.get("decision_context", ""))

        # Step 1: Full text cleaning (nav artifacts + structural noise)
        cleaned_dc, clean_drop = full_clean(dc)

        # Step 2: Check if text cleaner flagged it for drop
        if clean_drop is not None:
            row["_drop_reason"] = clean_drop
            dropped.append(row)
            drop_counts[clean_drop] += 1
            continue

        # Step 3: Check for navigation noise (too short after cleaning)
        if len(cleaned_dc.strip()) < 30:
            row["_drop_reason"] = DropReason.NAVIGATION_NOISE.value
            dropped.append(row)
            drop_counts[DropReason.NAVIGATION_NOISE.value] += 1
            continue

        # Step 4: Update decision_context with cleaned version
        row["decision_context"] = cleaned_dc

        # Step 5: Apply the hard decision-narrative gate
        reason = classify_drop_reason(cleaned_dc)

        if reason == DropReason.PASSED:
            kept.append(row)
            drop_counts[DropReason.PASSED.value] += 1
        else:
            row["_drop_reason"] = reason.value
            dropped.append(row)
            drop_counts[reason.value] += 1

    logger.info(f"Decision narrative gate: {len(kept)} kept, {len(dropped)} dropped")
    for reason, count in sorted(drop_counts.items()):
        logger.info(f"  {reason}: {count}")

    return kept, dropped, dict(drop_counts)


def write_debug_csv(
    kept: list[dict],
    dropped: list[dict],
    output_dir: Path,
) -> Path:
    """
    Write debug artifact with 50 kept + 50 dropped examples.
    """
    import random
    random.seed(42)

    kept_sample = random.sample(kept, min(50, len(kept)))
    dropped_sample = random.sample(dropped, min(50, len(dropped)))

    csv_path = output_dir / "debug_kept_vs_dropped.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "status", "source_url", "decision_context", "drop_reason",
        ])

        for row in kept_sample:
            writer.writerow([
                "KEPT",
                row.get("source_url", ""),
                row.get("decision_context", "")[:300],
                "",
            ])

        for row in dropped_sample:
            writer.writerow([
                "DROPPED",
                row.get("source_url", ""),
                row.get("decision_context", "")[:300],
                row.get("_drop_reason", "unknown"),
            ])

    logger.info(f"Debug CSV written to {csv_path} "
                f"({len(kept_sample)} kept + {len(dropped_sample)} dropped)")
    return csv_path


def write_before_after_report(
    before_rows: list[dict],
    after_rows: list[dict],
    drop_stats: dict,
    output_dir: Path,
) -> Path:
    """
    Write BEFORE vs AFTER decision_narrative_report.json.
    """
    from lib.narrative_gate import (
        DECISION_VERB_RE,
        REASONING_MARKER_RE,
    )

    def compute_metrics(rows: list[dict]) -> dict:
        if not rows:
            return {
                "row_count": 0,
                "pct_with_decision_verb": 0,
                "pct_with_reasoning_marker": 0,
                "pct_with_both": 0,
                "median_confidence": 0,
                "top_domains": {},
            }

        has_verb = sum(
            1 for r in rows
            if DECISION_VERB_RE.search(str(r.get("decision_context", "")))
        )
        has_reason = sum(
            1 for r in rows
            if REASONING_MARKER_RE.search(str(r.get("decision_context", "")))
        )
        has_both = sum(
            1 for r in rows
            if DECISION_VERB_RE.search(str(r.get("decision_context", "")))
            and REASONING_MARKER_RE.search(str(r.get("decision_context", "")))
        )

        confidences = [r.get("confidence", 0) for r in rows]
        confidences.sort()
        median_conf = confidences[len(confidences) // 2] if confidences else 0

        # Top domains
        domain_counter = Counter()
        for r in rows:
            url = str(r.get("source_url", ""))
            domain = re.sub(r'^https?://(www\.)?', '', url).split('/')[0]
            if domain:
                domain_counter[domain] += 1

        top_domains = dict(domain_counter.most_common(15))

        return {
            "row_count": len(rows),
            "pct_with_decision_verb": round(100 * has_verb / len(rows), 1),
            "pct_with_reasoning_marker": round(100 * has_reason / len(rows), 1),
            "pct_with_both": round(100 * has_both / len(rows), 1),
            "median_confidence": round(median_conf, 3),
            "top_domains": top_domains,
        }

    before_metrics = compute_metrics(before_rows)
    after_metrics = compute_metrics(after_rows)

    # Sample kept rows
    import random
    random.seed(42)
    kept_examples = []
    sample_kept = random.sample(after_rows, min(5, len(after_rows)))
    for r in sample_kept:
        kept_examples.append({
            "decision_context": str(r.get("decision_context", ""))[:200],
            "source_url": r.get("source_url", ""),
            "confidence": r.get("confidence", 0),
        })

    # Sample dropped rows
    dropped_list = [r for r in before_rows if r not in after_rows]
    dropped_examples = []
    sample_dropped = random.sample(dropped_list, min(5, len(dropped_list)))
    for r in sample_dropped:
        dropped_examples.append({
            "decision_context": str(r.get("decision_context", ""))[:200],
            "source_url": r.get("source_url", ""),
            "confidence": r.get("confidence", 0),
            "drop_reason": r.get("_drop_reason", "failed_gate"),
        })

    report = {
        "report": "decision_narrative_gate_before_vs_after",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "before": before_metrics,
        "after": after_metrics,
        "drop_reasons": drop_stats,
        "examples_kept": kept_examples,
        "examples_dropped": dropped_examples,
        "gate_rules": {
            "decision_verbs": [
                "chose", "choose", "selected", "select", "evaluated", "evaluate",
                "shortlisted", "shortlist", "compared", "compare", "trialed",
                "trial", "piloted", "pilot", "adopted", "adopt", "purchased",
                "purchase", "procured", "procurement", "rejected", "reject",
                "replaced", "replace", "switched", "switch", "migrated", "migrate",
            ],
            "reasoning_markers": [
                "because", "due to", "since", "so that", "therefore",
                "as a result", "needed", "required", "requirement",
                "must have", "deciding factor", "dealbreaker",
                "concern", "concerns", "issue", "issues",
                "challenge", "challenges", "risk", "risks",
                "too expensive", "pricing", "security review",
                "compliance", "integration", "vendor lock-in",
                "onboarding", "implementation", "support",
            ],
            "enforcement": "HARD GATE: both required, no bypass via scoring",
        },
    }

    report_path = output_dir / "decision_narrative_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"BEFORE vs AFTER report written to {report_path}")
    return report_path


def main():
    cfg = get_config()
    extract_cfg = cfg["extraction"]
    max_text_length = cfg["filtering"]["max_snippet_chars"]

    filtered_dir = resolve_path("data/filtered")
    extracted_dir = resolve_path("data/extracted")
    extracted_dir.mkdir(parents=True, exist_ok=True)

    output_dir = resolve_path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    perf = PerfLogger()
    perf.start_stage("extraction")

    # Initialize LLM cache
    cache_dir = resolve_path("data/cache")
    cache = LLMCache(cache_dir)

    passages = load_scored_passages(filtered_dir)
    if not passages:
        logger.error("No passages to extract from. Exiting.")
        sys.exit(1)

    # ── Step 1: Rule-based extraction ─────────────────────────────────
    logger.info("=== Rule-Based Extraction ===")
    rows, low_conf_passages = rule_based_extraction(passages, max_text_length)
    logger.info(
        f"Rule-based: {len(rows)} rows extracted, "
        f"{len(low_conf_passages)} low-confidence passages"
    )

    # ── Step 2: LLM fallback (optional) ──────────────────────────────
    llm_rows = []
    if extract_cfg["llm_fallback"]["enabled"]:
        logger.info("=== LLM Fallback Classification ===")
        llm_rows = llm_classify_batch(low_conf_passages, cfg, cache)
        rows.extend(llm_rows)

    # Save BEFORE state (pre-gate) for comparison
    before_rows = [dict(r) for r in rows]  # deep copy
    logger.info(f"Pre-gate row count: {len(before_rows)}")

    # ── Step 3: STRICT Decision-Narrative Gate ────────────────────────
    logger.info("=== Applying Decision-Narrative Gate (HARD FILTER) ===")
    kept_rows, dropped_rows, drop_stats = apply_decision_narrative_gate(rows)

    # Remove internal _drop_reason from dropped rows for the final data
    # (keep it for debug CSV only)
    for row in kept_rows:
        row.pop("_drop_reason", None)

    # ── Step 4: Write debug artifact ─────────────────────────────────
    logger.info("=== Writing Debug Artifact ===")
    write_debug_csv(kept_rows, dropped_rows, output_dir)

    # ── Step 5: Write BEFORE vs AFTER report ─────────────────────────
    logger.info("=== Writing BEFORE vs AFTER Report ===")
    write_before_after_report(before_rows, kept_rows, drop_stats, output_dir)

    # ── Step 6: Build DataFrame from KEPT rows only ──────────────────
    df = pd.DataFrame(kept_rows)

    if df.empty:
        logger.warning("No rows passed the decision-narrative gate. Creating empty output.")
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
    llm_cache_stats = cache.stats()
    stats = {
        "stage": "extraction",
        "passages_input": len(passages),
        "rows_pre_gate": len(before_rows),
        "rows_post_gate": len(kept_rows),
        "rows_dropped_by_gate": len(dropped_rows),
        "drop_reasons": drop_stats,
        "rows_rule_based": len(before_rows) - len(llm_rows),
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
        "llm_cache": llm_cache_stats,
        "pipeline_timestamp": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
    }

    stats_path = extracted_dir / "extraction_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    perf.end_stage()
    perf.record("passages_input", len(passages))
    perf.record("rows_pre_gate", len(before_rows))
    perf.record("rows_post_gate", len(kept_rows))
    perf.record("rows_dropped_by_gate", len(dropped_rows))
    perf.record("rows_extracted", len(df))
    perf.record("llm_cache_hits", llm_cache_stats["hits"])
    perf.record("llm_cache_misses", llm_cache_stats["misses"])
    perf.save_report(resolve_path("data/manifests/extract_perf.json"))

    logger.info(f"Extraction complete. Stats saved to {stats_path}")

    # Print gate enforcement summary
    logger.info("=" * 60)
    logger.info("DECISION-NARRATIVE GATE ENFORCEMENT SUMMARY")
    logger.info(f"  Pre-gate rows:  {len(before_rows)}")
    logger.info(f"  Post-gate rows: {len(kept_rows)}")
    logger.info(f"  Dropped:        {len(dropped_rows)} ({100*len(dropped_rows)/max(len(before_rows),1):.1f}%)")
    logger.info(f"  Gate is HARD: no row can bypass without both decision verb + reasoning marker")
    logger.info("=" * 60)

    return len(df)


if __name__ == "__main__":
    count = main()
    sys.exit(0 if count > 0 else 1)
