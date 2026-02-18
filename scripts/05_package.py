#!/usr/bin/env python3
"""
Stage 5: Packaging

Assembles the final intelligence module:
  - dataset.parquet (validated data)
  - schema.json
  - provenance.json (full chain)
  - quality_report.json (already written by Stage 4)
  - embeddings/ (generation script + optional pre-computed)
  - rag_recipes/
  - prompts/
  - eval/
  - README.md
"""

import json
import logging
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config_loader import get_config, resolve_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("05_package")


def copy_validated_dataset(extracted_dir: Path, output_dir: Path, filename: str) -> pd.DataFrame:
    """Copy validated parquet to output directory."""
    src = extracted_dir / "validated_rows.parquet"
    dst = output_dir / filename

    if not src.exists():
        logger.error(f"Validated data not found at {src}")
        logger.error("Run 04_validate.py first.")
        return pd.DataFrame()

    shutil.copy2(src, dst)
    df = pd.read_parquet(dst)
    logger.info(f"Copied {len(df)} rows to {dst}")
    return df


def generate_schema(df: pd.DataFrame, output_dir: Path, filename: str):
    """Generate schema.json describing the dataset structure."""
    schema = {
        "name": "saas_buying_intelligence",
        "version": "1.0.0",
        "description": (
            "Structured behavioral intelligence for B2B SaaS buying decisions. "
            "Each row represents a detected buying signal with context, criteria, "
            "objections, workflow stage, and industry classification."
        ),
        "row_count": len(df),
        "fields": {
            "decision_context": {
                "type": "string",
                "max_length": 240,
                "description": (
                    "Brief context of the SaaS buying situation. "
                    "Truncated to 240 chars for legal compliance."
                ),
                "nullable": False,
            },
            "criteria": {
                "type": "string",
                "description": (
                    "Comma-separated evaluation criteria detected "
                    "(pricing, security, scalability, integration, usability, "
                    "support, features)."
                ),
                "nullable": True,
            },
            "objection": {
                "type": "string",
                "description": (
                    "Buying objections or blockers detected "
                    "(too_expensive, security_concern, poor_integration, "
                    "complexity, vendor_lock_in, missing_feature)."
                ),
                "nullable": True,
            },
            "workflow_step": {
                "type": "string",
                "description": (
                    "Stage in the B2B buying process "
                    "(discovery, evaluation, shortlisting, trial, "
                    "negotiation, procurement, implementation, renewal)."
                ),
                "nullable": True,
            },
            "industry_hint": {
                "type": "string",
                "description": (
                    "Detected industry context "
                    "(healthcare, finance, education, ecommerce, "
                    "technology, manufacturing, government, media)."
                ),
                "nullable": True,
            },
            "confidence": {
                "type": "float64",
                "range": [0.0, 1.0],
                "description": "Extraction confidence score.",
                "nullable": False,
            },
            "source_url": {
                "type": "string",
                "description": "Original source URL from Common Crawl.",
                "nullable": True,
            },
            "crawl_date": {
                "type": "string",
                "description": "Crawl timestamp from Common Crawl.",
                "nullable": True,
            },
            "matched_keywords": {
                "type": "string",
                "description": "Comma-separated SaaS buying keywords that triggered extraction.",
                "nullable": True,
            },
        },
        "legal_notes": {
            "source": "Common Crawl (CC-BY-4.0 terms) and optional licensed datasets",
            "pii_status": "All rows scanned; rows with detected PII removed",
            "text_limit": "All text fields limited to 240 characters",
            "redistribution": (
                "This dataset contains structured insights, not raw page text. "
                "No long verbatim passages are stored."
            ),
        },
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    path = output_dir / filename
    with open(path, "w") as f:
        json.dump(schema, f, indent=2)
    logger.info(f"Schema written to {path}")


def assemble_provenance(output_dir: Path, filename: str):
    """
    Assemble full provenance chain from all stage outputs.
    """
    raw_dir = resolve_path("data/raw")
    filtered_dir = resolve_path("data/filtered")
    extracted_dir = resolve_path("data/extracted")

    provenance = {
        "module": "saas_buying_intelligence",
        "version": "1.0.0",
        "pipeline_stages": [],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Ingestion provenance
    ing_prov = raw_dir / "ingestion_provenance.json"
    if ing_prov.exists():
        with open(ing_prov) as f:
            provenance["pipeline_stages"].append(json.load(f))

    # Filter stats
    filt_stats = filtered_dir / "filter_stats.json"
    if filt_stats.exists():
        with open(filt_stats) as f:
            provenance["pipeline_stages"].append(json.load(f))

    # Extraction stats
    ext_stats = extracted_dir / "extraction_stats.json"
    if ext_stats.exists():
        with open(ext_stats) as f:
            provenance["pipeline_stages"].append(json.load(f))

    # Quality report (already in output/)
    qr_path = output_dir / "quality_report.json"
    if qr_path.exists():
        with open(qr_path) as f:
            provenance["pipeline_stages"].append(json.load(f))

    path = output_dir / filename
    with open(path, "w") as f:
        json.dump(provenance, f, indent=2)
    logger.info(f"Provenance chain written to {path}")


def write_embedding_script(embeddings_dir: Path):
    """Write the embedding generation script."""
    script = '''#!/usr/bin/env python3
"""
Generate embeddings for the SaaS buying intelligence dataset.

Supports:
  - sentence-transformers → numpy .npy files
  - Optional FAISS index creation
  - Optional pgvector-compatible output
  - Incremental mode: only embed new rows, append to existing index

Usage:
  python generate_embeddings.py [--faiss] [--pgvector] [--incremental]
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _row_hash(row):
    """Hash row content for change detection."""
    text = f"{row.get('decision_context', '')}|{row.get('criteria', '')}|{row.get('workflow_step', '')}"
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings")
    parser.add_argument("--faiss", action="store_true", help="Build FAISS index")
    parser.add_argument("--pgvector", action="store_true", help="Export for pgvector")
    parser.add_argument("--incremental", action="store_true",
                        help="Only embed new rows, append to existing")
    parser.add_argument(
        "--model", default="all-MiniLM-L6-v2",
        help="sentence-transformers model name"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Encoding batch size"
    )
    args = parser.parse_args()

    # Load dataset
    module_dir = Path(__file__).resolve().parent.parent
    dataset_path = module_dir / "dataset.parquet"
    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}")
        sys.exit(1)

    df = pd.read_parquet(dataset_path)
    print(f"Loaded {len(df)} rows")

    output_dir = Path(__file__).resolve().parent
    npy_path = output_dir / "embeddings.npy"
    hashes_path = output_dir / "row_hashes.json"

    # Incremental mode: detect new rows
    new_indices = list(range(len(df)))
    existing_embeddings = None

    if args.incremental and npy_path.exists() and hashes_path.exists():
        existing_embeddings = np.load(npy_path)
        with open(hashes_path) as f:
            old_hashes = set(json.load(f))

        current_hashes = [_row_hash(df.iloc[i]) for i in range(len(df))]
        new_indices = [i for i, h in enumerate(current_hashes) if h not in old_hashes]

        if not new_indices:
            print("No new rows to embed. Embeddings are up to date.")
            return

        print(f"Incremental mode: {len(new_indices)} new rows "
              f"(out of {len(df)} total)")

    # Build texts for new rows
    texts = []
    for i in (new_indices if args.incremental else range(len(df))):
        row = df.iloc[i]
        text = (
            str(row.get("decision_context", "") or "")
            + " | " + str(row.get("criteria", "") or "")
            + " | " + str(row.get("workflow_step", "") or "")
        )
        texts.append(text)

    # Generate embeddings
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Install sentence-transformers: pip install sentence-transformers")
        sys.exit(1)

    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)

    print(f"Encoding {len(texts)} texts...")
    new_embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # In incremental mode, rebuild full embedding array
    if args.incremental and existing_embeddings is not None:
        # Re-encode ALL rows for correct alignment with new dataset
        print("Rebuilding full embeddings for alignment...")
        all_texts = (
            df["decision_context"].fillna("")
            + " | " + df["criteria"].fillna("")
            + " | " + df["workflow_step"].fillna("")
        ).tolist()
        embeddings = model.encode(
            all_texts,
            batch_size=args.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
    else:
        embeddings = new_embeddings

    # Save numpy
    np.save(npy_path, embeddings)
    print(f"Saved embeddings to {npy_path} — shape: {embeddings.shape}")

    # Save row hashes for incremental detection
    all_hashes = [_row_hash(df.iloc[i]) for i in range(len(df))]
    with open(hashes_path, "w") as f:
        json.dump(all_hashes, f)

    # Save row IDs for alignment
    ids_path = output_dir / "row_ids.json"
    with open(ids_path, "w") as f:
        json.dump(list(range(len(df))), f)

    # Optional: FAISS index
    if args.faiss:
        try:
            import faiss

            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)  # inner product (cosine on normalized)
            index.add(embeddings.astype(np.float32))

            faiss_path = output_dir / "faiss_index.bin"
            faiss.write_index(index, str(faiss_path))
            print(f"FAISS index saved to {faiss_path}")
        except ImportError:
            print("faiss-cpu not installed. Skipping FAISS index.")

    # Optional: pgvector SQL
    if args.pgvector:
        sql_path = output_dir / "pgvector_insert.sql"
        with open(sql_path, "w") as f:
            f.write(
                "-- pgvector table for SaaS buying intelligence\\n"
                "CREATE EXTENSION IF NOT EXISTS vector;\\n"
                f"CREATE TABLE saas_embeddings (\\n"
                f"  id SERIAL PRIMARY KEY,\\n"
                f"  decision_context TEXT,\\n"
                f"  criteria TEXT,\\n"
                f"  workflow_step TEXT,\\n"
                f"  embedding vector({embeddings.shape[1]})\\n"
                f");\\n\\n"
            )
            for i, emb in enumerate(embeddings):
                vec_str = "[" + ",".join(f"{v:.6f}" for v in emb) + "]"
                ctx = str(df.iloc[i]["decision_context"]).replace("\'", "\'\'")[:240]
                crit = str(df.iloc[i]["criteria"]).replace("\'", "\'\'")[:240]
                ws = str(df.iloc[i]["workflow_step"]).replace("\'", "\'\'")[:240]
                f.write(
                    f"INSERT INTO saas_embeddings "
                    f"(decision_context, criteria, workflow_step, embedding) "
                    f"VALUES (\'{ctx}\', \'{crit}\', \'{ws}\', \'{vec_str}\');\\n"
                )
        print(f"pgvector SQL saved to {sql_path}")

    print("Embedding generation complete.")


if __name__ == "__main__":
    main()
'''
    path = embeddings_dir / "generate_embeddings.py"
    with open(path, "w") as f:
        f.write(script)
    logger.info(f"Embedding script written to {path}")


def write_rag_config(rag_dir: Path, cfg: dict):
    """Write RAG retrieval configuration."""
    rag_cfg = cfg["rag"]
    emb_cfg = cfg["embeddings"]

    config = {
        "retrieval": {
            "embedding_model": emb_cfg["model"],
            "embedding_dimension": emb_cfg["dimension"],
            "similarity_metric": rag_cfg["similarity_metric"],
            "top_k": rag_cfg["top_k"],
            "similarity_threshold": rag_cfg["similarity_threshold"],
            "chunk_strategy": rag_cfg["chunk_strategy"],
        },
        "chunking": {
            "method": "row_level",
            "description": (
                "Each dataset row is a self-contained chunk. "
                "The decision_context + criteria + workflow_step fields "
                "are concatenated for embedding and retrieval."
            ),
            "max_chunk_chars": 720,
        },
        "reranking": {
            "enabled": False,
            "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "top_k_rerank": 5,
        },
        "index": {
            "type": "faiss_flat_ip",
            "path": "embeddings/faiss_index.bin",
            "fallback": "embeddings/embeddings.npy",
        },
        "metadata_filters": {
            "available_fields": [
                "workflow_step", "criteria", "industry_hint",
                "confidence", "objection",
            ],
            "example_filter": {
                "workflow_step": "evaluation",
                "confidence_gte": 0.5,
            },
        },
    }

    path = rag_dir / "retrieval_config.json"
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"RAG config written to {path}")

    # Also write a Python retrieval helper
    helper = '''#!/usr/bin/env python3
"""
Minimal RAG retrieval helper for the SaaS buying intelligence module.

Usage:
    from rag_recipes.retrieval_helper import SaaSRetriever
    retriever = SaaSRetriever()
    results = retriever.query("What criteria do enterprises use to evaluate CRM?")
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class SaaSRetriever:
    def __init__(self, module_dir: Optional[str] = None):
        if module_dir is None:
            module_dir = Path(__file__).resolve().parent.parent
        else:
            module_dir = Path(module_dir)

        self.df = pd.read_parquet(module_dir / "dataset.parquet")

        # Load config
        with open(module_dir / "rag_recipes" / "retrieval_config.json") as f:
            self.config = json.load(f)

        # Load embeddings
        emb_path = module_dir / "embeddings" / "embeddings.npy"
        if emb_path.exists():
            self.embeddings = np.load(emb_path)
        else:
            self.embeddings = None

        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            model_name = self.config["retrieval"]["embedding_model"]
            self._model = SentenceTransformer(model_name)
        return self._model

    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        min_confidence: float = 0.0,
        workflow_step: Optional[str] = None,
        industry: Optional[str] = None,
    ) -> list[dict]:
        """
        Retrieve relevant SaaS buying intelligence rows.

        Args:
            query_text: natural language query
            top_k: number of results (default from config)
            min_confidence: minimum confidence filter
            workflow_step: filter by workflow step
            industry: filter by industry hint

        Returns:
            List of dicts with row data + similarity score
        """
        if self.embeddings is None:
            raise RuntimeError(
                "Embeddings not found. Run: "
                "python embeddings/generate_embeddings.py"
            )

        if top_k is None:
            top_k = self.config["retrieval"]["top_k"]

        threshold = self.config["retrieval"]["similarity_threshold"]

        # Encode query
        model = self._get_model()
        query_emb = model.encode(
            [query_text], normalize_embeddings=True
        )[0]

        # Compute similarities
        similarities = self.embeddings @ query_emb

        # Apply metadata filters
        mask = np.ones(len(self.df), dtype=bool)
        if min_confidence > 0:
            mask &= self.df["confidence"].values >= min_confidence
        if workflow_step:
            mask &= self.df["workflow_step"].str.contains(
                workflow_step, case=False, na=False
            ).values
        if industry:
            mask &= self.df["industry_hint"].str.contains(
                industry, case=False, na=False
            ).values

        # Zero out filtered rows
        filtered_sims = similarities.copy()
        filtered_sims[~mask] = -1.0

        # Get top-k
        top_indices = np.argsort(filtered_sims)[::-1][:top_k]

        results = []
        for idx in top_indices:
            sim = float(filtered_sims[idx])
            if sim < threshold:
                break
            row = self.df.iloc[idx].to_dict()
            row["similarity_score"] = round(sim, 4)
            results.append(row)

        return results
'''
    helper_path = rag_dir / "retrieval_helper.py"
    with open(helper_path, "w") as f:
        f.write(helper)
    logger.info(f"Retrieval helper written to {helper_path}")


def write_prompt_pack(prompts_dir: Path):
    """Write the prompt pack for agent integration."""

    # System prompt
    system_prompt = {
        "name": "saas_buying_intelligence_system",
        "version": "1.0.0",
        "prompt": (
            "You are an AI assistant specialized in B2B SaaS buying decisions. "
            "You have access to a structured intelligence module containing "
            "real-world buying signals, evaluation criteria, common objections, "
            "and workflow patterns across industries.\n\n"
            "When answering questions:\n"
            "1. Ground your answers in retrieved intelligence data\n"
            "2. Cite the workflow stage and criteria when relevant\n"
            "3. Distinguish between different industry contexts\n"
            "4. Note confidence levels of the underlying data\n"
            "5. If data is insufficient, say so clearly\n\n"
            "Your knowledge covers: discovery, evaluation, shortlisting, "
            "trial, negotiation, procurement, implementation, and renewal "
            "stages of SaaS buying."
        ),
    }

    # Reasoning prompt
    reasoning_prompt = {
        "name": "saas_buying_reasoning",
        "version": "1.0.0",
        "prompt": (
            "Given the following retrieved SaaS buying intelligence:\n\n"
            "{retrieved_context}\n\n"
            "User question: {question}\n\n"
            "Reason through the following:\n"
            "1. What buying stage(s) does this question relate to?\n"
            "2. What evaluation criteria are most relevant?\n"
            "3. What common objections might arise?\n"
            "4. What patterns across industries apply?\n"
            "5. What is your confidence in this analysis?\n\n"
            "Provide a structured answer with clear reasoning."
        ),
        "variables": ["retrieved_context", "question"],
    }

    # Classification prompt
    classification_prompt = {
        "name": "saas_signal_classifier",
        "version": "1.0.0",
        "prompt": (
            "Classify the following text passage into B2B SaaS buying signals.\n\n"
            "Text: {text}\n\n"
            "Extract:\n"
            "- decision_context: Brief summary (max 200 chars)\n"
            "- criteria: Evaluation factors (comma-separated from: "
            "pricing, security, scalability, integration, usability, "
            "support, features)\n"
            "- objection: Buying blockers (comma-separated from: "
            "too_expensive, security_concern, poor_integration, "
            "complexity, vendor_lock_in, missing_feature)\n"
            "- workflow_step: Buying stage (one of: discovery, evaluation, "
            "shortlisting, trial, negotiation, procurement, implementation, "
            "renewal)\n"
            "- industry_hint: Industry context\n"
            "- confidence: 0.0-1.0\n\n"
            "Return valid JSON only."
        ),
        "variables": ["text"],
    }

    for name, data in [
        ("system_prompt.json", system_prompt),
        ("reasoning_prompt.json", reasoning_prompt),
        ("classification_prompt.json", classification_prompt),
    ]:
        path = prompts_dir / name
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    logger.info(f"Prompt pack written to {prompts_dir}")


def write_eval_set(eval_dir: Path):
    """Write evaluation set with representative queries and expected patterns."""

    eval_set = {
        "name": "saas_buying_intelligence_eval",
        "version": "1.0.0",
        "description": (
            "Evaluation queries for testing RAG retrieval and "
            "agent reasoning quality."
        ),
        "queries": [
            {
                "id": "eval_001",
                "query": "What criteria do enterprises use when evaluating CRM software?",
                "expected_workflow_steps": ["evaluation", "shortlisting"],
                "expected_criteria": ["features", "integration", "pricing", "scalability"],
                "expected_min_results": 3,
                "expected_confidence_gte": 0.5,
            },
            {
                "id": "eval_002",
                "query": "What are common objections during SaaS procurement?",
                "expected_workflow_steps": ["procurement", "negotiation"],
                "expected_criteria": ["pricing", "security"],
                "expected_min_results": 2,
                "expected_confidence_gte": 0.4,
            },
            {
                "id": "eval_003",
                "query": "How do healthcare organizations handle security reviews for cloud vendors?",
                "expected_workflow_steps": ["evaluation"],
                "expected_criteria": ["security"],
                "expected_industries": ["healthcare"],
                "expected_min_results": 1,
                "expected_confidence_gte": 0.4,
            },
            {
                "id": "eval_004",
                "query": "What does a typical SaaS trial or proof of concept look like?",
                "expected_workflow_steps": ["trial"],
                "expected_criteria": ["features", "usability"],
                "expected_min_results": 2,
                "expected_confidence_gte": 0.5,
            },
            {
                "id": "eval_005",
                "query": "What factors lead companies to switch from one SaaS vendor to another?",
                "expected_workflow_steps": ["renewal"],
                "expected_criteria": ["pricing", "features", "support"],
                "expected_objections": ["too_expensive", "missing_feature"],
                "expected_min_results": 2,
                "expected_confidence_gte": 0.4,
            },
            {
                "id": "eval_006",
                "query": "What integration requirements are important in vendor evaluation?",
                "expected_workflow_steps": ["evaluation", "shortlisting"],
                "expected_criteria": ["integration"],
                "expected_min_results": 2,
                "expected_confidence_gte": 0.5,
            },
            {
                "id": "eval_007",
                "query": "How do fintech companies approach SaaS compliance evaluation?",
                "expected_workflow_steps": ["evaluation"],
                "expected_criteria": ["security"],
                "expected_industries": ["finance"],
                "expected_min_results": 1,
                "expected_confidence_gte": 0.4,
            },
            {
                "id": "eval_008",
                "query": "What does the enterprise software procurement RFP process involve?",
                "expected_workflow_steps": ["procurement"],
                "expected_min_results": 2,
                "expected_confidence_gte": 0.5,
            },
        ],
    }

    path = eval_dir / "eval_queries.json"
    with open(path, "w") as f:
        json.dump(eval_set, f, indent=2)

    # Write eval runner script
    runner = '''#!/usr/bin/env python3
"""
Evaluation runner for the SaaS buying intelligence module.

Runs eval queries against the retriever and reports hit rates.

Usage:
    python run_eval.py [--module-dir ../]
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module-dir", default=None)
    args = parser.parse_args()

    eval_dir = Path(__file__).resolve().parent
    module_dir = Path(args.module_dir) if args.module_dir else eval_dir.parent

    # Load eval set
    with open(eval_dir / "eval_queries.json") as f:
        eval_set = json.load(f)

    # Import retriever
    sys.path.insert(0, str(module_dir / "rag_recipes"))
    try:
        from retrieval_helper import SaaSRetriever
        retriever = SaaSRetriever(str(module_dir))
    except Exception as e:
        print(f"Failed to initialize retriever: {e}")
        print("Ensure embeddings are generated first.")
        sys.exit(1)

    results = []
    for q in eval_set["queries"]:
        query_id = q["id"]
        query_text = q["query"]
        expected_min = q.get("expected_min_results", 1)
        expected_conf = q.get("expected_confidence_gte", 0.0)

        retrieved = retriever.query(
            query_text,
            top_k=10,
            min_confidence=expected_conf,
        )

        # Check expectations
        passed = len(retrieved) >= expected_min

        # Check workflow steps if expected
        if "expected_workflow_steps" in q and retrieved:
            ws_found = any(
                any(ews in str(r.get("workflow_step", ""))
                    for ews in q["expected_workflow_steps"])
                for r in retrieved
            )
        else:
            ws_found = True

        results.append({
            "query_id": query_id,
            "query": query_text,
            "results_count": len(retrieved),
            "min_expected": expected_min,
            "count_pass": passed,
            "workflow_pass": ws_found,
            "overall_pass": passed and ws_found,
        })

        status = "PASS" if (passed and ws_found) else "FAIL"
        print(f"  [{status}] {query_id}: {len(retrieved)} results "
              f"(need >={expected_min})")

    # Summary
    total = len(results)
    passed = sum(1 for r in results if r["overall_pass"])
    print(f"\\n=== Eval Summary ===")
    print(f"Passed: {passed}/{total} ({100*passed/max(total,1):.0f}%)")

    # Save results
    report_path = eval_dir / "eval_results.json"
    with open(report_path, "w") as f:
        json.dump({"summary": {"total": total, "passed": passed},
                    "results": results}, f, indent=2)
    print(f"Results saved to {report_path}")


if __name__ == "__main__":
    main()
'''
    runner_path = eval_dir / "run_eval.py"
    with open(runner_path, "w") as f:
        f.write(runner)

    logger.info(f"Eval set and runner written to {eval_dir}")


def write_readme(output_dir: Path):
    """Write README for the intelligence module."""
    readme = """# B2B SaaS Buying Intelligence Module

## Overview

Agent-ready intelligence module containing structured behavioral data on B2B SaaS buying decisions. Built from public Common Crawl data with automated signal extraction, PII removal, and quality validation.

## Quick Start

### 1. Use in RAG Pipeline

```python
from rag_recipes.retrieval_helper import SaaSRetriever

retriever = SaaSRetriever("/path/to/this/module")
results = retriever.query("What criteria matter for enterprise CRM evaluation?")

for r in results:
    print(f"[{r['confidence']:.2f}] {r['decision_context']}")
    print(f"  Criteria: {r['criteria']}")
    print(f"  Stage: {r['workflow_step']}")
```

### 2. Generate Embeddings

```bash
pip install sentence-transformers
python embeddings/generate_embeddings.py          # numpy output
python embeddings/generate_embeddings.py --faiss   # + FAISS index
python embeddings/generate_embeddings.py --pgvector # + pgvector SQL
```

### 3. Run Evaluation

```bash
python eval/run_eval.py
```

### 4. Use Prompts in Agent System

Load prompts from `prompts/`:
- `system_prompt.json` — agent system context
- `reasoning_prompt.json` — structured reasoning template
- `classification_prompt.json` — classify new text into buying signals

## Dataset Schema

| Field | Type | Description |
|-------|------|-------------|
| decision_context | string (max 240 chars) | Buying situation context |
| criteria | string | Evaluation criteria (comma-separated) |
| objection | string | Buying blockers detected |
| workflow_step | string | B2B buying stage |
| industry_hint | string | Industry context |
| confidence | float [0,1] | Extraction confidence |
| source_url | string | Common Crawl source |
| crawl_date | string | Original crawl timestamp |
| matched_keywords | string | Triggering SaaS keywords |

## Files

```
dataset.parquet        — Clean structured dataset
schema.json            — Dataset schema definition
provenance.json        — Full data lineage chain
quality_report.json    — Validation statistics
embeddings/            — Embedding generation scripts
rag_recipes/           — RAG retrieval config and helper
prompts/               — Agent prompt templates
eval/                  — Evaluation queries and runner
```

## Legal

- Source data: Common Crawl (public, CC-BY-4.0 terms)
- All text snippets limited to 240 characters
- PII detection applied; rows with PII removed
- Contains structured insights, not raw page text
- See `provenance.json` for full data lineage

## Integration Examples

### LangChain

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("embeddings/", embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
```

### LlamaIndex

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
# Load dataset.parquet as documents, use with any VectorStore
```

### Direct NumPy

```python
import numpy as np
embeddings = np.load("embeddings/embeddings.npy")
# Use with any similarity search library
```
"""
    path = output_dir / "README.md"
    with open(path, "w") as f:
        f.write(readme)
    logger.info(f"README written to {path}")


def main():
    cfg = get_config()
    pkg_cfg = cfg["packaging"]

    output_dir = resolve_path(pkg_cfg["output_dir"])
    extracted_dir = resolve_path("data/extracted")

    # Ensure output subdirectories exist
    for subdir in ["embeddings", "rag_recipes", "prompts", "eval"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Step 1: Copy validated dataset
    logger.info("=== Copying validated dataset ===")
    df = copy_validated_dataset(
        extracted_dir, output_dir, pkg_cfg["dataset_filename"]
    )

    # Step 2: Generate schema
    logger.info("=== Generating schema ===")
    generate_schema(df, output_dir, pkg_cfg["schema_filename"])

    # Step 3: Assemble provenance
    logger.info("=== Assembling provenance ===")
    assemble_provenance(output_dir, pkg_cfg["provenance_filename"])

    # Step 4: Write embedding script
    logger.info("=== Writing embedding generation script ===")
    write_embedding_script(output_dir / "embeddings")

    # Step 5: Write RAG config
    logger.info("=== Writing RAG configuration ===")
    write_rag_config(output_dir / "rag_recipes", cfg)

    # Step 6: Write prompt pack
    logger.info("=== Writing prompt pack ===")
    write_prompt_pack(output_dir / "prompts")

    # Step 7: Write eval set
    logger.info("=== Writing evaluation set ===")
    write_eval_set(output_dir / "eval")

    # Step 8: Write README
    logger.info("=== Writing README ===")
    write_readme(output_dir)

    logger.info("=" * 60)
    logger.info("PACKAGING COMPLETE")
    logger.info(f"Module directory: {output_dir}")
    logger.info("Contents:")
    for item in sorted(output_dir.rglob("*")):
        if item.is_file():
            rel = item.relative_to(output_dir)
            size = item.stat().st_size
            logger.info(f"  {rel} ({size:,} bytes)")


if __name__ == "__main__":
    main()
