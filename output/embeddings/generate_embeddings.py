#!/usr/bin/env python3
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
                "-- pgvector table for SaaS buying intelligence\n"
                "CREATE EXTENSION IF NOT EXISTS vector;\n"
                f"CREATE TABLE saas_embeddings (\n"
                f"  id SERIAL PRIMARY KEY,\n"
                f"  decision_context TEXT,\n"
                f"  criteria TEXT,\n"
                f"  workflow_step TEXT,\n"
                f"  embedding vector({embeddings.shape[1]})\n"
                f");\n\n"
            )
            for i, emb in enumerate(embeddings):
                vec_str = "[" + ",".join(f"{v:.6f}" for v in emb) + "]"
                ctx = str(df.iloc[i]["decision_context"]).replace("'", "''")[:240]
                crit = str(df.iloc[i]["criteria"]).replace("'", "''")[:240]
                ws = str(df.iloc[i]["workflow_step"]).replace("'", "''")[:240]
                f.write(
                    f"INSERT INTO saas_embeddings "
                    f"(decision_context, criteria, workflow_step, embedding) "
                    f"VALUES ('{ctx}', '{crit}', '{ws}', '{vec_str}');\n"
                )
        print(f"pgvector SQL saved to {sql_path}")

    print("Embedding generation complete.")


if __name__ == "__main__":
    main()
