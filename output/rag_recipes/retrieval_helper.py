#!/usr/bin/env python3
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
