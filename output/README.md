# B2B SaaS Buying Intelligence Module

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
