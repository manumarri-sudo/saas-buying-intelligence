# B2B SaaS Buying Intelligence Module

**Agent-ready structured dataset for B2B SaaS buying behavior signals.**

> 1,640 high-confidence rows extracted from Common Crawl, covering evaluation criteria, procurement workflows, vendor objections, and industry-specific buying patterns.

---

## Quick Start (5 Steps)

1. **Load the dataset** — `pip install pandas pyarrow` then `pd.read_parquet("dataset.parquet")`
2. **Generate embeddings** — `pip install sentence-transformers` then `python embeddings/generate_embeddings.py`
3. **Query via RAG** — `from rag_recipes.retrieval_helper import SaaSRetriever; r = SaaSRetriever(); r.query("CRM evaluation criteria")`
4. **Use the prompts** — Load `prompts/system_prompt.json` as your agent's system context
5. **Evaluate** — `python eval/run_eval.py` to benchmark retrieval quality

---

## What's Inside

| File | Purpose |
|------|---------|
| `dataset.parquet` | 1,640 structured buying signal rows (256KB) |
| `schema.json` | Full field definitions and legal notes |
| `provenance.json` | Complete data lineage from crawl to output |
| `quality_report.json` | Dedup rates, PII stats, confidence distribution |
| `embeddings/` | Embedding generation script (FAISS + pgvector) |
| `rag_recipes/` | Retrieval config + Python helper class |
| `prompts/` | System, reasoning, and classification prompts |
| `eval/` | 8 test queries with expected answer patterns |

## Dataset Schema

| Field | Type | Description |
|-------|------|-------------|
| `decision_context` | string (max 240 chars) | Buying situation context |
| `criteria` | string | Evaluation criteria: pricing, security, scalability, integration, usability, support, features |
| `objection` | string | Buying blockers: too_expensive, security_concern, poor_integration, complexity, vendor_lock_in, missing_feature |
| `workflow_step` | string | discovery, evaluation, shortlisting, trial, negotiation, procurement, implementation, renewal |
| `industry_hint` | string | healthcare, finance, education, ecommerce, technology, manufacturing, government, media |
| `confidence` | float | 0.70 - 0.92 (all rows >= 0.70) |
| `source_url` | string | Common Crawl source URL |
| `crawl_date` | string | Original crawl timestamp |
| `matched_keywords` | string | Triggering SaaS buying keywords |

## Agent Integration

### Python (direct)
```python
import pandas as pd
df = pd.read_parquet("dataset.parquet")
context = df[df["workflow_step"].str.contains("evaluation")].head(5)
```

### LangChain
```python
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("embeddings/", embeddings)
```

### HuggingFace Datasets
```python
from datasets import load_dataset
ds = load_dataset("parquet", data_files="dataset.parquet")
```

## Data Quality

- **Confidence**: All rows >= 0.70 (mean: 0.75, max: 0.92)
- **PII**: Regex-scanned and removed (emails, phones, addresses, SSNs)
- **Dedup**: Fuzzy deduplicated at 85% similarity threshold
- **Text limit**: All snippets <= 240 characters
- **Source**: Common Crawl WET files (CC-MAIN-2024-46), no direct scraping

## Legal

- Source: Common Crawl public corpus (CC-BY-4.0 terms)
- No long verbatim passages stored (240 char max)
- PII removed via automated detection
- Contains structured insights, not raw web text
- Full provenance chain in `provenance.json`

## License

MIT License for the pipeline code. Dataset derived from Common Crawl under their terms of use.
