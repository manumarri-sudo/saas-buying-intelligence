---
license: mit
task_categories:
  - text-classification
  - feature-extraction
language:
  - en
tags:
  - saas
  - b2b
  - buying-behavior
  - procurement
  - agent-ready
  - rag
size_categories:
  - 1K<n<10K
---

# B2B SaaS Buying Intelligence Dataset

## Dataset Description

Structured behavioral intelligence for B2B SaaS buying decisions. Each row represents a detected buying signal with decision context, evaluation criteria, objections, workflow stage, and industry classification.

**1,640 high-confidence rows** extracted from Common Crawl (CC-MAIN-2024-46) via automated NLP pipeline with PII removal and fuzzy deduplication.

### Use Cases
- RAG knowledge base for SaaS sales agents
- Training data for buying intent classifiers
- Grounding data for procurement advisory AI
- Market intelligence for SaaS competitive analysis

## Dataset Structure

| Field | Description |
|-------|-------------|
| `decision_context` | Buying situation (max 240 chars) |
| `criteria` | Evaluation criteria (pricing, security, scalability, integration, usability, support, features) |
| `objection` | Buying blockers (too_expensive, security_concern, poor_integration, complexity, vendor_lock_in, missing_feature) |
| `workflow_step` | B2B buying stage (discovery/evaluation/shortlisting/trial/negotiation/procurement/implementation/renewal) |
| `industry_hint` | Industry context (healthcare, finance, education, ecommerce, technology, manufacturing, government, media) |
| `confidence` | Extraction confidence (0.70 - 0.92) |
| `source_url` | Common Crawl source |
| `crawl_date` | Crawl timestamp |

## Quick Start

```python
from datasets import load_dataset
ds = load_dataset("parquet", data_files="dataset.parquet")
print(ds["train"][0])
```

## Data Quality
- All rows >= 0.70 confidence
- PII removed (emails, phones, addresses, SSNs)
- Fuzzy deduplicated at 85% similarity
- All text <= 240 characters

## Source
Common Crawl CC-MAIN-2024-46 (20 WET segments, 292k+ pages scanned). No direct web scraping.
