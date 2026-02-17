#!/usr/bin/env python3
"""
Upload the SaaS Buying Intelligence dataset to HuggingFace Hub.

Usage:
  pip install huggingface_hub datasets
  huggingface-cli login
  python scripts/upload_to_huggingface.py --repo-id YOUR_USERNAME/saas-buying-intelligence
"""

import argparse
import shutil
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Upload dataset to HuggingFace Hub"
    )
    parser.add_argument(
        "--repo-id", required=True,
        help="HuggingFace repo ID (e.g., username/saas-buying-intelligence)"
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Make the dataset private"
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Install huggingface_hub: pip install huggingface_hub")
        sys.exit(1)

    module_dir = Path(__file__).resolve().parent.parent / "output"

    api = HfApi()

    # Create repo
    print(f"Creating dataset repo: {args.repo_id}")
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    # Upload dataset card as README
    card_path = module_dir / "dataset_card.md"
    if card_path.exists():
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
        )
        print("  Uploaded README.md (dataset card)")

    # Upload core files
    files_to_upload = [
        ("dataset.parquet", "data/dataset.parquet"),
        ("schema.json", "schema.json"),
        ("provenance.json", "provenance.json"),
        ("quality_report.json", "quality_report.json"),
        ("embeddings/generate_embeddings.py", "embeddings/generate_embeddings.py"),
        ("rag_recipes/retrieval_config.json", "rag_recipes/retrieval_config.json"),
        ("rag_recipes/retrieval_helper.py", "rag_recipes/retrieval_helper.py"),
        ("prompts/system_prompt.json", "prompts/system_prompt.json"),
        ("prompts/reasoning_prompt.json", "prompts/reasoning_prompt.json"),
        ("prompts/classification_prompt.json", "prompts/classification_prompt.json"),
        ("eval/eval_queries.json", "eval/eval_queries.json"),
        ("eval/run_eval.py", "eval/run_eval.py"),
    ]

    for local_name, remote_name in files_to_upload:
        local_path = module_dir / local_name
        if local_path.exists():
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=remote_name,
                repo_id=args.repo_id,
                repo_type="dataset",
            )
            print(f"  Uploaded {remote_name}")

    print(f"\nDone! View at: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
