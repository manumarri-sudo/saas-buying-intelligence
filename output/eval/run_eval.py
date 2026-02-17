#!/usr/bin/env python3
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
    print(f"\n=== Eval Summary ===")
    print(f"Passed: {passed}/{total} ({100*passed/max(total,1):.0f}%)")

    # Save results
    report_path = eval_dir / "eval_results.json"
    with open(report_path, "w") as f:
        json.dump({"summary": {"total": total, "passed": passed},
                    "results": results}, f, indent=2)
    print(f"Results saved to {report_path}")


if __name__ == "__main__":
    main()
