#!/usr/bin/env python3
"""
Stack Exchange ANSWERS ingestion — answers contain first-person buying narratives.
This fetches the actual answers to decision/comparison questions, which say "we chose X because..."
"""
import requests, time, json, gzip, re, sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).parent.parent
RAW_PATH = BASE / "data" / "raw" / "ingested_records.json.gz"
sys.path.insert(0, str(BASE))
from lib.safe_append import safe_load, safe_append

ACTOR_RE = re.compile(r'\b(we |our team|our company|our org|my team|my company|i |i\'ve|we\'ve|we\'re|our stack|we chose|we migrated|we moved|we switched|we replaced|we selected|we went with|we decided|we ended up|we evaluated)\b', re.I)
VERB_RE = re.compile(r'\b(chose|switched|migrated|replaced|moved|adopted|selected|evaluated|decided|implemented|deployed|went with|ended up|settled on)\b', re.I)

def se_get(url, params, retries=3):
    for _ in range(retries):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 429:
                print("  [429 backoff]", flush=True)
                time.sleep(30)
                continue
            if r.status_code == 200:
                d = r.json()
                if d.get("backoff"):
                    time.sleep(d["backoff"] + 1)
                return d
        except Exception as e:
            time.sleep(5)
    return None

def passes_gate(text):
    return bool(VERB_RE.search(text) and ACTOR_RE.search(text))

def fetch_answers_for_site(site, known_urls):
    records = []
    seen = set(known_urls)
    
    # Search for questions that likely have decision narrative answers
    decision_queries = [
        "we switched from to", "we migrated from", "we chose because",
        "we selected over", "we replaced with", "we moved away from",
        "we went with instead", "we decided on", "we evaluated and chose",
        "we ended up using", "why we use", "we adopted because",
        "we implemented instead", "we deployed over", "we cancelled",
        "why we switched", "why we migrated", "we stopped using",
    ]
    
    question_ids_seen = set()
    
    for query in decision_queries:
        # Get questions
        data = se_get("https://api.stackexchange.com/2.3/search/advanced", {
            "site": site, "q": query, "sort": "votes", "order": "desc",
            "pagesize": 100, "page": 1, "filter": "default",
        })
        if not data or not data.get("items"):
            time.sleep(0.5)
            continue
        
        qids = [item["question_id"] for item in data["items"] 
                if item["question_id"] not in question_ids_seen and item.get("answer_count",0) > 0]
        question_ids_seen.update(qids)
        
        # Batch fetch answers for these questions
        for i in range(0, len(qids), 30):
            batch = qids[i:i+30]
            ids_str = ";".join(str(q) for q in batch)
            
            ans_data = se_get(f"https://api.stackexchange.com/2.3/questions/{ids_str}/answers", {
                "site": site, "sort": "votes", "order": "desc",
                "pagesize": 100, "filter": "withbody",
            })
            
            if not ans_data or not ans_data.get("items"):
                time.sleep(0.5)
                continue
            
            for ans in ans_data["items"]:
                # Construct URL
                url = f"https://{site}.stackexchange.com/a/{ans['answer_id']}"
                if site == "serverfault":
                    url = f"https://serverfault.com/a/{ans['answer_id']}"
                elif site == "superuser":
                    url = f"https://superuser.com/a/{ans['answer_id']}"
                
                if url in seen:
                    continue
                
                body = ans.get("body","") or ""
                body_clean = re.sub(r'<[^>]+>', ' ', body)
                body_clean = re.sub(r'\s+', ' ', body_clean).strip()
                
                if len(body_clean) < 100:
                    continue
                if not passes_gate(body_clean):
                    continue
                
                seen.add(url)
                records.append({
                    "url": url, "source_url": url,
                    "full_text": body_clean[:8000],
                    "text": body_clean[:2000],
                    "crawl_date": datetime.utcfromtimestamp(ans.get("creation_date",0)).strftime("%Y-%m-%d"),
                    "segment_id": f"se_answers_{site}",
                    "domain": f"{site}.stackexchange.com",
                    "score": ans.get("score",0),
                })
            
            time.sleep(0.8)
        
        time.sleep(0.3)
    
    print(f"  SE-answers/{site}: {len(records)}", flush=True)
    return records

def main():
    print("Loading existing records (snapshot for dedup)...", flush=True)
    existing = safe_load()
    known_urls = {r.get("url","") for r in existing}
    print(f"Loaded {len(existing):,} records", flush=True)

    SITES = ["softwarerecs", "serverfault", "superuser", "devops", "security", "sre", "webmasters", "sharepoint", "salesforce"]

    all_new = []
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(fetch_answers_for_site, site, set(known_urls)): site for site in SITES}
        for fut in as_completed(futures):
            site = futures[fut]
            try:
                recs = fut.result()
                all_new.extend(recs)
            except Exception as e:
                print(f"  SE-answers/{site} error: {e}", flush=True)

    # Use safe_append — re-reads under lock, deduplicates, atomic write
    added = safe_append(all_new)
    print(f"✓ Done. New SE answer records added: {added}", flush=True)

if __name__ == "__main__":
    main()
