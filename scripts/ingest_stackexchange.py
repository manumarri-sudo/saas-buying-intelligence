#!/usr/bin/env python3
"""
Stack Exchange API ingestion — targets software buying/switching decision posts.
Uses the public API (no key needed, 300 req/day; with key 10K/day).
Focuses on: softwarerecs.stackexchange.com, superuser, serverfault, devops.stackexchange
"""
import requests, time, json, gzip, re, os, sys
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).parent.parent
RAW_PATH = BASE / "scripts" / "ingested_records.json.gz"

DECISION_KEYWORDS = [
    "switched from", "migrated from", "replaced", "chose instead",
    "moved away from", "why we use", "we chose", "we selected",
    "alternatives considered", "comparison", "vs ", "instead of",
    "moved to", "switched to", "migrated to", "evaluated", "shortlisted",
    "why not", "we decided", "our team chose", "we went with",
]

SITES = [
    "softwarerecs",
    "serverfault", 
    "superuser",
    "devops",
    "sysadmin",  # mapped to serverfault
    "security",
]

SEARCH_QUERIES = [
    "switched from to",
    "migrated from to",
    "chose instead of",
    "why we use",
    "alternatives considered",
    "replaced with",
    "moved away from",
    "we selected because",
    "comparison between",
    "vs which better",
    "evaluated options",
    "why not use",
]

def se_request(url, params, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 429:
                time.sleep(30)
                continue
            if r.status_code == 200:
                data = r.json()
                # Check backoff
                if data.get("backoff"):
                    time.sleep(data["backoff"] + 1)
                return data
        except Exception as e:
            time.sleep(5)
    return None

def fetch_questions_for_site(site, known_urls):
    records = []
    base_url = "https://api.stackexchange.com/2.3/search/advanced"
    
    for query in SEARCH_QUERIES:
        params = {
            "site": site,
            "q": query,
            "sort": "votes",
            "order": "desc",
            "pagesize": 100,
            "page": 1,
            "filter": "withbody",  # includes body in response
        }
        
        for page in range(1, 4):  # up to 3 pages = 300 posts per query
            params["page"] = page
            data = se_request(base_url, params)
            if not data or not data.get("items"):
                break
            
            for item in data["items"]:
                url = item.get("link", "")
                if not url or url in known_urls:
                    continue
                
                title = item.get("title", "")
                body = item.get("body", "") or ""
                # Strip HTML tags
                body_clean = re.sub(r'<[^>]+>', ' ', body)
                body_clean = re.sub(r'\s+', ' ', body_clean).strip()
                
                full_text = f"{title}\n\n{body_clean}"
                
                # Quick prefilter
                text_lower = full_text.lower()
                if not any(kw in text_lower for kw in DECISION_KEYWORDS):
                    continue
                
                # Also fetch answers for highly voted questions
                answers_text = ""
                if item.get("answer_count", 0) > 0 and item.get("score", 0) >= 2:
                    ans_data = se_request(
                        f"https://api.stackexchange.com/2.3/questions/{item['question_id']}/answers",
                        {"site": site, "sort": "votes", "order": "desc", "pagesize": 5, "filter": "withbody"}
                    )
                    if ans_data and ans_data.get("items"):
                        for ans in ans_data["items"][:3]:
                            ans_body = ans.get("body", "") or ""
                            ans_clean = re.sub(r'<[^>]+>', ' ', ans_body)
                            answers_text += " " + re.sub(r'\s+', ' ', ans_clean).strip()
                        time.sleep(0.5)
                
                combined = (full_text + " " + answers_text)[:8000]
                
                record = {
                    "url": url,
                    "source_url": url,
                    "title": title,
                    "full_text": combined,
                    "text": combined[:2000],
                    "crawl_date": datetime.utcfromtimestamp(item.get("creation_date", 0)).strftime("%Y-%m-%d"),
                    "segment_id": f"stackexchange_{site}",
                    "score": item.get("score", 0),
                    "domain": f"{site}.stackexchange.com",
                }
                records.append(record)
                known_urls.add(url)
            
            time.sleep(1)  # be polite
            
            if not data.get("has_more"):
                break
        
        time.sleep(0.5)
    
    print(f"  SE/{site}: {len(records)} records", flush=True)
    return records

def ingest_stackexchange(known_urls):
    all_records = []
    for site in SITES:
        recs = fetch_questions_for_site(site, set(known_urls))
        all_records.extend(recs)
    print(f"Stack Exchange total: {len(all_records)}", flush=True)
    return all_records

if __name__ == "__main__":
    with gzip.open(RAW_PATH, "rt", encoding="utf-8") as f:
        existing = json.load(f)
    known = {r.get("url","") for r in existing}
    new_recs = ingest_stackexchange(known)
    
    seen = set(known)
    deduped = []
    for r in new_recs:
        if r["url"] not in seen:
            seen.add(r["url"])
            deduped.append(r)
    
    all_records = existing + deduped
    with gzip.open(RAW_PATH, "wt", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False)
    print(f"Saved {len(all_records):,} total records (+{len(deduped)} new)")
