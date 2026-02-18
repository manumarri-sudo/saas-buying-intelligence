#!/usr/bin/env python3
"""
Targeted HN ingestion — focuses on Ask HN posts and comments about software switching.
"Ask HN: What's your stack?", "Ask HN: We moved from X to Y", etc.
Also fetches full comment trees for top decision-related stories.
"""
import requests, time, json, gzip, re, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).parent.parent
RAW_PATH = BASE / "data" / "raw" / "ingested_records.json.gz"
sys.path.insert(0, str(BASE))
from lib.safe_append import safe_load, safe_append

ACTOR_RE = re.compile(r'\b(we |our team|our company|my team|my company|i |i\'ve|we\'ve|we\'re|we chose|we migrated|we moved|we switched|we replaced)\b', re.I)
VERB_RE = re.compile(r'\b(chose|switched|migrated|replaced|moved|adopted|selected|evaluated|decided|implemented|deployed|went with|ended up)\b', re.I)

def passes_gate(text):
    return bool(VERB_RE.search(text) and ACTOR_RE.search(text))

def algolia_search(query, tags="comment", num_pages=5):
    results = []
    for page in range(num_pages):
        try:
            r = requests.get("https://hn.algolia.com/api/v1/search_by_date", params={
                "query": query,
                "tags": tags,
                "hitsPerPage": 1000,
                "page": page,
            }, timeout=15)
            if r.status_code != 200:
                break
            data = r.json()
            hits = data.get("hits", [])
            results.extend(hits)
            if not data.get("nbPages", 0) > page + 1:
                break
            time.sleep(0.3)
        except:
            break
    return results

def fetch_comment_tree(story_id, known_urls):
    """Fetch all comments for a story via Firebase API"""
    records = []
    try:
        r = requests.get(f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json", timeout=15)
        if r.status_code != 200:
            return records
        story = r.json()
        if not story:
            return records
        
        kids = story.get("kids", [])[:200]  # limit to top 200 comments
        for kid_id in kids:
            url = f"https://news.ycombinator.com/item?id={kid_id}"
            if url in known_urls:
                continue
            try:
                cr = requests.get(f"https://hacker-news.firebaseio.com/v0/item/{kid_id}.json", timeout=10)
                if cr.status_code != 200:
                    continue
                comment = cr.json()
                if not comment or comment.get("deleted") or comment.get("dead"):
                    continue
                text = comment.get("text","") or ""
                text = re.sub(r'<[^>]+>', ' ', text)
                text = re.sub(r'&#\d+;', '', text)
                text = re.sub(r'&[a-z]+;', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()
                if len(text) < 80 or not passes_gate(text):
                    continue
                known_urls.add(url)
                records.append({
                    "url": url, "source_url": url,
                    "full_text": text[:8000], "text": text[:2000],
                    "crawl_date": datetime.utcfromtimestamp(comment.get("time",0)).strftime("%Y-%m-%d"),
                    "segment_id": "hn_targeted_comments",
                    "domain": "news.ycombinator.com",
                })
                time.sleep(0.05)
            except:
                pass
    except:
        pass
    return records

def ingest_hn_comments(known_urls):
    """Direct comment search via Algolia"""
    records = []
    seen = set(known_urls)
    
    comment_queries = [
        "we switched from to",
        "we migrated from to", 
        "we chose instead",
        "we moved away from",
        "we replaced with",
        "we went with because",
        "we selected over",
        "we evaluated and chose",
        "we cancelled subscription",
        "we stopped using",
        "our team switched",
        "our company moved",
        "we ended up using",
        "we decided on",
    ]
    
    for query in comment_queries:
        hits = algolia_search(query, tags="comment", num_pages=3)
        for hit in hits:
            url = hit.get("permalink") or f"https://news.ycombinator.com/item?id={hit.get('objectID','')}"
            if url in seen:
                continue
            text = hit.get("comment_text","") or hit.get("story_text","") or ""
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) < 80 or not passes_gate(text):
                continue
            seen.add(url)
            records.append({
                "url": url, "source_url": url,
                "full_text": text[:8000], "text": text[:2000],
                "crawl_date": hit.get("created_at","")[:10],
                "segment_id": "hn_targeted_comments",
                "domain": "news.ycombinator.com",
            })
    
    print(f"[HN Comments] {len(records)} from {len(comment_queries)} queries", flush=True)
    return records

def ingest_hn_stories_trees(known_urls):
    """Fetch full comment trees for Ask HN stories about software decisions"""
    records = []
    seen = set(known_urls)
    
    story_queries = [
        "Ask HN: What software did you switch",
        "Ask HN: What did you migrate",
        "Ask HN: What tools does your team use",
        "Ask HN: switched from to",
        "Ask HN: migrated away from",
        "Ask HN: What are you using instead",
        "Show HN: we built because switched",
        "we replaced our with",
        "migration from AWS to",
        "migration from to cheaper",
        "why we left for",
        "we moved from heroku",
        "we switched from slack",
        "we replaced jira with",
        "we moved from github to",
    ]
    
    story_ids = set()
    for query in story_queries:
        hits = algolia_search(query, tags="(story,ask_hn,show_hn)", num_pages=2)
        for hit in hits:
            sid = hit.get("objectID")
            if sid and sid not in story_ids:
                story_ids.add(sid)
                # Also add the story itself if it passes
                url = f"https://news.ycombinator.com/item?id={sid}"
                if url not in seen:
                    text = (hit.get("title","") + " " + (hit.get("story_text","") or ""))
                    text = re.sub(r'<[^>]+>', ' ', text).strip()
                    if passes_gate(text) and len(text) > 50:
                        seen.add(url)
                        records.append({
                            "url": url, "source_url": url,
                            "full_text": text[:8000], "text": text[:2000],
                            "crawl_date": hit.get("created_at","")[:10],
                            "segment_id": "hn_targeted_stories",
                            "domain": "news.ycombinator.com",
                        })
    
    print(f"Found {len(story_ids)} HN stories to fetch comment trees for...", flush=True)
    
    # Fetch comment trees in parallel
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = [ex.submit(fetch_comment_tree, sid, seen) for sid in list(story_ids)[:300]]
        for fut in as_completed(futures):
            try:
                recs = fut.result()
                records.extend(recs)
            except:
                pass
    
    print(f"[HN Story Trees] {len(records)} records from {len(story_ids)} stories", flush=True)
    return records

def main():
    print("Loading existing records (snapshot for dedup)...", flush=True)
    existing = safe_load()
    known_urls = {r.get("url","") for r in existing}
    print(f"Loaded {len(existing):,} records", flush=True)

    all_new = []
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = {
            ex.submit(ingest_hn_comments, set(known_urls)): "HN-Comments",
            ex.submit(ingest_hn_stories_trees, set(known_urls)): "HN-Story-Trees",
        }
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                recs = fut.result()
                all_new.extend(recs)
                print(f"✓ {name}: {len(recs)}", flush=True)
            except Exception as e:
                print(f"✗ {name}: {e}", flush=True)

    # Use safe_append — re-reads file under lock, deduplicates, atomic write
    added = safe_append(all_new)
    print(f"✓ Done. New HN records added: {added}", flush=True)

if __name__ == "__main__":
    main()
