#!/usr/bin/env python3
"""
Expansion ingestion — runs Stack Exchange + broader CC review domains + 
more Reddit subreddits + Medium/Substack/Ghost blogs — all in parallel threads.
Single save at end to avoid race conditions.
"""
import requests, time, json, gzip, re, os, sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).parent.parent
RAW_PATH = BASE / "data" / "raw" / "ingested_records.json.gz"
sys.path.insert(0, str(BASE))
from lib.safe_append import safe_load, safe_append

url_lock = threading.Lock()

# ─── PREFILTER ────────────────────────────────────────────────────────────────
VERB_RE = re.compile(r'\b(chose|switched|migrated|replaced|moved|adopted|selected|evaluated|decided|implemented|deployed)\b', re.I)
ACTOR_RE = re.compile(r'\b(we|our team|our company|our org|i |the team|the company|my team|our stack|we\'ve|we\'re)\b', re.I)
REASON_RE = re.compile(r'\b(because|due to|since|as a result|given that|the reason|pricing|cost|support|performance|reliability|scalability|integration|compliance|security|ux|vendor|contract)\b', re.I)

def passes_prefilter(text):
    tl = text.lower()
    return bool(VERB_RE.search(tl) and ACTOR_RE.search(tl))

# ─── STACK EXCHANGE ──────────────────────────────────────────────────────────
def se_get(url, params, retries=3):
    for _ in range(retries):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(30)
                continue
            if r.status_code == 200:
                d = r.json()
                if d.get("backoff"):
                    time.sleep(d["backoff"] + 1)
                return d
        except:
            time.sleep(5)
    return None

SE_SITES = ["softwarerecs", "serverfault", "superuser", "devops", "security", "sre"]
SE_QUERIES = [
    "switched from migrated", "chose instead of", "why we use",
    "replaced with", "moved away from", "we selected because",
    "comparison vs which better", "evaluated shortlisted decided",
    "alternatives considered", "why not use", "we went with",
    "our team chose", "we implemented", "we deployed",
]

def ingest_stackexchange(known_urls):
    records = []
    seen = set(known_urls)
    
    for site in SE_SITES:
        site_count = 0
        for query in SE_QUERIES:
            for page in range(1, 4):
                data = se_get("https://api.stackexchange.com/2.3/search/advanced", {
                    "site": site, "q": query, "sort": "votes", "order": "desc",
                    "pagesize": 100, "page": page, "filter": "withbody",
                })
                if not data or not data.get("items"):
                    break
                
                for item in data["items"]:
                    url = item.get("link", "")
                    if not url or url in seen:
                        continue
                    title = item.get("title", "")
                    body = re.sub(r'<[^>]+>', ' ', item.get("body","") or "")
                    body = re.sub(r'\s+', ' ', body).strip()
                    full = (title + "\n\n" + body)[:8000]
                    if not passes_prefilter(full):
                        continue
                    seen.add(url)
                    records.append({
                        "url": url, "source_url": url, "title": title,
                        "full_text": full, "text": full[:2000],
                        "crawl_date": datetime.utcfromtimestamp(item.get("creation_date",0)).strftime("%Y-%m-%d"),
                        "segment_id": f"stackexchange_{site}",
                        "domain": f"{site}.stackexchange.com",
                    })
                    site_count += 1
                
                time.sleep(0.8)
                if not data.get("has_more"):
                    break
            time.sleep(0.3)
        print(f"  SE/{site}: {site_count}", flush=True)
    
    print(f"[SE] Total: {len(records)}", flush=True)
    return records

# ─── REDDIT (more subreddits) ────────────────────────────────────────────────
REDDIT_HEADERS = {"User-Agent": "b2b-saas-research/1.0 (academic; contact@example.com)"}
EXTRA_SUBS = [
    "projectmanagement", "cscareerquestions", "startups", "SaaS",
    "devopsish", "kubernetes", "cloudarchitecture", "microservices",
    "businessanalysis", "ProductManagement", "sales", "marketing",
    "CustomerSuccess", "B2Bsales", "techsupport", "homelab",
    "linuxadmin", "networking", "cybersecurity", "cloudsecurity",
]
REDDIT_QUERIES = [
    "switched from to SaaS", "migrated from software",
    "replaced with tool", "why we chose platform",
    "moved away from vendor", "we selected because pricing",
    "team decided software", "our company uses instead",
    "cancelled subscription switched", "churned moved to",
]

def reddit_search(sub, query, seen):
    records = []
    url = f"https://www.reddit.com/r/{sub}/search.json"
    params = {"q": query, "sort": "relevance", "t": "all", "limit": 100}
    try:
        r = requests.get(url, params=params, headers=REDDIT_HEADERS, timeout=15)
        if r.status_code == 429:
            time.sleep(30)
            return records
        if r.status_code != 200:
            return records
        data = r.json()
        for post in data.get("data", {}).get("children", []):
            d = post.get("data", {})
            post_url = "https://reddit.com" + d.get("permalink","")
            if post_url in seen:
                continue
            title = d.get("title","")
            body = d.get("selftext","") or ""
            full = (title + "\n\n" + body)[:8000]
            if not passes_prefilter(full):
                continue
            seen.add(post_url)
            records.append({
                "url": post_url, "source_url": post_url, "title": title,
                "full_text": full, "text": full[:2000],
                "crawl_date": datetime.utcfromtimestamp(d.get("created_utc",0)).strftime("%Y-%m-%d"),
                "segment_id": "reddit_api_exp",
                "domain": "reddit.com",
                "score": d.get("score",0),
            })
    except:
        pass
    return records

def ingest_reddit_extra(known_urls):
    records = []
    seen = set(known_urls)
    for sub in EXTRA_SUBS:
        sub_count = 0
        for query in REDDIT_QUERIES:
            new = reddit_search(sub, query, seen)
            records.extend(new)
            sub_count += len(new)
            time.sleep(1.5)
        if sub_count:
            print(f"  r/{sub}: {sub_count}", flush=True)
        time.sleep(0.5)
    print(f"[Reddit-Extra] Total: {len(records)}", flush=True)
    return records

# ─── COMMON CRAWL: BROADER DOMAINS ───────────────────────────────────────────
CC_INDEX = "CC-MAIN-2024-10"
CC_EXTRA_DOMAINS = [
    # Review/comparison sites
    "g2.com/reviews/*",
    "capterra.com/reviews/*",
    "getapp.com/reviews/*",
    "softwareadvice.com/*",
    "gartner.com/reviews/*",
    "peerspot.com/*",
    "crozdesk.com/*",
    "sourceforge.net/software/*",
    "alternativeto.net/*",
    # Tech blogs with decision posts
    "medium.com/*",
    "substack.com/*",
    "firstround.com/review/*",
    "saastr.com/*",
    "tomtunguz.com/*",
    "a16z.com/*",
    "blog.hubspot.com/*",
    "blog.salesforce.com/*",
    "atlassian.com/blog/*",
    # Engineering blogs
    "engineering.fb.com/*",
    "netflixtechblog.com/*",
    "slack.engineering/*",
    "stripe.com/blog/*",
    "dropbox.tech/*",
    "shopify.engineering/*",
    "github.blog/*",
    "engineering.linkedin.com/*",
    "tech.target.com/*",
    "engineering.twitter.com/*",
]

DECISION_KWS = ["switched", "migrated", "replaced", "chose", "moved from", "alternatives", "we decided", "we selected"]

def fetch_cc_domain(domain_pattern, known_urls):
    records = []
    seen = set(known_urls)
    
    for idx_name in ["CC-MAIN-2024-10", "CC-MAIN-2024-18", "CC-MAIN-2023-50"]:
        try:
            cdx_url = f"https://index.commoncrawl.org/{idx_name}-index"
            params = {"url": domain_pattern, "output": "json", "limit": 200, "fl": "url,filename,offset,length,status"}
            r = requests.get(cdx_url, params=params, timeout=30)
            if r.status_code != 200:
                continue
            
            lines = [l for l in r.text.strip().split("\n") if l]
            for line in lines[:80]:  # limit per domain/index
                try:
                    rec = json.loads(line)
                    url = rec.get("url","")
                    if rec.get("status","") not in ("200","") or url in seen:
                        continue
                    
                    # Fetch WARC bytes
                    offset = int(rec.get("offset",0))
                    length = int(rec.get("length",0))
                    if length > 500000:
                        continue
                    
                    warc_url = f"https://data.commoncrawl.org/{rec['filename']}"
                    warc_r = requests.get(warc_url, 
                        headers={"Range": f"bytes={offset}-{offset+length-1}"},
                        timeout=30)
                    if warc_r.status_code not in (200, 206):
                        continue
                    
                    # Extract text from WARC
                    raw = warc_r.content
                    # Find HTTP response body
                    header_end = raw.find(b"\r\n\r\n", raw.find(b"\r\n\r\n")+4)
                    if header_end < 0:
                        header_end = raw.find(b"\r\n\r\n")
                    body_bytes = raw[header_end+4:] if header_end >= 0 else raw
                    
                    try:
                        html = body_bytes.decode("utf-8", errors="ignore")
                    except:
                        continue
                    
                    # Strip HTML
                    text = re.sub(r'<script[^>]*>.*?</script>', ' ', html, flags=re.S|re.I)
                    text = re.sub(r'<style[^>]*>.*?</style>', ' ', text, flags=re.S|re.I)
                    text = re.sub(r'<[^>]+>', ' ', text)
                    text = re.sub(r'&[a-z]+;', ' ', text)
                    text = re.sub(r'\s+', ' ', text).strip()
                    
                    if len(text) < 200:
                        continue
                    
                    text_lower = text.lower()
                    if not any(kw in text_lower for kw in DECISION_KWS):
                        continue
                    if not passes_prefilter(text):
                        continue
                    
                    seen.add(url)
                    records.append({
                        "url": url, "source_url": url,
                        "full_text": text[:8000], "text": text[:2000],
                        "crawl_date": datetime.utcnow().strftime("%Y-%m-%d"),
                        "segment_id": f"cc_exp_{domain_pattern.split('.')[0]}",
                        "domain": domain_pattern.split('/')[0],
                    })
                    time.sleep(0.3)
                except:
                    continue
        except:
            pass
    
    return records

def ingest_cc_broader(known_urls):
    all_records = []
    # Use a thread pool for CC domains too
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(fetch_cc_domain, d, set(known_urls)): d for d in CC_EXTRA_DOMAINS}
        for future in as_completed(futures):
            domain = futures[future]
            try:
                recs = future.result()
                all_records.extend(recs)
                if recs:
                    print(f"  CC/{domain}: {len(recs)}", flush=True)
            except Exception as e:
                print(f"  CC/{domain} error: {e}", flush=True)
    print(f"[CC-Broader] Total: {len(all_records)}", flush=True)
    return all_records

# ─── SAASTR / FIRSTROUND via their APIs ──────────────────────────────────────
def ingest_saastr_rss(known_urls):
    """Scrape SaaStr blog RSS feed for decision-related posts"""
    records = []
    seen = set(known_urls)
    feeds = [
        "https://www.saastr.com/feed/",
        "https://review.firstround.com/feed.xml",
        "https://tomtunguz.com/index.xml",
    ]
    for feed_url in feeds:
        try:
            r = requests.get(feed_url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code != 200:
                continue
            # Extract links from RSS
            links = re.findall(r'<link>([^<]+)</link>', r.text)
            links += re.findall(r'<link rel="alternate"[^>]+href="([^"]+)"', r.text)
            for link in links[:50]:
                if link in seen or not link.startswith("http"):
                    continue
                try:
                    pr = requests.get(link, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
                    if pr.status_code != 200:
                        continue
                    html = pr.text
                    text = re.sub(r'<script[^>]*>.*?</script>', ' ', html, flags=re.S|re.I)
                    text = re.sub(r'<style[^>]*>.*?</style>', ' ', text, flags=re.S|re.I)
                    text = re.sub(r'<[^>]+>', ' ', text)
                    text = re.sub(r'\s+', ' ', text).strip()
                    if len(text) < 300 or not passes_prefilter(text):
                        continue
                    seen.add(link)
                    records.append({
                        "url": link, "source_url": link,
                        "full_text": text[:8000], "text": text[:2000],
                        "crawl_date": datetime.utcnow().strftime("%Y-%m-%d"),
                        "segment_id": "saastr_rss",
                        "domain": link.split('/')[2] if '//' in link else link[:30],
                    })
                    time.sleep(0.5)
                except:
                    pass
        except:
            pass
    print(f"[RSS blogs] Total: {len(records)}", flush=True)
    return records

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("Loading existing records (snapshot for dedup)...", flush=True)
    existing = safe_load()
    known_urls = {r.get("url","") for r in existing}
    print(f"Loaded {len(existing):,} existing records", flush=True)

    all_new = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(ingest_stackexchange, set(known_urls)): "StackExchange",
            executor.submit(ingest_reddit_extra, set(known_urls)): "Reddit-Extra",
            executor.submit(ingest_cc_broader, set(known_urls)): "CC-Broader",
            executor.submit(ingest_saastr_rss, set(known_urls)): "RSS-Blogs",
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                recs = future.result()
                all_new.extend(recs)
                print(f"✓ {name}: {len(recs)} records", flush=True)
            except Exception as e:
                print(f"✗ {name} failed: {e}", flush=True)

    # Use safe_append — atomic read-under-lock + write, eliminates race conditions
    added = safe_append(all_new)
    print(f"✓ Saved. New records added: {added}", flush=True)

if __name__ == "__main__":
    main()
