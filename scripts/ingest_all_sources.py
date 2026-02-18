#!/usr/bin/env python3
"""
Unified multi-source ingestion — runs ALL sources in parallel threads,
collects ALL results in memory, deduplicates, then saves ONCE.

Sources (all in parallel):
  1. Hacker News — Algolia search API (no auth, fast)
  2. Reddit — public JSON API for r/sysadmin, r/ITManagers, r/msp, r/devops, etc.
  3. dev.to — public REST API, migration/switching articles
  4. Common Crawl — TrustRadius + Clutch WARC extraction

No intermediate saves. Single write at the end. No overwrites.
"""

import gzip
import hashlib
import json
import logging
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from pathlib import Path
from threading import Lock

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.config_loader import resolve_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingest_all")

# ─────────────────────────────────────────────────────────────────────────────
# SHARED UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

HEADERS = {"User-Agent": "SaaS-Intel-Research/1.0 (academic data collection)"}


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def html_to_text(html: str) -> str:
    """Strip HTML tags."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        for t in soup.find_all(["script", "style", "nav", "footer", "header"]):
            t.decompose()
        return soup.get_text(" ", strip=True)
    except Exception:
        return re.sub(r"<[^>]+>", " ", html).strip()


def passes_prefilter(text: str) -> bool:
    """Quick pre-filter: must have decision verb + actor signal."""
    if not text or len(text) < 80:
        return False
    tl = text.lower()
    has_decision = any(kw in tl for kw in [
        "chose", "selected", "switched", "migrated", "evaluated",
        "replaced", "rejected", "moved from", "adopted", "piloted",
        "trialed", "we decided", "we went with", "we ended up",
        "we picked", "procured", "purchased",
    ])
    has_actor = any(a in tl for a in [
        "we ", "our team", "our company", "our org", "we've ",
        "the team selected", "the company chose", "they selected",
        "the client", "my team",
    ])
    return has_decision and has_actor


def make_record(url: str, text: str, date: str, source: str, domain: str,
                source_type: str = "community_discussion") -> dict:
    return {
        "url": url,
        "text": text[:8000],
        "crawl_date": date,
        "segment_id": source,
        "domain": domain,
        "source_type": source_type,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 1: HACKER NEWS
# ─────────────────────────────────────────────────────────────────────────────

HN_ALGOLIA = "https://hn.algolia.com/api/v1"
HN_FIREBASE = "https://hacker-news.firebaseio.com/v0"

HN_QUERIES = [
    "switched from", "migrated from", "we chose", "we selected",
    "we evaluated", "we replaced", "moved from", "we rejected",
    "why we use", "why we switched", "how we chose", "we adopted",
    "vendor selection", "we went with", "after evaluating",
    "our team chose", "our company selected", "we ended up with",
]


def hn_algolia_search(query: str, content_type: str, session: requests.Session,
                      max_hits: int = 300) -> list[dict]:
    """Search HN Algolia. content_type: 'story' or 'comment'"""
    hits = []
    page = 0
    tag = "comment" if content_type == "comment" else "(story,ask_hn,show_hn)"
    while len(hits) < max_hits:
        try:
            r = session.get(f"{HN_ALGOLIA}/search_by_date",
                            params={"query": query, "tags": tag,
                                    "hitsPerPage": 200, "page": page},
                            timeout=15)
            r.raise_for_status()
            data = r.json()
            batch = data.get("hits", [])
            if not batch:
                break
            hits.extend(batch)
            page += 1
            if page >= min(data.get("nbPages", 1), 5):
                break
        except Exception as e:
            logger.debug(f"HN Algolia error ({query}): {e}")
            break
        time.sleep(0.15)
    return hits[:max_hits]


def fetch_hn_comments(item_id: int, session: requests.Session,
                      depth: int = 0) -> list[str]:
    """Recursively fetch HN comments."""
    if depth > 2:
        return []
    try:
        r = session.get(f"{HN_FIREBASE}/item/{item_id}.json", timeout=10)
        item = r.json()
        if not item:
            return []
    except Exception:
        return []

    texts = []
    text = html_to_text(item.get("text", "") or "")
    if text and len(text) > 80:
        texts.append(text)

    kids = item.get("kids", [])[:20]
    if kids and depth < 2:
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = [ex.submit(fetch_hn_comments, kid, session, depth + 1)
                    for kid in kids]
            for f in as_completed(futs):
                try:
                    texts.extend(f.result())
                except Exception:
                    pass
    return texts


def ingest_hn(existing_urls: set) -> list[dict]:
    """Collect HN records via Algolia API."""
    logger.info("[HN] Starting ingestion ...")
    session = make_session()
    records = []
    seen_ids: set[int] = set()
    story_meta: dict[int, dict] = {}

    # Phase 1: story search
    for q in HN_QUERIES:
        for hit in hn_algolia_search(q, "story", session, max_hits=200):
            sid = int(hit.get("objectID", 0) or 0)
            if sid and sid not in seen_ids:
                seen_ids.add(sid)
                story_meta[sid] = {
                    "title": hit.get("title", ""),
                    "points": hit.get("points", 0) or 0,
                    "num_comments": hit.get("num_comments", 0) or 0,
                    "created_at": hit.get("created_at", ""),
                }

    logger.info(f"[HN] Found {len(seen_ids)} unique stories")

    # Phase 2: comment search
    for q in HN_QUERIES[:10]:
        for hit in hn_algolia_search(q, "comment", session, max_hits=300):
            text = html_to_text(hit.get("comment_text", "") or "")
            if not passes_prefilter(text):
                continue
            url = f"https://news.ycombinator.com/item?id={hit.get('objectID','')}"
            if url in existing_urls:
                continue
            date = (hit.get("created_at", "") or "")[:10]
            records.append(make_record(url, text, date, "hn_comments",
                                       "news.ycombinator.com"))
            existing_urls.add(url)

    logger.info(f"[HN] {len(records)} comment records from direct search")

    # Phase 3: fetch comments for top stories
    top_stories = sorted(seen_ids,
                         key=lambda s: story_meta[s]["points"] *
                                       (story_meta[s]["num_comments"] + 1),
                         reverse=True)[:200]

    story_session = make_session()
    for sid in top_stories:
        meta = story_meta[sid]
        story_url = f"https://news.ycombinator.com/item?id={sid}"
        if story_url not in existing_urls:
            title = meta.get("title", "")
            if title:
                date = (meta.get("created_at", "") or "")[:10]
                story_record = make_record(story_url, title, date,
                                           "hn_stories", "news.ycombinator.com")
                records.append(story_record)
                existing_urls.add(story_url)

        # Fetch top-level comments
        try:
            r = story_session.get(f"{HN_FIREBASE}/item/{sid}.json", timeout=10)
            item = r.json()
            if not item:
                continue
            kids = item.get("kids", [])[:30]
            date = time.strftime("%Y-%m-%d", time.gmtime(item.get("time", 0)))
            with ThreadPoolExecutor(max_workers=6) as ex:
                futs = [ex.submit(fetch_hn_comments, kid, story_session, 0)
                        for kid in kids]
                for f in as_completed(futs):
                    for text in f.result():
                        if not passes_prefilter(text):
                            continue
                        h = hashlib.md5(text[:50].encode()).hexdigest()[:8]
                        url = f"{story_url}#c{h}"
                        if url in existing_urls:
                            continue
                        records.append(make_record(url, text, date,
                                                   "hn_comments", "news.ycombinator.com"))
                        existing_urls.add(url)
        except Exception:
            pass
        time.sleep(0.05)

    logger.info(f"[HN] Done: {len(records)} total records")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 2: REDDIT
# ─────────────────────────────────────────────────────────────────────────────

REDDIT_BASE = "https://www.reddit.com"
SUBREDDITS = [
    "sysadmin", "ITManagers", "msp", "devops",
    "securityprofessionals", "Entrepreneur", "smallbusiness",
    "salesforce", "hubspot", "selfhosted", "aws", "netsec",
]
REDDIT_QUERIES = [
    "switched from", "we chose", "we selected", "migrated from",
    "we evaluated", "we replaced", "why we use", "we went with",
]


def reddit_search(sub: str, query: str, session: requests.Session,
                  limit: int = 100) -> list[dict]:
    """Search a subreddit using public JSON API."""
    posts = []
    after = None
    for _ in range(3):
        params = {"q": query, "sort": "relevance", "t": "all",
                  "limit": limit, "restrict_sr": "true"}
        if after:
            params["after"] = after
        try:
            r = session.get(f"{REDDIT_BASE}/r/{sub}/search.json",
                            params=params, timeout=15)
            if r.status_code == 429:
                time.sleep(30)
                continue
            if r.status_code in (403, 404):
                break
            r.raise_for_status()
            data = r.json()
            children = data.get("data", {}).get("children", [])
            posts.extend(c.get("data", {}) for c in children)
            after = data.get("data", {}).get("after")
            if not after:
                break
        except Exception as e:
            logger.debug(f"Reddit error ({sub}/{query}): {e}")
            break
        time.sleep(2)
    return posts


def ingest_reddit(existing_urls: set) -> list[dict]:
    """Collect Reddit decision narratives via public JSON API."""
    logger.info("[Reddit] Starting ingestion ...")
    session = make_session()
    records = []

    for sub in SUBREDDITS:
        logger.info(f"[Reddit] r/{sub}")
        seen_ids: set[str] = set()
        sub_posts: list[dict] = []

        for q in REDDIT_QUERIES:
            for p in reddit_search(sub, q, session, limit=100):
                pid = p.get("id", "")
                if pid and pid not in seen_ids:
                    seen_ids.add(pid)
                    sub_posts.append(p)
            time.sleep(2)

        added = 0
        for post in sub_posts:
            title = post.get("title", "") or ""
            selftext = post.get("selftext", "") or ""
            if selftext in ("[deleted]", "[removed]"):
                selftext = ""
            text = f"{title}\n\n{selftext}".strip()
            if not passes_prefilter(text):
                continue

            permalink = post.get("permalink", "")
            url = f"https://www.reddit.com{permalink}" if permalink else ""
            if not url or url in existing_urls:
                continue

            created = post.get("created_utc", 0) or 0
            date = time.strftime("%Y-%m-%d", time.gmtime(created))
            records.append(make_record(url, text, date, "reddit_api",
                                       "www.reddit.com"))
            existing_urls.add(url)
            added += 1

        logger.info(f"[Reddit] r/{sub}: {added} records")

    logger.info(f"[Reddit] Done: {len(records)} total records")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 3: dev.to
# ─────────────────────────────────────────────────────────────────────────────

DEVTO_API = "https://dev.to/api"
DEVTO_TAGS = [
    "migration", "devops", "aws", "kubernetes", "tools",
    "productivity", "architecture", "backend", "cloud", "saas",
    "startup", "engineering", "platform", "security", "monitoring",
]
TITLE_SIGNALS = [
    "switched", "migrated", "migration", "chose", "selected",
    "moved from", "vs ", "compared", "why we", "how we",
    "replaced", "alternatives", "evaluation", "deciding",
]


def ingest_devto(existing_urls: set) -> list[dict]:
    """Collect dev.to articles via public API."""
    logger.info("[devto] Starting ingestion ...")
    session = make_session()
    records = []
    seen_ids: set[int] = set()
    candidates: list[dict] = []

    # Browse by tag
    for tag in DEVTO_TAGS:
        try:
            for page in range(1, 6):
                r = session.get(f"{DEVTO_API}/articles",
                                params={"tag": tag, "per_page": 1000, "page": page},
                                timeout=20)
                r.raise_for_status()
                batch = r.json()
                if not batch:
                    break
                for a in batch:
                    aid = a.get("id", 0)
                    title = (a.get("title", "") or "").lower()
                    if aid and aid not in seen_ids:
                        if any(s in title for s in TITLE_SIGNALS):
                            seen_ids.add(aid)
                            candidates.append(a)
                if len(batch) < 1000:
                    break
                time.sleep(0.5)
        except Exception as e:
            logger.debug(f"devto tag={tag} error: {e}")
        time.sleep(0.3)

    logger.info(f"[devto] {len(candidates)} candidate articles")

    # Fetch full content
    for article in candidates:
        aid = article.get("id", 0)
        url = article.get("url", f"https://dev.to/a/{aid}")
        if url in existing_urls:
            continue

        try:
            r = session.get(f"{DEVTO_API}/articles/{aid}", timeout=15)
            r.raise_for_status()
            data = r.json()
            body = data.get("body_markdown", "") or data.get("body_html", "") or ""
            if "<" in body:
                body = html_to_text(body)
        except Exception:
            body = article.get("description", "")

        title = article.get("title", "")
        text = f"{title}\n\n{body}".strip()
        if not passes_prefilter(text):
            time.sleep(0.3)
            continue

        date = (article.get("published_at", "") or "")[:10]
        records.append(make_record(url, text, date, "devto_api", "dev.to", "tech_blog"))
        existing_urls.add(url)
        time.sleep(0.3)

    logger.info(f"[devto] Done: {len(records)} total records")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 4: Common Crawl (TrustRadius + Clutch)
# ─────────────────────────────────────────────────────────────────────────────

CC_INDEXES = ["CC-MAIN-2024-46", "CC-MAIN-2024-42", "CC-MAIN-2024-38"]
CDX_BASE = "https://index.commoncrawl.org"
CC_DATA_BASE = "https://data.commoncrawl.org"

CC_TARGETS = [
    ("www.trustradius.com/products/*", "trustradius", 1000),
    ("clutch.co/profile/*", "clutch", 1000),
    ("www.saastr.com/*", "blog", 300),
    ("review.firstround.com/*", "blog", 200),
    ("*.hashnode.dev/*", "blog", 300),
    ("dzone.com/articles/*", "blog", 300),
]


def cdx_query(url_pattern: str, crawl: str, session: requests.Session,
              limit: int = 500) -> list[dict]:
    """Query CDX index."""
    results = []
    try:
        r = session.get(f"{CDX_BASE}/{crawl}-index",
                        params={"url": url_pattern, "output": "json",
                                "fl": "timestamp,filename,offset,length,url",
                                "filter": "statuscode:200", "limit": limit},
                        timeout=20)
        if r.status_code == 404:
            return []
        r.raise_for_status()
        for line in r.text.strip().split("\n"):
            if line and "{" in line:
                try:
                    results.append(json.loads(line))
                except Exception:
                    pass
    except Exception as e:
        logger.debug(f"CDX error ({url_pattern}): {e}")
    return results


def fetch_warc_text(rec: dict, session: requests.Session) -> str:
    """Fetch and extract text from a WARC byte range."""
    try:
        url = f"{CC_DATA_BASE}/{rec['filename']}"
        offset = int(rec["offset"])
        length = int(rec["length"])
        r = session.get(url, headers={"Range": f"bytes={offset}-{offset+length-1}"},
                        timeout=30)
        r.raise_for_status()
        content = r.content
        try:
            import gzip as gz_mod
            content = gz_mod.decompress(content)
        except Exception:
            pass
        text = content.decode("utf-8", errors="replace")
        # Skip HTTP headers
        sep = text.find("\r\n\r\n")
        if sep == -1:
            sep = text.find("\n\n")
        if sep != -1:
            text = text[sep + 4:]
        return html_to_text(text)
    except Exception as e:
        logger.debug(f"WARC fetch error: {e}")
        return ""


def ingest_cc_reviews(existing_urls: set) -> list[dict]:
    """Extract decision narratives from targeted CC WARC files."""
    logger.info("[CC-Reviews] Starting ingestion ...")
    session = make_session()
    records = []

    for url_pattern, source_key, max_recs in CC_TARGETS:
        logger.info(f"[CC-Reviews] Target: {url_pattern}")
        cdx_records: list[dict] = []
        seen_urls: set[str] = set()

        for crawl in CC_INDEXES:
            batch = cdx_query(url_pattern, crawl, session, limit=min(500, max_recs))
            for r in batch:
                u = r.get("url", "")
                if u and u not in seen_urls and u not in existing_urls:
                    seen_urls.add(u)
                    cdx_records.append(r)
            if len(cdx_records) >= max_recs:
                break
            time.sleep(0.3)

        logger.info(f"[CC-Reviews] {len(cdx_records)} CDX records for {url_pattern}")

        # Parallel WARC fetch
        target_records = cdx_records[:max_recs]

        def process_one(rec: dict) -> list[dict]:
            text = fetch_warc_text(rec, session)
            if not passes_prefilter(text):
                return []
            url = rec.get("url", "")
            ts = rec.get("timestamp", "")
            date = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}" if len(ts) >= 8 else ""
            domain = url.split("/")[2] if "/" in url else "unknown"
            source_type = "case_study" if source_key in ("trustradius", "clutch") else "tech_blog"
            return [make_record(url, text, date, f"cc_{source_key}", domain, source_type)]

        with ThreadPoolExecutor(max_workers=12) as ex:
            futs = {ex.submit(process_one, rec): rec for rec in target_records}
            done = 0
            for f in as_completed(futs):
                try:
                    for r in f.result():
                        if r["url"] not in existing_urls:
                            records.append(r)
                            existing_urls.add(r["url"])
                except Exception:
                    pass
                done += 1
                if done % 200 == 0:
                    logger.info(f"[CC-Reviews] {url_pattern}: {done}/{len(target_records)} done, "
                                f"{len(records)} records total")

    logger.info(f"[CC-Reviews] Done: {len(records)} total records")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — run all sources in parallel
# ─────────────────────────────────────────────────────────────────────────────

def main():
    raw_dir = resolve_path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    existing_path = raw_dir / "ingested_records.json.gz"

    logger.info("Loading existing records ...")
    t0 = time.time()
    with gzip.open(existing_path, "rt", encoding="utf-8") as f:
        existing_records: list[dict] = json.load(f)
    logger.info(f"Loaded {len(existing_records):,} existing records in {time.time()-t0:.1f}s")

    # Shared set of known URLs — protected by lock for thread safety
    url_lock = Lock()
    known_urls: set[str] = {r.get("url", "") for r in existing_records}

    def safe_existing_urls() -> set:
        """Return a copy of known_urls for each thread to use independently."""
        with url_lock:
            return set(known_urls)

    # Run all 4 sources in parallel threads
    logger.info("Starting all sources in parallel ...")
    all_new: list[dict] = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures: dict[Future, str] = {
            executor.submit(ingest_hn, safe_existing_urls()): "HN",
            executor.submit(ingest_reddit, safe_existing_urls()): "Reddit",
            executor.submit(ingest_devto, safe_existing_urls()): "devto",
            executor.submit(ingest_cc_reviews, safe_existing_urls()): "CC-Reviews",
        }

        for future in as_completed(futures):
            source = futures[future]
            try:
                results = future.result()
                logger.info(f"  [{source}] returned {len(results):,} records")
                all_new.extend(results)
            except Exception as e:
                logger.error(f"  [{source}] FAILED: {e}", exc_info=True)

    logger.info(f"All sources done: {len(all_new):,} raw new records")

    # Deduplicate by URL (keep first occurrence)
    all_urls_seen: set[str] = {r.get("url", "") for r in existing_records}
    deduped_new: list[dict] = []
    for r in all_new:
        url = r.get("url", "")
        if url and url not in all_urls_seen:
            all_urls_seen.add(url)
            deduped_new.append(r)

    logger.info(f"After dedup: {len(deduped_new):,} unique new records")

    if not deduped_new:
        logger.info("No new records to add.")
        return 0

    # Single save
    all_records = existing_records + deduped_new
    logger.info(f"Saving {len(all_records):,} total records to {existing_path} ...")
    t0 = time.time()
    with gzip.open(existing_path, "wt", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False)
    logger.info(f"Saved in {time.time()-t0:.1f}s ({existing_path.stat().st_size / 1_048_576:.1f} MB)")

    # Print source breakdown
    from collections import Counter
    source_counts = Counter(r.get("segment_id", "unknown") for r in deduped_new)
    logger.info("New records by source:")
    for src, cnt in source_counts.most_common():
        logger.info(f"  {src}: {cnt:,}")

    return len(deduped_new)


if __name__ == "__main__":
    total = main()
    logger.info(f"=== DONE: {total:,} new records added ===")
    sys.exit(0 if total >= 0 else 1)
