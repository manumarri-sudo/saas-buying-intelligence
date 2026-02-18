#!/usr/bin/env python3
"""
Reddit ingestion — fetches B2B SaaS buying decision narratives via Reddit API.

Uses PRAW (Python Reddit API Wrapper) with unauthenticated read-only mode
(no OAuth needed for public read-only access via pushshift-style requests).

Also uses Arctic Shift API for historical data and the Reddit public JSON API
for subreddit browsing.

Target subreddits (all public, high B2B signal):
  - r/sysadmin      — 800K+ members, IT tool decisions
  - r/ITManagers    — 150K members, pure buying decisions
  - r/msp           — 150K members, MSP tooling
  - r/devops        — 400K members, DevOps tooling
  - r/securityprofessionals — security tool evaluations
  - r/netsec        — network security tool decisions
  - r/Entrepreneur  — startup tool stack decisions
  - r/smallbusiness — SMB software decisions
  - r/salesforce    — Salesforce vs alternatives
  - r/hubspot       — HubSpot migration discussions
  - r/selfhosted    — self-hosted vs SaaS decisions

No credentials needed for public JSON API approach.
"""

import gzip
import json
import logging
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.config_loader import resolve_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingest_reddit")

# ── Config ────────────────────────────────────────────────────────────────

TARGET_SUBREDDITS = [
    "sysadmin",
    "ITManagers",
    "msp",
    "devops",
    "securityprofessionals",
    "netsec",
    "Entrepreneur",
    "smallbusiness",
    "salesforce",
    "hubspot",
    "selfhosted",
    "aws",
    "cloudcomputing",
    "datacenter",
    "k8s",
    "devops",
]

# Keywords to search within each subreddit
SEARCH_QUERIES = [
    "switched from",
    "migrated from",
    "we chose",
    "we selected",
    "we evaluated",
    "we replaced",
    "moved from",
    "why we use",
    "why we switched",
    "vendor selection",
    "tool evaluation",
    "we adopted",
    "alternatives to",
    "comparing",
    "we went with",
    "we ended up",
]

REDDIT_BASE = "https://www.reddit.com"
ARCTIC_SHIFT_BASE = "https://arctic-shift.photon-reddit.com/api"

HEADERS = {
    "User-Agent": "SaaS-Intel-Research/1.0 (academic; contact: research@example.com)",
    "Accept": "application/json",
}


def passes_prefilter(text: str) -> bool:
    """Quick pre-filter for decision narrative signals."""
    if not text or len(text) < 80:
        return False
    tl = text.lower()
    has_decision = any(kw in tl for kw in [
        "chose", "selected", "switched", "migrated", "evaluated",
        "replaced", "rejected", "moved from", "adopted", "piloted", "trialed",
        "we decided", "we went with", "we ended up", "we picked"
    ])
    has_actor = any(a in tl for a in [
        "we ", "our team", "our company", "our org", "we've ", "we'd ",
        "the team", "the company", "my team", "our shop"
    ])
    return has_decision and has_actor


def reddit_search(subreddit: str, query: str, session: requests.Session,
                  sort: str = "relevance", time_filter: str = "all",
                  limit: int = 100) -> list[dict]:
    """Search a subreddit using Reddit's public JSON API."""
    posts = []
    after = None
    attempts = 0

    while len(posts) < limit and attempts < 5:
        params = {
            "q": query,
            "sort": sort,
            "t": time_filter,
            "limit": min(100, limit - len(posts)),
            "restrict_sr": "true",
        }
        if after:
            params["after"] = after

        url = f"{REDDIT_BASE}/r/{subreddit}/search.json"
        try:
            r = session.get(url, params=params, timeout=15)
            if r.status_code == 429:
                logger.warning("Reddit rate limit — sleeping 60s")
                time.sleep(60)
                continue
            if r.status_code == 403:
                logger.warning(f"  r/{subreddit} is private or restricted")
                break
            r.raise_for_status()
            data = r.json()
            children = data.get("data", {}).get("children", [])
            if not children:
                break
            for child in children:
                post = child.get("data", {})
                posts.append(post)
            after = data.get("data", {}).get("after")
            if not after:
                break
        except Exception as e:
            logger.warning(f"  Search error ({subreddit} / {query}): {e}")
            break
        attempts += 1
        time.sleep(1.5)  # Reddit rate limit: ~1 req/sec safe

    return posts


def fetch_post_comments(permalink: str, session: requests.Session) -> list[str]:
    """Fetch top-level comments for a post."""
    comments = []
    url = f"{REDDIT_BASE}{permalink}.json"
    try:
        r = session.get(url, params={"limit": 200, "depth": 3}, timeout=15)
        r.raise_for_status()
        data = r.json()
        if len(data) < 2:
            return comments
        for child in data[1].get("data", {}).get("children", []):
            comment_data = child.get("data", {})
            body = comment_data.get("body", "")
            if body and body != "[deleted]" and body != "[removed]":
                comments.append(body)
            # Nested replies
            replies = comment_data.get("replies")
            if isinstance(replies, dict):
                for reply_child in replies.get("data", {}).get("children", []):
                    reply_body = reply_child.get("data", {}).get("body", "")
                    if reply_body and reply_body not in ("[deleted]", "[removed]"):
                        comments.append(reply_body)
    except Exception as e:
        pass
    return comments


def fetch_arctic_shift(subreddit: str, keywords: list[str],
                       session: requests.Session, limit: int = 500) -> list[dict]:
    """
    Fetch historical Reddit posts via Arctic Shift API.
    Arctic Shift indexes Reddit data from 2005-2023.
    API docs: https://arctic-shift.photon-reddit.com/
    """
    posts = []
    for keyword in keywords[:5]:  # limit keywords for Arctic Shift
        url = f"{ARCTIC_SHIFT_BASE}/posts/search"
        params = {
            "subreddit": subreddit,
            "q": keyword,
            "limit": 100,
            "sort": "score",
            "min_score": "5",
        }
        try:
            r = session.get(url, params=params, timeout=20)
            if r.status_code in (404, 503, 429):
                time.sleep(5)
                continue
            r.raise_for_status()
            data = r.json()
            posts.extend(data.get("data", []))
        except Exception as e:
            logger.debug(f"Arctic Shift error ({subreddit}/{keyword}): {e}")
        time.sleep(0.5)

    return posts[:limit]


def post_to_record(post: dict, source: str = "reddit_api") -> dict | None:
    """Convert a Reddit post dict to a pipeline record."""
    # Combine title + selftext
    title = post.get("title", "") or ""
    selftext = post.get("selftext", "") or post.get("self_text", "") or post.get("body", "") or ""
    if selftext in ("[deleted]", "[removed]"):
        selftext = ""

    text = f"{title}\n\n{selftext}".strip()
    if not text or len(text) < 50:
        return None

    # Build URL
    permalink = post.get("permalink", "")
    if permalink:
        url = f"https://www.reddit.com{permalink}"
    else:
        post_id = post.get("id", "")
        subreddit = post.get("subreddit", "")
        url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}/"

    # Date
    created = post.get("created_utc", 0) or post.get("created", 0)
    if isinstance(created, (int, float)):
        crawl_date = time.strftime("%Y-%m-%d", time.gmtime(created))
    else:
        crawl_date = str(created)[:10]

    subreddit_name = post.get("subreddit", "reddit")

    return {
        "url": url,
        "text": text[:8000],
        "crawl_date": crawl_date,
        "segment_id": source,
        "domain": "www.reddit.com",
        "source_type": "community_discussion",
        "subreddit": subreddit_name,
    }


def main():
    raw_dir = resolve_path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    existing_path = raw_dir / "ingested_records.json.gz"
    existing_urls: set[str] = set()
    existing_records: list[dict] = []

    if existing_path.exists():
        logger.info("Loading existing records ...")
        with gzip.open(existing_path, "rt", encoding="utf-8") as f:
            existing_records = json.load(f)
        existing_urls = {r.get("url", "") for r in existing_records}
        logger.info(f"  Existing: {len(existing_records):,} records")

    new_records: list[dict] = []
    session = requests.Session()
    session.headers.update(HEADERS)

    # ── Phase 1: Reddit API search ────────────────────────────────────────
    logger.info("=== Phase 1: Reddit public JSON API search ===")

    for subreddit in TARGET_SUBREDDITS:
        logger.info(f"\n--- r/{subreddit} ---")
        sub_posts = []
        seen_post_ids: set[str] = set()

        for query in SEARCH_QUERIES[:8]:  # top 8 queries per sub
            posts = reddit_search(subreddit, query, session, limit=100)
            for p in posts:
                pid = p.get("id", "")
                if pid and pid not in seen_post_ids:
                    seen_post_ids.add(pid)
                    sub_posts.append(p)
            time.sleep(1)

        logger.info(f"  Found {len(sub_posts)} unique posts")
        added = 0

        for post in sub_posts:
            record = post_to_record(post, source="reddit_api")
            if not record:
                continue
            if record["url"] in existing_urls:
                continue
            if not passes_prefilter(record["text"]):
                continue
            new_records.append(record)
            existing_urls.add(record["url"])
            added += 1

            # For high-engagement posts, also fetch comments
            score = post.get("score", 0) or 0
            num_comments = post.get("num_comments", 0) or 0
            if score >= 20 and num_comments >= 5:
                permalink = post.get("permalink", "")
                if permalink:
                    comments = fetch_post_comments(permalink, session)
                    for comment_text in comments:
                        if not passes_prefilter(comment_text):
                            continue
                        import hashlib
                        comment_url = f"https://www.reddit.com{permalink}#c{hashlib.md5(comment_text[:50].encode()).hexdigest()[:8]}"
                        if comment_url in existing_urls:
                            continue
                        comment_record = {
                            "url": comment_url,
                            "text": comment_text[:8000],
                            "crawl_date": record["crawl_date"],
                            "segment_id": "reddit_comments",
                            "domain": "www.reddit.com",
                            "source_type": "community_discussion",
                            "subreddit": subreddit,
                        }
                        new_records.append(comment_record)
                        existing_urls.add(comment_url)
                        added += 1
                    time.sleep(1)

        logger.info(f"  Added {added} new records from r/{subreddit}")

    # ── Phase 2: Arctic Shift historical data ────────────────────────────
    logger.info("\n=== Phase 2: Arctic Shift historical Reddit data ===")

    priority_subs = ["sysadmin", "ITManagers", "msp", "devops", "securityprofessionals"]
    historical_keywords = [
        "switched from", "we chose", "we selected", "we evaluated", "migrated from"
    ]

    for subreddit in priority_subs:
        logger.info(f"  Arctic Shift: r/{subreddit}")
        posts = fetch_arctic_shift(subreddit, historical_keywords, session, limit=300)
        added = 0
        for post in posts:
            record = post_to_record(post, source="arctic_shift")
            if not record:
                continue
            if record["url"] in existing_urls:
                continue
            if not passes_prefilter(record["text"]):
                continue
            new_records.append(record)
            existing_urls.add(record["url"])
            added += 1
        logger.info(f"    Added {added} new records")

    logger.info(f"\nReddit ingestion complete: {len(new_records):,} new records")

    # ── Save ─────────────────────────────────────────────────────────────
    if not new_records:
        logger.info("No new records to add.")
        return 0

    all_records = existing_records + new_records
    logger.info(f"Saving {len(all_records):,} total records ...")
    with gzip.open(existing_path, "wt", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False)
    logger.info(f"Saved ({existing_path.stat().st_size / 1_048_576:.1f} MB)")

    return len(new_records)


if __name__ == "__main__":
    count = main()
    logger.info(f"Done: {count:,} new Reddit records added")
    sys.exit(0 if count >= 0 else 1)
