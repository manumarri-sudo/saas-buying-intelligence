#!/usr/bin/env python3
"""
Hacker News ingestion — fetches decision-narrative-rich items via HN Firebase API.

Strategy:
  1. Fetch Ask HN / Show HN stories via the official public Firebase API
  2. Download comments for stories that contain relevant keywords in title
  3. Filter individual comments for 3-condition narrative gate signals
  4. Merge into existing ingested_records.json.gz

HN Firebase API: https://hacker-news.firebaseio.com/v0/
No auth required. Returns JSON. Unofficial rate: ~1000 req/min safe.
"""

import gzip
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.config_loader import resolve_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingest_hackernews")

# ── Config ────────────────────────────────────────────────────────────────

HN_BASE = "https://hacker-news.firebaseio.com/v0"
HN_ALGOLIA = "https://hn.algolia.com/api/v1"

# Algolia HN search API (no auth, 1000 req/hr soft limit)
# More flexible than Firebase for keyword search

DECISION_KEYWORDS = [
    "switched from", "migrated from", "we chose", "we selected",
    "we evaluated", "we replaced", "moved from", "we rejected",
    "why we use", "why we switched", "how we chose", "we adopted",
    "we procured", "vendor selection", "tool selection", "we decided",
    "we went with", "we picked", "we trialed", "we piloted",
    "after evaluating", "after comparing", "our team chose",
    "our company chose", "the team selected", "we ended up with",
    "we migrated away", "we transitioned from",
]

TITLE_KEYWORDS = [
    "switched", "migrated", "chose", "selected", "from .* to",
    "vs ", "compared", "why we", "how we", "moved from",
    "tool for", "stack", "alternatives", "replaced",
    "new tool", "vendor", "evaluation", "choosing",
]

# HN tags to search
TAGS = ["ask_hn", "show_hn", "story"]


def search_hn_algolia(query: str, content_type: str = "story",
                      max_hits: int = 1000) -> list[dict]:
    """Search HN via Algolia API — returns list of items.
    content_type: 'story' or 'comment'
    """
    hits = []
    page = 0
    # Use search_by_date for comments (more results), search for stories
    endpoint = f"{HN_ALGOLIA}/search_by_date"
    while len(hits) < max_hits:
        params = {
            "query": query,
            "hitsPerPage": 200,
            "page": page,
        }
        # Add type tag — note: tags parameter uses parentheses for OR grouping
        if content_type == "comment":
            params["tags"] = "comment"
        elif content_type == "story":
            params["tags"] = "(story,ask_hn,show_hn)"
        try:
            r = requests.get(endpoint, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            batch = data.get("hits", [])
            if not batch:
                break
            # Filter out very low quality
            batch = [h for h in batch if h.get("points", 0) > 0 or
                     h.get("num_comments", 0) > 0 or content_type == "comment"]
            hits.extend(batch)
            page += 1
            if page >= min(data.get("nbPages", 1), 10):  # max 10 pages per query
                break
        except Exception as e:
            logger.warning(f"Algolia search error ({query}): {e}")
            break
    return hits[:max_hits]


def fetch_hn_item(item_id: int, session: requests.Session) -> dict | None:
    """Fetch a single HN item via Firebase API."""
    try:
        r = session.get(f"{HN_BASE}/item/{item_id}.json", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def get_all_comments(item: dict, session: requests.Session, depth: int = 0) -> list[str]:
    """Recursively collect all comment texts from a story."""
    if depth > 3:  # limit depth
        return []
    texts = []
    kids = item.get("kids", [])
    if not kids:
        return texts

    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(fetch_hn_item, kid, session): kid for kid in kids[:50]}
        for fut in as_completed(futures):
            child = fut.result()
            if not child or child.get("deleted") or child.get("dead"):
                continue
            text = child.get("text", "") or ""
            if text:
                texts.append(text)
            # recurse
            if child.get("kids"):
                texts.extend(get_all_comments(child, session, depth + 1))
    return texts


def html_to_text(html: str) -> str:
    """Strip HTML tags from HN comment text."""
    try:
        from bs4 import BeautifulSoup
        return BeautifulSoup(html, "lxml").get_text(" ", strip=True)
    except Exception:
        import re
        return re.sub(r"<[^>]+>", " ", html).strip()


def passes_prefilter(text: str) -> bool:
    """Quick keyword check before full gate evaluation."""
    tl = text.lower()
    has_decision = any(kw in tl for kw in [
        "chose", "selected", "switched", "migrated", "evaluated",
        "replaced", "rejected", "moved from", "adopted", "piloted", "trialed"
    ])
    has_actor = any(a in tl for a in ["we ", "our team", "our company", "we've ", "we'd "])
    return has_decision and has_actor and len(text) > 100


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
        logger.info(f"  Existing: {len(existing_records):,} records, {len(existing_urls):,} unique URLs")

    new_records: list[dict] = []
    session = requests.Session()
    session.headers["User-Agent"] = "SaaS-Intel-Research/1.0 (academic data collection)"

    # ── Phase 1: Algolia search for decision-narrative stories ───────────
    logger.info("=== Phase 1: Algolia keyword search ===")
    all_story_ids: set[int] = set()
    story_metadata: dict[int, dict] = {}

    for keyword in DECISION_KEYWORDS[:20]:  # limit to top 20 queries
        logger.info(f"  Searching: '{keyword}'")
        hits = search_hn_algolia(keyword, content_type="story", max_hits=200)
        for h in hits:
            sid = int(h.get("objectID", 0) or 0)
            if sid and sid not in all_story_ids:
                all_story_ids.add(sid)
                story_metadata[sid] = {
                    "title": h.get("title", ""),
                    "url": h.get("url", ""),
                    "points": h.get("points", 0),
                    "num_comments": h.get("num_comments", 0),
                    "created_at": h.get("created_at", ""),
                }
        time.sleep(0.2)

    logger.info(f"Found {len(all_story_ids)} unique stories via search")

    # ── Phase 2: Also search for comments directly ───────────────────────
    logger.info("=== Phase 2: Algolia comment search ===")
    comment_records: list[dict] = []

    for keyword in DECISION_KEYWORDS[:15]:
        hits = search_hn_algolia(keyword, content_type="comment", max_hits=500)
        for h in hits:
            text = html_to_text(h.get("comment_text", "") or "")
            if not text or len(text) < 100:
                continue
            if not passes_prefilter(text):
                continue
            url = f"https://news.ycombinator.com/item?id={h.get('objectID','')}"
            if url in existing_urls:
                continue
            comment_records.append({
                "url": url,
                "text": text[:8000],
                "crawl_date": h.get("created_at", "")[:10],
                "segment_id": "hn_algolia_comments",
                "domain": "news.ycombinator.com",
                "source_type": "community_discussion",
            })
        time.sleep(0.2)

    logger.info(f"Found {len(comment_records):,} direct comment candidates from Algolia")
    new_records.extend(comment_records)

    # ── Phase 3: Fetch story + comments for top stories ─────────────────
    logger.info("=== Phase 3: Fetching story comments ===")

    # Sort by engagement (points × comments)
    sorted_stories = sorted(
        all_story_ids,
        key=lambda sid: story_metadata.get(sid, {}).get("points", 0) *
                        (story_metadata.get(sid, {}).get("num_comments", 0) + 1),
        reverse=True
    )

    # Take top 300 stories (most engaged = highest quality)
    top_stories = sorted_stories[:300]
    logger.info(f"Processing top {len(top_stories)} stories by engagement")

    processed = 0
    for sid in top_stories:
        meta = story_metadata.get(sid, {})
        # Add story text itself if it has a URL
        story_url = f"https://news.ycombinator.com/item?id={sid}"
        if story_url in existing_urls:
            processed += 1
            continue

        # Fetch the story item
        item = fetch_hn_item(sid, session)
        if not item:
            processed += 1
            continue

        # Add story body (Ask HN posts have text)
        story_text = html_to_text(item.get("text", "") or "")
        story_title = item.get("title", "")

        # Combine title + text for the story record
        full_story = f"{story_title}\n\n{story_text}".strip()
        if full_story and len(full_story) > 50:
            record = {
                "url": story_url,
                "text": full_story[:8000],
                "crawl_date": time.strftime("%Y-%m-%d", time.gmtime(item.get("time", 0))),
                "segment_id": "hn_firebase_stories",
                "domain": "news.ycombinator.com",
                "source_type": "community_discussion",
            }
            new_records.append(record)
            existing_urls.add(story_url)

        # Fetch and process comments
        comments = get_all_comments(item, session, depth=0)
        for comment_html in comments:
            text = html_to_text(comment_html)
            if not text or len(text) < 100:
                continue
            if not passes_prefilter(text):
                continue
            # We don't have unique IDs for individual comments easily here,
            # use a content-based key
            import hashlib
            comment_url = f"https://news.ycombinator.com/item?id={sid}#c{hashlib.md5(text[:50].encode()).hexdigest()[:8]}"
            if comment_url in existing_urls:
                continue
            new_records.append({
                "url": comment_url,
                "text": text[:8000],
                "crawl_date": time.strftime("%Y-%m-%d", time.gmtime(item.get("time", 0))),
                "segment_id": "hn_firebase_comments",
                "domain": "news.ycombinator.com",
                "source_type": "community_discussion",
            })
            existing_urls.add(comment_url)

        processed += 1
        if processed % 50 == 0:
            logger.info(f"  Processed {processed}/{len(top_stories)} stories — {len(new_records):,} new records so far")
        time.sleep(0.05)  # be respectful

    logger.info(f"\nHN ingestion complete: {len(new_records):,} new records")

    # ── Merge and save ────────────────────────────────────────────────────
    if not new_records:
        logger.info("No new records to add.")
        return 0

    all_records = existing_records + new_records
    logger.info(f"Saving {len(all_records):,} total records to {existing_path} ...")
    with gzip.open(existing_path, "wt", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False)
    logger.info(f"Saved ({existing_path.stat().st_size / 1_048_576:.1f} MB)")

    return len(new_records)


if __name__ == "__main__":
    count = main()
    logger.info(f"Done: {count:,} new HN records added")
    sys.exit(0 if count >= 0 else 1)
