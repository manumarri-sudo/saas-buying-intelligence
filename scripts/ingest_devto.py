#!/usr/bin/env python3
"""
dev.to ingestion — fetches B2B SaaS buying/migration articles via the public dev.to API.

dev.to API: https://dev.to/api/ — no auth required for read-only access.
Articles are public, API is officially supported by Forem (dev.to's platform).

Strategy:
  1. Search by tags: migration, devops, aws, kubernetes, productivity, tools, etc.
  2. Filter by title keywords: switched, migrated, chose, moved from, vs, etc.
  3. Fetch full article content for matching articles
  4. Also fetch comments on high-signal articles
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
logger = logging.getLogger("ingest_devto")

DEVTO_API = "https://dev.to/api"
HEADERS = {
    "User-Agent": "SaaS-Intel-Research/1.0 (academic)",
    "Accept": "application/json",
}

# Tags with high density of migration/switching content
TARGET_TAGS = [
    "migration",
    "devops",
    "aws",
    "kubernetes",
    "productivity",
    "tools",
    "architecture",
    "backend",
    "databases",
    "cloud",
    "infrastructure",
    "saas",
    "startup",
    "management",
    "agile",
    "engineering",
    "platform",
    "security",
    "monitoring",
    "ci",
]

# Title keywords that strongly indicate decision narratives
TITLE_SIGNAL_KEYWORDS = [
    "switched", "migrated", "migration", "chose", "selected",
    "moved from", "from.*to", "vs ", "compared", "why we",
    "how we", "replaced", "alternatives", "evaluation",
    "choosing", "decided", "abandoned", "deprecated",
    "away from", "instead of", "over.*because",
]

# Text pre-filter
DECISION_KEYWORDS = [
    "chose", "selected", "switched", "migrated", "evaluated",
    "replaced", "rejected", "moved from", "adopted", "piloted",
]
ACTOR_KEYWORDS = ["we ", "our team", "our company", "we've ", "we'd "]


def title_is_relevant(title: str) -> bool:
    """Check if article title suggests a decision narrative."""
    title_lower = title.lower()
    import re
    for kw in TITLE_SIGNAL_KEYWORDS:
        if re.search(kw, title_lower):
            return True
    return False


def passes_prefilter(text: str) -> bool:
    """Quick keyword check."""
    if not text or len(text) < 100:
        return False
    tl = text.lower()
    has_decision = any(kw in tl for kw in DECISION_KEYWORDS)
    has_actor = any(a in tl for a in ACTOR_KEYWORDS)
    return has_decision and has_actor


def fetch_articles_by_tag(tag: str, session: requests.Session,
                          max_pages: int = 10) -> list[dict]:
    """Fetch articles for a given tag, paginated."""
    articles = []
    for page in range(1, max_pages + 1):
        params = {"tag": tag, "per_page": 1000, "page": page, "state": "rising"}
        try:
            r = session.get(f"{DEVTO_API}/articles", params=params, timeout=15)
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            articles.extend(batch)
            if len(batch) < 1000:
                break
        except Exception as e:
            logger.warning(f"  devto tag={tag} page={page} error: {e}")
            break
        time.sleep(0.5)
    return articles


def fetch_article_content(article_id: int, session: requests.Session) -> str:
    """Fetch full article body text."""
    try:
        r = session.get(f"{DEVTO_API}/articles/{article_id}", timeout=15)
        r.raise_for_status()
        data = r.json()
        body = data.get("body_markdown", "") or data.get("body_html", "") or ""
        # Strip HTML if needed
        if "<" in body and ">" in body:
            try:
                from bs4 import BeautifulSoup
                body = BeautifulSoup(body, "lxml").get_text(" ", strip=True)
            except Exception:
                import re
                body = re.sub(r"<[^>]+>", " ", body).strip()
        return body
    except Exception:
        return ""


def fetch_article_comments(article_id: int, session: requests.Session) -> list[str]:
    """Fetch comments on an article."""
    try:
        r = session.get(f"{DEVTO_API}/comments", params={"a_id": article_id}, timeout=15)
        r.raise_for_status()
        comments = r.json()
        texts = []
        for c in comments:
            body = c.get("body_html", "") or ""
            if "<" in body:
                try:
                    from bs4 import BeautifulSoup
                    body = BeautifulSoup(body, "lxml").get_text(" ", strip=True)
                except Exception:
                    import re
                    body = re.sub(r"<[^>]+>", " ", body).strip()
            if body and len(body) > 50:
                texts.append(body)
        return texts
    except Exception:
        return []


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

    # ── Phase 1: Browse by tag ────────────────────────────────────────────
    logger.info("=== Phase 1: Browse articles by tag ===")
    seen_article_ids: set[int] = set()
    candidate_articles: list[dict] = []

    for tag in TARGET_TAGS:
        logger.info(f"  Tag: {tag}")
        articles = fetch_articles_by_tag(tag, session, max_pages=5)
        for a in articles:
            aid = a.get("id", 0)
            if aid and aid not in seen_article_ids:
                # Check if title is relevant
                title = a.get("title", "")
                if title_is_relevant(title):
                    seen_article_ids.add(aid)
                    candidate_articles.append(a)
        logger.info(f"    {len(candidate_articles)} candidates so far")

    # Also add all articles without title filter but with tag = migration
    logger.info("  Tag: migration (all articles, no title filter)")
    migration_articles = fetch_articles_by_tag("migration", session, max_pages=10)
    for a in migration_articles:
        aid = a.get("id", 0)
        if aid and aid not in seen_article_ids:
            seen_article_ids.add(aid)
            candidate_articles.append(a)

    logger.info(f"Total candidate articles: {len(candidate_articles)}")

    # ── Phase 2: Fetch full content for candidates ────────────────────────
    logger.info("=== Phase 2: Fetching full article content ===")
    fetched = 0

    for article in candidate_articles:
        aid = article.get("id", 0)
        url = article.get("url", f"https://dev.to/article/{aid}")
        if not url:
            url = f"https://dev.to/article/{aid}"

        if url in existing_urls:
            continue

        # Fetch full body
        body = fetch_article_content(aid, session)
        if not body:
            # Fall back to description
            body = article.get("description", "")

        title = article.get("title", "")
        full_text = f"{title}\n\n{body}".strip()

        if not passes_prefilter(full_text):
            # Even without body, add if title is very strong signal
            snippet = article.get("description", "")
            full_text = f"{title}\n\n{snippet}".strip()
            if not passes_prefilter(full_text) or len(full_text) < 100:
                time.sleep(0.3)
                continue

        # Build record
        pub_date = article.get("published_at", "") or article.get("created_at", "")
        record = {
            "url": url,
            "text": full_text[:8000],
            "crawl_date": pub_date[:10] if pub_date else "",
            "segment_id": "devto_api",
            "domain": "dev.to",
            "source_type": "tech_blog",
        }
        new_records.append(record)
        existing_urls.add(url)
        fetched += 1

        # Also grab comments for high-reaction articles
        reactions = article.get("positive_reactions_count", 0) or 0
        if reactions >= 50:
            comments = fetch_article_comments(aid, session)
            for comment_text in comments:
                if not passes_prefilter(comment_text):
                    continue
                import hashlib
                comment_url = f"{url}#comment_{hashlib.md5(comment_text[:50].encode()).hexdigest()[:8]}"
                if comment_url in existing_urls:
                    continue
                new_records.append({
                    "url": comment_url,
                    "text": comment_text[:8000],
                    "crawl_date": pub_date[:10] if pub_date else "",
                    "segment_id": "devto_comments",
                    "domain": "dev.to",
                    "source_type": "tech_blog",
                })
                existing_urls.add(comment_url)

        if fetched % 100 == 0:
            logger.info(f"  Fetched {fetched} articles — {len(new_records)} new records")
        time.sleep(0.4)  # dev.to rate limit is generous but be respectful

    logger.info(f"\ndev.to ingestion complete: {len(new_records):,} new records")

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
    logger.info(f"Done: {count:,} new dev.to records added")
    sys.exit(0 if count >= 0 else 1)
