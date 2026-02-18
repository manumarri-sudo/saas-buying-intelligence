#!/usr/bin/env python3
"""
Common Crawl targeted extraction for review platforms with high decision-narrative density.

Targets (in priority order):
  1. TrustRadius — /products/* pages with "Alternatives Considered" field
  2. Clutch.co — /profile/* company case studies (The Challenge / Solution narrative)
  3. dev.to — /*/article/* articles with migration/switching titles (CC supplement)
  4. saastr.com — SaaS content with procurement narratives
  5. infoq.com — Engineering decision write-ups
  6. openviewpartners.com — SaaS buying intelligence content

Uses CDX API to get WARC record offsets, then fetches targeted byte ranges from S3.
Falls back to HTTP if S3 is unavailable (slower but works without AWS credentials).

Key insight: TrustRadius "Alternatives Considered" field contains the highest-density
decision narratives of any public web source. We extract specifically that field.
"""

import gzip
import hashlib
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from io import BytesIO

import requests

try:
    from warcio.archiveiterator import ArchiveIterator
    HAS_WARCIO = True
except ImportError:
    HAS_WARCIO = False
    logger_init = logging.getLogger("ingest_cc_reviews")
    logger_init.warning("warcio not available — will use direct HTTP fetch fallback")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.config_loader import resolve_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingest_cc_reviews")

# ── Config ────────────────────────────────────────────────────────────────

CC_INDEXES = [
    "CC-MAIN-2024-46",
    "CC-MAIN-2024-42",
    "CC-MAIN-2024-38",
    "CC-MAIN-2024-33",
]

CDX_BASE = "https://index.commoncrawl.org"
CC_S3_BASE = "https://data.commoncrawl.org"

HEADERS = {
    "User-Agent": "SaaS-Intel-Research/1.0 (academic; Common Crawl data user)",
    "Accept": "*/*",
}

# Target URL patterns — (url_pattern, parser_key, max_records)
TARGETS = [
    # TrustRadius product review pages
    ("www.trustradius.com/products/*", "trustradius", 2000),
    # Clutch company profile pages (case studies)
    ("clutch.co/profile/*", "clutch", 2000),
    # saastr.com articles
    ("www.saastr.com/*", "generic_blog", 500),
    # First Round Capital Review
    ("review.firstround.com/*", "generic_blog", 500),
    # OpenView Partners
    ("openviewpartners.com/*", "generic_blog", 300),
    # InfoQ articles
    ("www.infoq.com/articles/*", "generic_blog", 500),
    # Software Engineering Daily
    ("softwareengineeringdaily.com/*", "generic_blog", 300),
    # dzone articles
    ("dzone.com/articles/*", "generic_blog", 500),
    # hashnode.dev tech blogs
    ("*.hashnode.dev/*", "generic_blog", 500),
]

MAX_WORKERS = 8


def query_cdx(url_pattern: str, crawl_index: str,
              session: requests.Session, max_results: int = 2000) -> list[dict]:
    """Query CDX API for URLs matching pattern."""
    results = []
    page = 0
    while len(results) < max_results:
        params = {
            "url": url_pattern,
            "output": "json",
            "fl": "timestamp,filename,offset,length,statuscode,url",
            "filter": "statuscode:200",
            "limit": min(500, max_results - len(results)),
            "page": page,
        }
        try:
            r = session.get(f"{CDX_BASE}/{crawl_index}-index", params=params, timeout=20)
            if r.status_code == 404:
                break
            r.raise_for_status()
            lines = [l for l in r.text.strip().split("\n") if l and "{" in l]
            if not lines:
                break
            for line in lines:
                try:
                    record = json.loads(line)
                    results.append(record)
                except Exception:
                    pass
            if len(lines) < 500:
                break
            page += 1
        except Exception as e:
            logger.debug(f"CDX error ({url_pattern}): {e}")
            break
        time.sleep(0.2)
    return results


def fetch_warc_content(record: dict, session: requests.Session) -> bytes | None:
    """Fetch a specific byte range from a Common Crawl WARC file."""
    filename = record.get("filename", "")
    offset = int(record.get("offset", 0))
    length = int(record.get("length", 0))

    if not filename or not length:
        return None

    url = f"{CC_S3_BASE}/{filename}"
    byte_range = f"bytes={offset}-{offset + length - 1}"

    try:
        r = session.get(url, headers={"Range": byte_range}, timeout=30)
        r.raise_for_status()
        return r.content
    except Exception as e:
        logger.debug(f"WARC fetch error: {e}")
        return None


def extract_text_from_warc_bytes(content: bytes) -> str | None:
    """Extract text content from WARC response bytes."""
    if not content:
        return None

    if HAS_WARCIO:
        try:
            for record in ArchiveIterator(BytesIO(content)):
                if record.rec_type == "response":
                    payload = record.content_stream().read()
                    # Find body after HTTP headers
                    sep = payload.find(b"\r\n\r\n")
                    if sep == -1:
                        sep = payload.find(b"\n\n")
                    if sep != -1:
                        html = payload[sep + 4:]
                    else:
                        html = payload
                    return html_to_text(html.decode("utf-8", errors="replace"))
        except Exception:
            pass

    # Fallback: try to find HTML directly
    try:
        text = content.decode("utf-8", errors="replace")
        sep = text.find("\r\n\r\n")
        if sep == -1:
            sep = text.find("\n\n")
        if sep != -1:
            html = text[sep + 4:]
        else:
            html = text
        return html_to_text(html)
    except Exception:
        return None


def html_to_text(html: str) -> str:
    """Extract readable text from HTML."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        # Remove script, style, nav, footer
        for tag in soup.find_all(["script", "style", "nav", "footer",
                                   "header", "aside", "form"]):
            tag.decompose()
        return soup.get_text(" ", strip=True)
    except Exception:
        # Regex fallback
        clean = re.sub(r"<script[^>]*>.*?</script>", " ", html,
                       flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r"<style[^>]*>.*?</style>", " ", clean,
                       flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r"<[^>]+>", " ", clean)
        return re.sub(r"\s+", " ", clean).strip()


def parse_trustradius(html: str) -> list[str]:
    """
    Extract decision-narrative-rich text blocks from TrustRadius pages.
    TrustRadius has these key fields:
      - Review body (long-form)
      - "Alternatives Considered" — explicitly lists competitors evaluated
      - "Return on Investment" — sometimes mentions why they switched
    """
    snippets = []
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")

        # Find review content blocks
        # TrustRadius uses various class patterns across versions
        for block in soup.find_all(class_=re.compile(
            r"review|alternative|considered|competitor|switching|reason", re.I
        )):
            text = block.get_text(" ", strip=True)
            if len(text) > 100:
                snippets.append(text)

        # Also look for section headers followed by decision-narrative content
        for header in soup.find_all(["h2", "h3", "h4", "strong", "b"]):
            header_text = header.get_text(strip=True).lower()
            if any(kw in header_text for kw in [
                "alternative", "consider", "switched", "replaced",
                "reason", "previous", "competitor", "instead"
            ]):
                # Get the following sibling content
                sibling = header.find_next_sibling()
                if sibling:
                    text = sibling.get_text(" ", strip=True)
                    if len(text) > 50:
                        snippets.append(text)

        if not snippets:
            # Fall back to full page text
            for tag in soup.find_all(["script", "style", "nav", "footer"]):
                tag.decompose()
            full = soup.get_text(" ", strip=True)
            if len(full) > 200:
                snippets.append(full)

    except Exception:
        # Just use html_to_text result
        text = html_to_text(html)
        if text:
            snippets.append(text)

    return snippets


def parse_clutch(html: str) -> list[str]:
    """
    Extract case study narratives from Clutch.co profile pages.
    Clutch case studies have sections:
      - "The Challenge" — what problem they needed to solve
      - "The Solution" — what vendor they chose and why
      - "The Results" — outcomes
    These combine to form excellent decision narratives.
    """
    snippets = []
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")

        # Look for challenge/solution/results sections
        for header in soup.find_all(["h2", "h3", "h4", "strong", "b"]):
            header_text = header.get_text(strip=True).lower()
            if any(kw in header_text for kw in [
                "challenge", "solution", "result", "project", "background",
                "approach", "outcome", "summary", "overview"
            ]):
                # Get text from this section until next similar header
                content_parts = []
                sibling = header.find_next_sibling()
                for _ in range(5):  # max 5 siblings per section
                    if not sibling:
                        break
                    tag_name = sibling.name
                    if tag_name in ["h2", "h3", "h4"] or (
                        tag_name in ["strong", "b"] and
                        any(kw in sibling.get_text(strip=True).lower() for kw in
                            ["challenge", "solution", "result", "project"])
                    ):
                        break
                    text = sibling.get_text(" ", strip=True)
                    if text:
                        content_parts.append(text)
                    sibling = sibling.find_next_sibling()

                if content_parts:
                    section_text = f"{header.get_text(strip=True)}: {' '.join(content_parts)}"
                    if len(section_text) > 100:
                        snippets.append(section_text)

        if not snippets:
            # Full page text as fallback
            for tag in soup.find_all(["script", "style", "nav", "footer"]):
                tag.decompose()
            full = soup.get_text(" ", strip=True)
            if len(full) > 200:
                snippets.append(full)

    except Exception:
        text = html_to_text(html)
        if text:
            snippets.append(text)

    return snippets


def parse_generic_blog(html: str) -> list[str]:
    """Extract main article content from generic blog pages."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        for tag in soup.find_all(["script", "style", "nav", "footer",
                                   "header", "aside", "form", "iframe"]):
            tag.decompose()

        # Try article/main content tags first
        main = soup.find("article") or soup.find("main") or soup.find(
            class_=re.compile(r"post|article|content|body|text", re.I)
        )
        if main:
            text = main.get_text(" ", strip=True)
        else:
            text = soup.get_text(" ", strip=True)

        return [text] if len(text) > 200 else []
    except Exception:
        text = html_to_text(html)
        return [text] if text and len(text) > 200 else []


PARSERS = {
    "trustradius": parse_trustradius,
    "clutch": parse_clutch,
    "generic_blog": parse_generic_blog,
}


def passes_prefilter(text: str) -> bool:
    """Check if text has decision narrative signals."""
    if not text or len(text) < 80:
        return False
    tl = text.lower()
    has_decision = any(kw in tl for kw in [
        "chose", "selected", "switched", "migrated", "evaluated",
        "replaced", "rejected", "moved from", "adopted", "piloted",
        "procured", "purchased", "trialed",
    ])
    has_actor = any(a in tl for a in [
        "we ", "our team", "our company", "our org", "we've ",
        "the team", "the company", "they selected", "they chose",
        "the client", "the organization",
    ])
    return has_decision and has_actor


def process_cdx_record(cdx_record: dict, parser_key: str,
                        session: requests.Session) -> list[dict]:
    """Fetch and parse a single CDX record into pipeline records."""
    content = fetch_warc_content(cdx_record, session)
    if not content:
        return []

    # Decompress if needed
    try:
        raw_html = gzip.decompress(content)
    except Exception:
        raw_html = content

    try:
        html_text = raw_html.decode("utf-8", errors="replace")
    except Exception:
        return []

    parser = PARSERS.get(parser_key, parse_generic_blog)
    snippets = parser(html_text)

    records = []
    source_url = cdx_record.get("url", "")
    crawl_date = cdx_record.get("timestamp", "")[:8]
    if len(crawl_date) == 8:
        crawl_date = f"{crawl_date[:4]}-{crawl_date[4:6]}-{crawl_date[6:8]}"

    for snippet in snippets:
        if not passes_prefilter(snippet):
            continue
        # Deduplicate snippets from same page using hash
        snippet_hash = hashlib.md5(snippet[:100].encode()).hexdigest()[:8]
        rec_url = f"{source_url}#snippet_{snippet_hash}" if len(snippets) > 1 else source_url
        records.append({
            "url": rec_url,
            "text": snippet[:8000],
            "crawl_date": crawl_date,
            "segment_id": f"cc_{parser_key}",
            "domain": source_url.split("/")[2] if "/" in source_url else "unknown",
            "source_type": "case_study" if parser_key in ("trustradius", "clutch") else "tech_blog",
        })

    return records


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

    for url_pattern, parser_key, max_recs in TARGETS:
        logger.info(f"\n=== Processing {url_pattern} (parser={parser_key}, max={max_recs}) ===")

        # Try multiple CC indexes
        all_cdx = []
        seen_urls: set[str] = set()
        for crawl_idx in CC_INDEXES:
            cdx_records = query_cdx(url_pattern, crawl_idx, session,
                                    max_results=max_recs)
            for r in cdx_records:
                u = r.get("url", "")
                if u and u not in seen_urls and u not in existing_urls:
                    seen_urls.add(u)
                    all_cdx.append(r)
            logger.info(f"  {crawl_idx}: {len(cdx_records)} CDX records, "
                        f"{len(all_cdx)} unique new so far")
            if len(all_cdx) >= max_recs:
                break
            time.sleep(0.5)

        logger.info(f"  Total unique CDX records to process: {len(all_cdx)}")

        # Process in parallel
        done = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(process_cdx_record, rec, parser_key, session): rec
                for rec in all_cdx[:max_recs]
            }
            for future in as_completed(futures):
                try:
                    recs = future.result()
                    for r in recs:
                        if r["url"] not in existing_urls:
                            new_records.append(r)
                            existing_urls.add(r["url"])
                except Exception as e:
                    logger.debug(f"Record processing error: {e}")
                done += 1
                if done % 100 == 0:
                    logger.info(f"  Processed {done}/{len(all_cdx)} — "
                                f"{len(new_records)} new records total")

        logger.info(f"  Done: {len(new_records)} total new records so far")

    logger.info(f"\nCC review ingestion complete: {len(new_records):,} new records")

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
    logger.info(f"Done: {count:,} new CC review records added")
    sys.exit(0 if count >= 0 else 1)
