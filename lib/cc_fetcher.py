"""
Common Crawl WET file fetcher.

Workflow:
  1. Query the CDX index API for URLs matching SaaS keywords.
  2. Identify which WET segment files contain those URLs.
  3. Download WET segments (gzipped WARC-style text records).
  4. Yield (url, text, metadata) tuples for downstream processing.

Only fetches pre-extracted text (WET) — never raw HTML.
"""

import gzip
import io
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Generator

import requests

logger = logging.getLogger(__name__)

CC_S3_BASE = "https://data.commoncrawl.org"


@dataclass
class WETRecord:
    url: str
    text: str
    crawl_date: str
    segment_id: str
    cc_index: str
    fetch_timestamp: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def query_cdx_index(
    cc_index: str,
    keywords: list[str],
    cdx_api_url: str,
    max_pages: int = 3,
) -> list[dict]:
    """
    Query Common Crawl CDX API for URLs matching keywords.
    Returns list of CDX records with WARC pointers.
    """
    results = []
    base = f"{cdx_api_url}/{cc_index}-index"

    for keyword in keywords:
        for page in range(max_pages):
            params = {
                "url": f"*.com/*{keyword.lower().replace(' ', '')}*",
                "output": "json",
                "page": page,
                "pageSize": 50,
                "filter": "=status:200",
            }
            try:
                resp = requests.get(base, params=params, timeout=30)
                if resp.status_code == 404:
                    logger.warning(f"CDX returned 404 for keyword={keyword} page={page}")
                    break
                resp.raise_for_status()

                for line in resp.text.strip().split("\n"):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        results.append(record)
                    except json.JSONDecodeError:
                        continue

            except requests.RequestException as e:
                logger.warning(f"CDX query failed: {e}")
                break

            time.sleep(0.5)  # rate limiting

    logger.info(f"CDX query returned {len(results)} records")
    return results


def download_wet_segment(
    filename: str,
    offset: int,
    length: int,
) -> bytes | None:
    """
    Download a byte range from a WET file on Common Crawl S3.
    Uses HTTP Range header to fetch only the relevant slice.
    """
    url = f"{CC_S3_BASE}/{filename}"
    headers = {"Range": f"bytes={offset}-{offset + length - 1}"}

    try:
        resp = requests.get(url, headers=headers, timeout=60)
        resp.raise_for_status()
        return resp.content
    except requests.RequestException as e:
        logger.warning(f"WET download failed for {filename}: {e}")
        return None


def parse_wet_records(
    raw_data: bytes,
    cc_index: str,
    segment_id: str,
    max_records: int = 2000,
) -> Generator[WETRecord, None, None]:
    """
    Parse WET text records from raw gzipped WARC data.
    Yields WETRecord instances.
    """
    try:
        decompressed = gzip.decompress(raw_data)
    except (gzip.BadGzipFile, OSError):
        # Data may already be decompressed or use a different format
        decompressed = raw_data

    text = decompressed.decode("utf-8", errors="replace")

    # WET records are separated by WARC headers
    records = text.split("WARC/1.0")
    count = 0

    for record in records:
        if count >= max_records:
            break

        # Extract URL from WARC-Target-URI header
        url = ""
        crawl_date = ""
        body = ""

        lines = record.split("\n")
        header_done = False
        body_lines = []

        for line in lines:
            if not header_done:
                if line.startswith("WARC-Target-URI:"):
                    url = line.split(":", 1)[1].strip()
                elif line.startswith("WARC-Date:"):
                    crawl_date = line.split(":", 1)[1].strip()
                elif line.strip() == "":
                    header_done = True
            else:
                body_lines.append(line)

        body = "\n".join(body_lines).strip()

        if url and body and len(body) > 50:
            count += 1
            yield WETRecord(
                url=url,
                text=body,
                crawl_date=crawl_date,
                segment_id=segment_id,
                cc_index=cc_index,
                fetch_timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )


def fetch_wet_paths(cc_index: str) -> list[str]:
    """
    Fetch the list of all WET file paths for a given crawl index.
    Returns a list of S3 paths.
    """
    url = f"{CC_S3_BASE}/crawl-data/{cc_index}/wet.paths.gz"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        decompressed = gzip.decompress(resp.content).decode("utf-8")
        paths = [p.strip() for p in decompressed.strip().split("\n") if p.strip()]
        logger.info(f"Found {len(paths)} WET paths for {cc_index}")
        return paths
    except requests.RequestException as e:
        logger.error(f"Failed to fetch WET paths: {e}")
        return []


def download_full_wet_file(
    wet_path: str,
    output_path: Path,
    max_bytes: int = 50_000_000,  # 50MB cap per file
) -> bool:
    """
    Download a full WET file (gzipped). Streams to disk.
    Caps download at max_bytes to control costs.
    """
    url = f"{CC_S3_BASE}/{wet_path}"
    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded = 0
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1_048_576):  # 1MB
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded >= max_bytes:
                    logger.warning(f"Hit max_bytes cap ({max_bytes}) for {wet_path}")
                    break

        logger.info(f"Downloaded {downloaded:,} bytes to {output_path}")
        return True

    except requests.RequestException as e:
        logger.error(f"Failed to download {wet_path}: {e}")
        return False
