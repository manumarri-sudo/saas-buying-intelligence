"""
LLM result caching keyed by snippet hash.

Caches LLM classification results to avoid re-processing identical
or near-identical text snippets across incremental runs.

Storage: data/cache/llm_cache.json
Key: SHA-256 of normalized snippet text
"""

import hashlib
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class LLMCache:
    """Disk-backed cache for LLM classification results."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = cache_dir / "llm_cache.json"
        self._cache: dict[str, dict] = {}
        self._hits = 0
        self._misses = 0
        self._load()

    def _load(self):
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r") as f:
                    self._cache = json.load(f)
                logger.info(f"LLM cache loaded: {len(self._cache)} entries")
            except (json.JSONDecodeError, IOError):
                logger.warning("LLM cache corrupted, starting fresh")
                self._cache = {}

    def _save(self):
        with open(self.cache_path, "w") as f:
            json.dump(self._cache, f, ensure_ascii=False)

    @staticmethod
    def _hash_snippet(text: str) -> str:
        """Normalize and hash a text snippet."""
        normalized = " ".join(text.lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    def get(self, text: str) -> dict | None:
        """Look up a cached result. Returns None on miss."""
        key = self._hash_snippet(text)
        result = self._cache.get(key)
        if result is not None:
            self._hits += 1
            return result.get("data")
        self._misses += 1
        return None

    def put(self, text: str, result: dict):
        """Store a classification result."""
        key = self._hash_snippet(text)
        self._cache[key] = {
            "data": result,
            "cached_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def flush(self):
        """Write cache to disk."""
        self._save()
        logger.info(
            f"LLM cache flushed: {len(self._cache)} entries, "
            f"hits={self._hits}, misses={self._misses}"
        )

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "cache_size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / max(total, 1), 3),
        }
