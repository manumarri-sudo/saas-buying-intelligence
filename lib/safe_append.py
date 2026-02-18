#!/usr/bin/env python3
"""
Thread-safe and process-safe append to ingested_records.json.gz.
Uses a file lock to prevent race conditions when multiple processes save.
"""
import gzip, json, fcntl, os, time
from pathlib import Path

RAW_PATH = Path(__file__).parent.parent / "data" / "raw" / "ingested_records.json.gz"
LOCK_PATH = RAW_PATH.with_suffix(".lock")

def safe_load():
    """Load records with shared lock (multiple readers OK)"""
    with open(LOCK_PATH, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_SH)
        with gzip.open(RAW_PATH, "rt", encoding="utf-8") as f:
            records = json.load(f)
        fcntl.flock(lock_f, fcntl.LOCK_UN)
    return records

def safe_append(new_records, verbose=True):
    """
    Atomically append new records to the file.
    Uses exclusive file lock — only one process can write at a time.
    Re-reads the file under lock to pick up any changes from concurrent processes.
    """
    LOCK_PATH.touch(exist_ok=True)
    
    with open(LOCK_PATH, "w") as lock_f:
        # Acquire exclusive lock (blocks until other writers release)
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            # Re-read under lock to get latest state
            with gzip.open(RAW_PATH, "rt", encoding="utf-8") as f:
                existing = json.load(f)
            
            existing_urls = {r.get("url","") for r in existing}
            
            # Deduplicate new records
            deduped = []
            for r in new_records:
                u = r.get("url","")
                if u and u not in existing_urls:
                    existing_urls.add(u)
                    deduped.append(r)
            
            if deduped:
                all_records = existing + deduped
                # Write atomically (temp file + rename)
                tmp_path = RAW_PATH.with_suffix(".json.gz.tmp")
                with gzip.open(tmp_path, "wt", encoding="utf-8") as f:
                    json.dump(all_records, f, ensure_ascii=False)
                os.rename(tmp_path, RAW_PATH)
                if verbose:
                    print(f"✓ Appended {len(deduped)} records (total: {len(all_records):,})", flush=True)
            else:
                if verbose:
                    print(f"No new unique records to append (0 added)", flush=True)
            
            return len(deduped)
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)
