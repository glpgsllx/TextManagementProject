"""Load the shared Meridian corpus for Part 4.

The loader repeats Part 3's manifest check on purpose: Part 4 must run on its
own (``python -m part4.run``) without importing another part. Line endings are
normalized to LF before hashing, so a Windows checkout, where Git rewrites the
JSON Lines files to CRLF, still passes the checksum.
"""

import hashlib
import json
from pathlib import Path


def _read_jsonl(path, expected):
    raw = path.read_bytes().replace(b"\r\n", b"\n")
    digest = hashlib.sha256(raw).hexdigest()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if digest != expected["sha256"] or len(rows) != expected["records"]:
        raise ValueError(f"Corpus checksum/count differs from export manifest: {path.name}")
    return rows


def load(directory):
    """Return (businesses, reviews, names) for one exported corpus directory."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    businesses = _read_jsonl(directory / "business.json", manifest["files"]["business.json"])
    reviews = _read_jsonl(directory / "review.json", manifest["files"]["review.json"])
    names = {b["business_id"]: b["name"] for b in businesses}
    return businesses, reviews, names


def review_rows(directory):
    """Reviews with the fields Part 4 needs: id, business, stars, year, text."""
    _, reviews, names = load(directory)
    rows = []
    for review in reviews:
        rows.append({
            "review_id": review["review_id"],
            "business_id": review["business_id"],
            "business": names[review["business_id"]],
            "stars": int(review["stars"]),
            "year": int(review["date"][:4]),
            "text": review["text"],
        })
    return rows
