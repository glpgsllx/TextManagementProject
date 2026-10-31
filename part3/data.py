"""Load the shared Meridian corpus and attach the fields Part 3 groups by."""

import hashlib
import json


SEASONS = ("Winter", "Spring", "Summer", "Autumn")


def season_of(month):
    """Northern-hemisphere meteorological season; Meridian, Idaho is at 43.6°N."""
    return SEASONS[month % 12 // 3]


def business_sectors(categories):
    """Split Yelp's comma-separated field into exact category labels."""
    return frozenset(c.strip() for c in (categories or "").split(",") if c.strip())


def read_verified(path, expected):
    """Parse one JSON Lines file after checking it against manifest.json."""
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if hashlib.sha256(raw).hexdigest() != expected["sha256"] or len(rows) != expected["records"]:
        raise ValueError(f"Corpus checksum/count differs from export manifest: {path.name}")
    return rows


def load_reviews(directory):
    """Return raw review text joined with sector, year and season."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    businesses = read_verified(directory / "business.json", manifest["files"]["business.json"])
    sectors = {b["business_id"]: business_sectors(b["categories"]) for b in businesses}
    names = {b["business_id"]: b["name"] for b in businesses}
    reviews = []
    for row in read_verified(directory / "review.json", manifest["files"]["review.json"]):
        reviews.append({
            "review_id": row["review_id"],
            "business_id": row["business_id"],
            "business": names[row["business_id"]],
            "sectors": sectors[row["business_id"]],
            "stars": row["stars"],
            "text": row["text"],
            "year": int(row["date"][:4]),
            "season": season_of(int(row["date"][5:7])),
        })
    return reviews
