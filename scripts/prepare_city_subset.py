"""Export a city-level Yelp subset, preserving original JSONL records.

Usage: python3 scripts/prepare_city_subset.py --city Meridian --state ID
Only the Python standard library is required. Existing output is never replaced.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile


ROOT = Path(__file__).resolve().parents[1]
BUSINESS_ID = re.compile(rb'"business_id"\s*:\s*"([^"\\]+)"')


def export_subset(source, destination, city, state):
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    source_business = source / "yelp_academic_dataset_business.json"
    source_review = source / "yelp_academic_dataset_review.json"
    for path in (source_business, source_review):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_stats = {
        p.name: {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
        for p in (source_business, source_review)
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    target_city = city.strip().casefold()
    target_state = state.strip().upper()
    business_ids = set()
    spellings = Counter()
    years = Counter()
    dates = []
    hashes = {"business.json": hashlib.sha256(), "review.json": hashlib.sha256()}
    counts = {"business.json": 0, "review.json": 0}
    scanned = {"business": 0, "review": 0}
    try:
        with source_business.open("rb") as incoming, (temporary / "business.json").open("wb") as outgoing:
            for line in incoming:
                scanned["business"] += 1
                row = json.loads(line)
                if (row.get("city") or "").strip().casefold() != target_city:
                    continue
                if (row.get("state") or "").strip().upper() != target_state:
                    continue
                identifier = row["business_id"]
                if identifier in business_ids:
                    raise ValueError(f"Duplicate selected business ID: {identifier}")
                business_ids.add(identifier)
                spellings[row["city"]] += 1
                outgoing.write(line)
                hashes["business.json"].update(line)
                counts["business.json"] += 1
        if not business_ids:
            raise ValueError(f"No businesses matched {city}, {state}")
        encoded_ids = {identifier.encode("utf-8") for identifier in business_ids}
        print(f"Matched {len(business_ids):,} businesses. Scanning reviews...", flush=True)
        with source_review.open("rb") as incoming, (temporary / "review.json").open("wb") as outgoing:
            for line in incoming:
                scanned["review"] += 1
                match = BUSINESS_ID.search(line, 0, 512)
                if match is None:
                    match = BUSINESS_ID.search(line)
                if match is None:
                    identifier = json.loads(line)["business_id"].encode("utf-8")
                else:
                    identifier = match.group(1)
                if identifier not in encoded_ids:
                    continue
                row = json.loads(line)
                if row["business_id"] not in business_ids:
                    raise ValueError("Review ID prefilter disagrees with parsed JSON")
                outgoing.write(line)
                hashes["review.json"].update(line)
                counts["review.json"] += 1
                years[row["date"][:4]] += 1
                dates.append(row["date"])
        for path in (source_business, source_review):
            now = {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            if now != source_stats[path.name]:
                raise RuntimeError(f"Source changed during export: {path}")
        manifest = {
            "city": city,
            "state": target_state,
            "format": "UTF-8 JSON Lines: one original JSON object per line",
            "filter": {
                "city": "city.strip().casefold() == requested_city.strip().casefold()",
                "state": "state.strip().upper() == requested_state.strip().upper()",
                "review": "business_id belongs to the selected business set",
                "additional_category_year_language_open_status_filters": None,
            },
            "source_directory": str(source.resolve()),
            "source_files": source_stats,
            "source_rows_scanned": scanned,
            "business_city_spellings": dict(sorted(spellings.items())),
            "review_date_min": min(dates) if dates else None,
            "review_date_max": max(dates) if dates else None,
            "reviews_by_year": dict(sorted(years.items())),
            "record_policy": "Original fields, text, city spellings and row order are preserved byte-for-byte; no tokenization or normalization of output records.",
            "files": {
                name: {
                    "records": count,
                    "bytes": (temporary / name).stat().st_size,
                    "sha256": hashes[name].hexdigest(),
                }
                for name, count in counts.items()
            },
            "reproduce": f'python3 scripts/prepare_city_subset.py --city "{city}" --state "{target_state}" --output-dir NEW_OUTPUT_DIRECTORY',
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.rename(destination)
        return manifest
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", default="Meridian")
    parser.add_argument("--state", default="ID")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "yelp-dataset/versions/4")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/meridian_id")
    args = parser.parse_args()
    result = export_subset(args.source_dir, args.output_dir, args.city, args.state)
    print(json.dumps({"output": str(args.output_dir.resolve()), "files": result["files"]}, indent=2))


if __name__ == "__main__":
    main()
