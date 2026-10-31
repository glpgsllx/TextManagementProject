"""Build the inverted index and print its statistics: python -m part2.run (from the repo root).

The index is built once and pickled to `part2/.cache/index.pkl`; every later run loads
it straight back instead of re-tokenizing. If the corpus or the tokenizer changes,
loading fails, so rebuild with --rebuild.

Optional arguments:
    --rebuild        ignore the index on disk, rebuild it and overwrite
    --index-file DIR where to store the index (default part2/.cache/index.pkl)
    --limit N        take only the first N reviews, for a quick trial run;
                     leave it out for the full 37,401
    --tokenizer NAME custom (default) / whitespace / wordpunct
    --output-dir DIR where to write the statistics

A full run also cross-checks against part1's custom-tokenizer counts: document count,
total tokens and vocabulary size all have to agree, which is what shows that this index
and part1 counted the same set of tokens.
"""

import argparse
import csv
import json
import time
from pathlib import Path

from .inverted_index import build_inverted_index, load_index, save_index

ROOT = Path(__file__).resolve().parents[1]


def read_reviews(path, limit=None):
    """Read review.json line by line (JSON Lines, one review per line), keeping only
    review_id and text."""
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            review = json.loads(line)
            yield {"review_id": review["review_id"], "text": review["text"]}
            count += 1
            if limit is not None and count >= limit:
                return


def index_statistics(index, seconds, from_disk):
    """Assemble the index dimensions into one table; the §3.3.1 statistics table in the
    report is produced from exactly these numbers."""
    lengths = sorted(index.doc_lengths.values())
    ranked = sorted(index.postings.items(), key=lambda item: (-len(item[1]), item[0]))
    return {
        "tokenizer": index.tokenizer,
        "index_source": "loaded from disk" if from_disk else "built from corpus",
        "index_seconds": round(seconds, 1),
        "reviews": len(index.doc_lengths),
        "terms": len(index.postings),
        "total_tokens": index.total_tokens,
        "avgdl": round(index.avg_doc_length, 4),
        "postings": sum(len(plist) for plist in index.postings.values()),
        "singleton_terms": sum(1 for plist in index.postings.values() if len(plist) == 1),
        "dl_min": lengths[0],
        "dl_median": lengths[len(lengths) // 2],
        "dl_max": lengths[-1],
        "top_df_terms": [{"term": term, "df": len(plist)} for term, plist in ranked[:5]],
    }


def cross_check_part1(index, comparison_csv):
    """Reconcile against the row for the same tokenizer in
    part1/results/tokenizer_comparison.csv."""
    with comparison_csv.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["method"] != index.tokenizer:
                continue
            part1 = (int(row["reviews"]), int(row["normalized_tokens"]),
                     int(row["normalized_types"]))
            ours = (len(index.doc_lengths), index.total_tokens, len(index.postings))
            return {
                "part1": {"reviews": part1[0], "tokens": part1[1], "types": part1[2]},
                "index": {"reviews": ours[0], "tokens": ours[1], "types": ours[2]},
                "matches": part1 == ours,
            }
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/meridian_id")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "part2/results")
    parser.add_argument("--index-file", type=Path, default=ROOT / "part2/.cache/index.pkl")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--tokenizer", default="custom")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    manifest = json.loads((args.input_dir / "manifest.json").read_text(encoding="utf-8"))
    reviews_path = args.input_dir / "review.json"
    source = {"file": reviews_path.name, "bytes": reviews_path.stat().st_size,
              "reviews": manifest["files"]["review.json"]["records"],
              "tokenizer": args.tokenizer}

    # --limit takes only the first N reviews: a partial index meant for trial runs,
    # so it is neither saved nor loaded.
    from_disk = args.limit is None and not args.rebuild and args.index_file.exists()
    started = time.monotonic()
    if from_disk:
        index = load_index(args.index_file, source)
    else:
        index = build_inverted_index(read_reviews(reviews_path, args.limit),
                                     args.tokenizer)
    statistics = index_statistics(index, time.monotonic() - started, from_disk)
    if args.limit is None and not from_disk:
        statistics["index_file_mb"] = round(save_index(index, args.index_file, source) / 1e6, 1)
    elif from_disk:
        statistics["index_file_mb"] = round(args.index_file.stat().st_size / 1e6, 1)

    expected = source["reviews"]
    statistics["manifest_reviews"] = expected
    statistics["manifest_matches"] = expected == statistics["reviews"]

    comparison = ROOT / "part1/results/tokenizer_comparison.csv"
    if args.limit is None and comparison.exists():
        statistics["cross_check_part1"] = cross_check_part1(index, comparison)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(statistics, ensure_ascii=False, indent=2) + "\n"
    (args.output_dir / "index_statistics.json").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
