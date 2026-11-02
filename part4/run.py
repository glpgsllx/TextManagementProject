"""Run the comparison detector over the corpus.

    python -m part4.run [--input-dir DIR] [--output-dir DIR] [--limit N]

Writes comparisons.csv (sentences that name or point at another business),
superlatives.csv (implicit comparisons with a scope) and summary.json (counts by
star rating, year, cue family and comparison target).
"""

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

from .data import load, review_rows
from .detect import BusinessComparisonDetector
from .sentences import split_sentences

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ("review_id", "business", "year", "stars", "kind", "families", "targets", "comparator", "sentence")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _rate(rows, sentences_by_key, key):
    found = Counter(r[key] for r in rows)
    table = {}
    for value, total in sorted(sentences_by_key.items()):
        count = found.get(value, 0)
        table[str(value)] = {"comparisons": count, "sentences": total,
                             "per_1000": round(1000 * count / total, 2) if total else 0.0}
    return table


def summarize(rows, sentences_star, sentences_year, cue_star, reviews):
    comparisons = [r for r in rows if r["kind"] == "business"]
    superlatives = [r for r in rows if r["kind"] == "superlative"]
    families = Counter(f for r in comparisons for f in r["families"].split("|"))
    targets = Counter(t for r in comparisons for t in r["targets"].split("|") if t)
    return {
        "reviews": len(reviews),
        "sentences": sum(sentences_star.values()),
        "cue_sentences": sum(cue_star.values()),
        "business_comparisons": len(comparisons),
        "superlatives": len(superlatives),
        "business_per_1000_sentences": round(1000 * len(comparisons) / sum(sentences_star.values()), 2),
        "by_star": _rate(comparisons, sentences_star, "stars"),
        "by_year": _rate(comparisons, sentences_year, "year"),
        "top_families": dict(families.most_common()),
        "top_targets": dict(targets.most_common(15)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/meridian_id")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "part4/results")
    parser.add_argument("--limit", type=int, default=0, help="only the first N reviews (0 = all)")
    args = parser.parse_args()

    started = time.time()
    reviews = review_rows(args.input_dir)
    if args.limit:
        reviews = reviews[:args.limit]
    _, _, names = load(args.input_dir)
    detector = BusinessComparisonDetector(list(names.values()))

    rows = []
    sentences_star, sentences_year, cue_star = Counter(), Counter(), Counter()
    for review in reviews:
        for sentence in split_sentences(review["text"]):
            sentences_star[review["stars"]] += 1
            sentences_year[review["year"]] += 1
            result = detector.analyze(sentence, review["business"])
            if result is None:
                continue
            cue_star[review["stars"]] += 1
            if result.kind == "other":
                continue
            rows.append({
                "review_id": review["review_id"],
                "business": review["business"],
                "year": review["year"],
                "stars": review["stars"],
                "kind": result.kind,
                "families": "|".join(result.families),
                "targets": "|".join(result.targets),
                "comparator": result.comparator,
                "sentence": result.sentence,
            })

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "comparisons.csv", [r for r in rows if r["kind"] == "business"])
    write_csv(output / "superlatives.csv", [r for r in rows if r["kind"] == "superlative"])
    summary = summarize(rows, sentences_star, sentences_year, cue_star, reviews)
    summary["seconds"] = round(time.time() - started, 1)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("reviews", "sentences", "business_comparisons",
                                              "superlatives", "seconds")}, indent=2))


if __name__ == "__main__":
    main()
