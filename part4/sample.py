"""Draw the hand-labelled evaluation samples.

    python -m part4.sample [--cue 120] [--random 40] [--seed 42]

Writes eval/candidates.csv and eval/random_audit.csv, which contain only the id
and the sentence, so the annotator does not see the prediction. The predictions
are kept in eval/keys.csv and joined back in part4.evaluate.
"""

import argparse
import csv
import random
from pathlib import Path

from .data import load, review_rows
from .detect import BusinessComparisonDetector
from .sentences import split_sentences

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/meridian_id")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "part4/eval")
    parser.add_argument("--cue", type=int, default=120)
    parser.add_argument("--random", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    reviews = review_rows(args.input_dir)
    _, _, names = load(args.input_dir)
    detector = BusinessComparisonDetector(list(names.values()))

    cue, everything = [], []
    for review in reviews:
        for sentence in split_sentences(review["text"]):
            result = detector.analyze(sentence, review["business"])
            row = {"sentence": sentence, "business": review["business"]}
            everything.append(row)
            if result is not None:
                cue.append({**row, "kind": result.kind,
                            "families": "|".join(result.families)})

    rng = random.Random(args.seed)
    picked_cue = rng.sample(cue, min(args.cue, len(cue)))
    picked_random = rng.sample(everything, min(args.random, len(everything)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "candidates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "sentence"])
        for i, row in enumerate(picked_cue):
            writer.writerow([i, row["sentence"].replace("\n", " ")])
    with (args.output_dir / "keys.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "kind", "families"])
        for i, row in enumerate(picked_cue):
            writer.writerow([i, row["kind"], row["families"]])
    with (args.output_dir / "random_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "sentence"])
        for i, row in enumerate(picked_random):
            writer.writerow([i, row["sentence"].replace("\n", " ")])

    print(f"{len(cue)} cue sentences of {len(everything)}; wrote {len(picked_cue)} candidates, "
          f"{len(picked_random)} random")


if __name__ == "__main__":
    main()
