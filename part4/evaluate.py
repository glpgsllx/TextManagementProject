"""Compare hand labels with the detector's predictions.

    python -m part4.evaluate [--labels part4/eval/labels.csv] [--keys part4/eval/keys.csv]

The positive class is "this sentence compares the reviewed business with another
business or place". A prediction counts as positive when the detector returned
``business`` or ``superlative``; ``other`` is a negative.
"""

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def wilson(k, n, z=1.96):
    """95% Wilson interval for a proportion; more stable than the normal one."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=ROOT / "part4/eval/labels.csv")
    parser.add_argument("--keys", type=Path, default=ROOT / "part4/eval/keys.csv")
    args = parser.parse_args()

    labels = {row["id"]: int(row["label"]) for row in csv.DictReader(args.labels.open(encoding="utf-8"))}
    keys = {row["id"]: row for row in csv.DictReader(args.keys.open(encoding="utf-8"))}
    ids = [i for i in labels if i in keys]

    tp = sum(1 for i in ids if labels[i] == 1 and keys[i]["kind"] != "other")
    fp = sum(1 for i in ids if labels[i] == 0 and keys[i]["kind"] != "other")
    fn = sum(1 for i in ids if labels[i] == 1 and keys[i]["kind"] == "other")
    tn = sum(1 for i in ids if labels[i] == 0 and keys[i]["kind"] == "other")

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    p_lo, p_hi = wilson(tp, tp + fp)
    r_lo, r_hi = wilson(tp, tp + fn)

    print(f"labelled {len(ids)}  positives {tp + fn}  predicted {tp + fp}")
    print(f"precision {precision:.3f}  95% CI [{p_lo:.3f}, {p_hi:.3f}]")
    print(f"recall    {recall:.3f}  95% CI [{r_lo:.3f}, {r_hi:.3f}]")
    print(f"F1        {f1:.3f}")
    print(f"confusion tp={tp} fp={fp} fn={fn} tn={tn}")

    by_family = {}
    for i in ids:
        family = keys[i]["families"].split("|")[0]
        hits = by_family.setdefault(family, [0, 0])
        hits[0] += 1
        hits[1] += labels[i] == 0 and keys[i]["kind"] != "other"
    print("false positives by first cue family:",
          {k: v[1] for k, v in sorted(by_family.items()) if v[1]})


if __name__ == "__main__":
    main()
