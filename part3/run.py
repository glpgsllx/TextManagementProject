"""Run Part 3 end to end: python -m part3.run (from repository root)."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .cooccurrence import (
    analyze_term, merge_periods, theme_rates, tokenize_by_year, tracked_words, window_sensitivity,
)
from .data import load_reviews
from .plots import make_drift_figures, make_figures
from .sentiment import (
    cohort_summaries, disagreements, excluded_businesses, group_summaries, pearson,
    same_business_change, score_reviews, select_sectors, star_validation,
)


ROOT = Path(__file__).resolve().parents[1]
EARLY_YEARS = (2015, 2016)
LATE_YEARS = (2020, 2021)
INCUMBENT_CUTOFF = 2015
PMI_SECTOR = "Restaurants"
PMI_EXCLUDE = {"service": ("customer",)}  # "customer service" is a fixed phrase, not a drifting association.
PERIODS = {"2014-2016": (2014, 2016), "2017-2019": (2017, 2019), "2020-2021": (2020, 2021)}
TRACKED = {"service": ("slow", "prompt", "friendly", "terrible", "excellent", "impeccable", "delivery", "covid"),
           "price": ("reasonable", "fair", "worth", "quality", "portions", "expensive", "paid")}
THEMES = {
    "Speed": frozenset("slow fast quick prompt efficient wait waiting".split()),
    "Staff manner": frozenset("friendly attentive polite helpful rude courteous".split()),
    "Praise": frozenset("excellent great outstanding exceptional amazing awesome fantastic wonderful stellar impeccable".split()),
    "Complaint": frozenset("poor terrible horrible bad awful worst lousy mediocre".split()),
}


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/meridian_id")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "part3/results")
    parser.add_argument("--sectors", nargs=2, default=["Italian", "Fast Food"], metavar="SECTOR",
                        help="exact Yelp category labels to compare")
    parser.add_argument("--start-year", type=int, default=2014)
    parser.add_argument("--end-year", type=int, default=2021)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--terms", nargs=2, default=["service", "price"], metavar="TERM",
                        help="aspect terms for PMI drift; the first also gets the theme chart")
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--min-businesses", type=int, default=3)
    parser.add_argument("--drift-resamples", type=int, default=50)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    reviews = load_reviews(args.input_dir)
    score_reviews(reviews)
    selected = select_sectors(reviews, args.sectors, args.start_year, args.end_year)
    if {r["sector"] for r in selected} != set(args.sectors):
        raise ValueError(f"No reviews for at least one of {args.sectors} in the selected years")
    rng = np.random.default_rng(args.seed)
    yearly = group_summaries(selected, "year", rng, args.bootstrap)
    seasonal = group_summaries(selected, "season", rng, args.bootstrap)
    validation = star_validation(reviews)

    write_csv(output / "sentiment_by_year.csv", yearly)
    write_csv(output / "sentiment_by_season.csv", seasonal)
    write_csv(output / "sentiment_same_business.csv", same_business_change(selected, EARLY_YEARS, LATE_YEARS))
    write_csv(output / "sentiment_cohorts.csv", cohort_summaries(selected, INCUMBENT_CUTOFF))
    write_csv(output / "vader_star_validation.csv", validation)
    write_csv(output / "vader_disagreements.csv", disagreements(selected))
    make_figures(output, yearly, seasonal, args.sectors)

    restaurants = [r for r in reviews if PMI_SECTOR in r["sectors"] and args.start_year <= r["year"] <= args.end_year]
    by_year = tokenize_by_year(restaurants)
    by_period = merge_periods(by_year, PERIODS)
    settings = (args.window, args.top_n, args.min_count, args.min_businesses)
    top_rows, drift_rows, period_top_rows, period_drift_rows, tracked_rows, window_rows = [], [], [], [], [], []
    for term in args.terms:
        exclude = PMI_EXCLUDE.get(term, ())
        top, drift, rankings = analyze_term(by_year, term, *settings, rng, args.drift_resamples, exclude)
        top_rows += top
        drift_rows += drift
        tracked_rows += tracked_words(rankings, term, TRACKED.get(term, ()))
        # Pooled periods have about three times the text, so the count filter doubles.
        top, drift, _ = analyze_term(by_period, term, args.window, args.top_n, args.min_count * 2, args.min_businesses,
                                     rng, args.drift_resamples, exclude)
        period_top_rows += top
        period_drift_rows += drift
        window_rows += window_sensitivity(by_year, term, (3, args.window, 10), args.window, args.top_n,
                                          args.min_count, args.min_businesses, exclude)
    themes = theme_rates(by_year, args.terms[0], THEMES, args.window, rng, args.bootstrap)
    write_csv(output / "pmi_top_words_by_year.csv", top_rows)
    write_csv(output / "pmi_drift_by_year.csv", drift_rows)
    write_csv(output / "pmi_top_words_by_period.csv", period_top_rows)
    write_csv(output / "pmi_drift_by_period.csv", period_drift_rows)
    write_csv(output / "pmi_tracked_words.csv", tracked_rows)
    write_csv(output / "pmi_window_sensitivity.csv", window_rows)
    write_csv(output / "pmi_themes.csv", themes)
    make_drift_figures(output, drift_rows, themes, args.terms)
    print(json.dumps({"reviews_scored": len(reviews), "reviews_selected": len(selected),
                      "restaurant_reviews_for_pmi": len(restaurants),
                      "excluded_businesses": excluded_businesses(reviews, args.sectors),
                      "vader_star_pearson": round(pearson(reviews), 4),
                      "output": str(output.relative_to(ROOT)) if output.is_relative_to(ROOT) else str(output)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
