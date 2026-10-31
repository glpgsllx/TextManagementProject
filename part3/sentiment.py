"""VADER scoring and sector aggregates; figures stay in plots.py."""

from collections import defaultdict

import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Thresholds recommended by the VADER authors.
POSITIVE = 0.05
NEGATIVE = -0.05


def score_reviews(reviews):
    """Add compound in place. VADER needs raw case and punctuation, so no tokenizer."""
    analyzer = SentimentIntensityAnalyzer()
    for review in reviews:
        review["compound"] = analyzer.polarity_scores(review["text"])["compound"]


def polarity(compound):
    if compound >= POSITIVE:
        return "positive"
    if compound <= NEGATIVE:
        return "negative"
    return "neutral"


def select_sectors(reviews, sectors, start_year, end_year):
    """Tag in-range reviews with one sector; a business in several compared sectors is excluded."""
    wanted = set(sectors)
    selected = []
    for review in reviews:
        matches = review["sectors"] & wanted
        if len(matches) == 1 and start_year <= review["year"] <= end_year:
            selected.append({**review, "sector": next(iter(matches))})
    return selected


def excluded_businesses(reviews, sectors):
    """Names of businesses dropped because they belong to more than one compared sector."""
    return sorted({r["business"] for r in reviews if len(r["sectors"] & set(sectors)) > 1})


def bootstrap_ci(values, rng, resamples=2000, level=0.95):
    """Percentile bootstrap interval for the mean, resampling individual reviews."""
    values = np.asarray(values, dtype=float)
    means = values[rng.integers(0, len(values), (resamples, len(values)))].mean(axis=1)
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(means, [tail, 100 - tail])
    return float(low), float(high)


def summarize(reviews, rng, resamples=2000):
    """Summary row for one group, including a mean that weights each business equally."""
    scores = [r["compound"] for r in reviews]
    by_business = defaultdict(list)
    for review in reviews:
        by_business[review["business_id"]].append(review["compound"])
    low, high = bootstrap_ci(scores, rng, resamples)
    labels = [polarity(score) for score in scores]
    return {
        "reviews": len(reviews),
        "businesses": len(by_business),
        "mean_compound": round(float(np.mean(scores)), 4),
        "ci_low": round(low, 4),
        "ci_high": round(high, 4),
        "business_equal_mean": round(float(np.mean([np.mean(v) for v in by_business.values()])), 4),
        "positive_share": round(labels.count("positive") / len(labels), 4),
        "neutral_share": round(labels.count("neutral") / len(labels), 4),
        "negative_share": round(labels.count("negative") / len(labels), 4),
        "mean_stars": round(float(np.mean([r["stars"] for r in reviews])), 4),
    }


def group_summaries(reviews, key, rng, resamples=2000):
    """Summaries per sector and value of key ("year" or "season"), in stable order."""
    groups = defaultdict(list)
    for review in reviews:
        groups[review["sector"], review[key]].append(review)
    return [{"sector": sector, key: value, **summarize(groups[sector, value], rng, resamples)}
            for sector, value in sorted(groups)]


def same_business_change(reviews, early, late, min_reviews=5):
    """Compare only businesses reviewed in both periods, so new openings cannot move the mean."""
    rows = []
    for sector in sorted({r["sector"] for r in reviews}):
        before, after = defaultdict(list), defaultdict(list)
        for review in reviews:
            if review["sector"] == sector and review["year"] in early:
                before[review["business_id"]].append(review["compound"])
            elif review["sector"] == sector and review["year"] in late:
                after[review["business_id"]].append(review["compound"])
        panel = [b for b in before if len(before[b]) >= min_reviews and len(after[b]) >= min_reviews]
        first = [float(np.mean(before[b])) for b in panel]
        second = [float(np.mean(after[b])) for b in panel]
        rows.append({
            "sector": sector,
            "early_years": f"{min(early)}-{max(early)}",
            "late_years": f"{min(late)}-{max(late)}",
            "min_reviews_each_period": min_reviews,
            "businesses": len(panel),
            "early_mean": round(float(np.mean(first)), 4),
            "late_mean": round(float(np.mean(second)), 4),
            "businesses_declined": sum(y < x for x, y in zip(first, second)),
        })
    return rows


def cohort_summaries(reviews, cutoff_year):
    """Yearly means for businesses first seen by cutoff_year versus later entrants."""
    first_seen = {}
    for review in reviews:
        business = review["business_id"]
        first_seen[business] = min(first_seen.get(business, review["year"]), review["year"])
    groups = defaultdict(list)
    for review in reviews:
        cohort = "incumbent" if first_seen[review["business_id"]] <= cutoff_year else "entrant"
        groups[review["sector"], review["year"], cohort].append(review["compound"])
    return [{"sector": sector, "year": year, "cohort": cohort, "reviews": len(groups[sector, year, cohort]),
             "mean_compound": round(float(np.mean(groups[sector, year, cohort])), 4)}
            for sector, year, cohort in sorted(groups)]


def star_validation(reviews):
    """How VADER polarity lines up with the star rating, which it never sees."""
    rows = []
    for stars in sorted({r["stars"] for r in reviews}):
        scores = [r["compound"] for r in reviews if r["stars"] == stars]
        labels = [polarity(score) for score in scores]
        rows.append({
            "stars": stars,
            "reviews": len(scores),
            "mean_compound": round(float(np.mean(scores)), 4),
            "positive_share": round(labels.count("positive") / len(labels), 4),
            "neutral_share": round(labels.count("neutral") / len(labels), 4),
            "negative_share": round(labels.count("negative") / len(labels), 4),
        })
    return rows


def pearson(reviews):
    return float(np.corrcoef([r["compound"] for r in reviews], [r["stars"] for r in reviews])[0, 1])


def disagreements(reviews, limit=20, excerpt=400):
    """Largest star/lexicon conflicts, for manual error analysis in the report."""
    rows = []
    cases = [("one_star_scored_positive", [r for r in reviews if r["stars"] == 1 and r["compound"] >= POSITIVE], True),
             ("five_star_scored_negative", [r for r in reviews if r["stars"] == 5 and r["compound"] <= NEGATIVE], False)]
    for kind, matches, descending in cases:
        ranked = sorted(matches, key=lambda r: (r["compound"], r["review_id"]), reverse=descending)
        for review in ranked[:limit]:
            rows.append({"kind": kind, "review_id": review["review_id"], "business": review["business"],
                         "sector": review["sector"], "year": review["year"], "stars": review["stars"],
                         "compound": review["compound"], "text": " ".join(review["text"].split())[:excerpt]})
    return rows
