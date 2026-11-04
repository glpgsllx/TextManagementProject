"""Deterministic sampling and descriptive log-log least-squares fits."""

import hashlib
import math

import numpy as np


REVIEW_FRACTIONS = (0.10, 0.30, 0.50, 1.00)

def log_fit(x, y):
    """OLS in log10 space; descriptive fit, not a power-law hypothesis test."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.any(x <= 0) or np.any(y <= 0) or len(set(x)) < 2:
        raise ValueError("fit requires distinct positive x values and positive y")
    lx, ly = np.log10(x), np.log10(y)
    slope, intercept = np.polyfit(lx, ly, 1)
    residual = ly - (slope * lx + intercept)
    denominator = float(np.sum((ly - np.mean(ly)) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / denominator if denominator else None
    return {
        "slope": float(slope), "intercept_log10": float(intercept),
        "coefficient": float(10**intercept), "r2_log10": r2, "n_points": len(x),
    }


def zipf_statistics(frequencies):
    ranked = sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))
    ranks = np.arange(1, len(ranked) + 1, dtype=float)
    counts = np.array([count for _, count in ranked], dtype=float)
    # Declare the interval before looking at results; exclude the count=1 floor.
    mask = (ranks >= 10) & (ranks <= 10_000) & (counts >= 2)
    fit = log_fit(ranks[mask], counts[mask])
    fit.update({"exponent_s": -fit["slope"], "rank_min": int(ranks[mask][0]),
                "rank_max": int(ranks[mask][-1]), "minimum_count": 2})
    full = log_fit(ranks, counts)
    full["exponent_s"] = -full["slope"]
    bands = []
    for lo, hi in ((1, 100), (10, 1000), (100, 10_000)):
        selected = (ranks >= lo) & (ranks <= hi)
        if selected.sum() < 2:
            continue
        band = log_fit(ranks[selected], counts[selected])
        band.update({"rank_min": int(ranks[selected][0]), "rank_max": int(ranks[selected][-1]),
                     "exponent_s": -band["slope"]})
        bands.append(band)
    return ranked, {"primary_fit": fit, "all_ranks_fit": full,
                    "rank_band_diagnostics": bands,
                    "hapax_types": int(np.sum(counts == 1)),
                    "definition": "collection occurrence count, not document frequency"}


def review_order(review_ids, seed):
    """Stable random-like permutation independent of Python's shuffle version."""
    return sorted(range(len(review_ids)), key=lambda i: (
        hashlib.sha256(f"{seed}\0{review_ids[i]}".encode()).digest(), review_ids[i]
    ))


def heaps_sampling_plan(review_ids, token_counts, seed=42, n_fit_points=80):
    """One review ordering, a log-spaced fit grid, and labelled checkpoints."""
    total = sum(token_counts)
    order = review_order(review_ids, seed)
    fit_tokens = set(np.rint(np.geomspace(1_000, total, n_fit_points)).astype(int).tolist())
    review_stops = {math.ceil(len(order) * f): f for f in REVIEW_FRACTIONS}
    review_fractions = {}
    cumulative = 0
    for rank, index in enumerate(order, start=1):
        cumulative += token_counts[index]
        if rank in review_stops:
            review_fractions[cumulative] = review_stops[rank]
    anchors = {n for n in (1_000, 10_000, 100_000, 1_000_000, total) if n <= total}
    return {"order": order, "fit_tokens": fit_tokens,
            "checkpoints": sorted(fit_tokens | anchors | set(review_fractions)),
            "review_fractions": review_fractions}


def heaps_token_points(tokenized_reviews, checkpoints=(1_000, 10_000, 100_000)):
    """Count nested token prefixes, stopping inside a review when necessary.

    Reviews arrive in the chosen sampling order. Checkpoints are increasing
    positive token counts; repeated words still count towards the token total.
    """
    targets = iter(checkpoints)
    target = next(targets)
    seen = set()
    total = 0
    result = []
    for reviews_started, terms in enumerate(tokenized_reviews, start=1):
        for term in terms:
            total += 1
            seen.add(term)
            if total == target:
                result.append({"tokens": total, "types": len(seen),
                               "reviews_started": reviews_started})
                target = next(targets, None)
                if target is None:
                    return result
    raise ValueError(f"Only {total} tokens available; cannot reach checkpoint {target}")


def heaps_statistics(points):
    """Fit the fixed log-spaced grid; separately predict the withheld endpoint."""
    fit_points = [p for p in points if p["fit_checkpoint"]]
    fit = log_fit([p["tokens"] for p in fit_points], [p["types"] for p in fit_points])
    fit.update({"k": fit["coefficient"], "beta": fit["slope"],
                "N_min": fit_points[0]["tokens"], "N_max": fit_points[-1]["tokens"]})
    full = points[-1]
    cutoff = math.floor(0.8 * full["tokens"])
    train = [p for p in fit_points if p["tokens"] <= cutoff]
    predictive = log_fit([p["tokens"] for p in train], [p["types"] for p in train])
    prediction = predictive["coefficient"] * full["tokens"] ** predictive["slope"]
    predictive.update({"k": predictive["coefficient"], "beta": predictive["slope"],
                       "cutoff_tokens": cutoff, "max_training_tokens": train[-1]["tokens"],
                       "total_tokens": full["tokens"], "actual_types": full["types"],
                       "predicted_types": prediction,
                       "absolute_percentage_error": abs(prediction / full["types"] - 1) * 100})
    return {"fit": fit, "prediction": predictive}
