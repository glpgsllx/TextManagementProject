"""Deterministic sampling and descriptive log-log least-squares fits."""

import hashlib
import math

import numpy as np


REQUIRED_FRACTIONS = (0.10, 0.30, 0.50, 1.00)
DENSE_FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00)


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


def heaps_points(review_ids, token_counts, document_types, seed=42):
    n_docs = len(review_ids)
    checkpoints = {max(1, math.ceil(n_docs * f)): f for f in DENSE_FRACTIONS}
    seen = set()
    total_tokens = 0
    result = []
    for rank, index in enumerate(review_order(review_ids, seed), start=1):
        total_tokens += token_counts[index]
        seen.update(document_types[index])
        if rank in checkpoints:
            fraction = checkpoints[rank]
            result.append({"seed": seed, "fraction": fraction, "reviews": rank,
                           "tokens": total_tokens, "types": len(seen),
                           "required_checkpoint": fraction in REQUIRED_FRACTIONS})
    return result


def heaps_statistics(points):
    selected = [p for p in points if p["required_checkpoint"]]
    if len(selected) != 4:
        raise ValueError("dataset too small to form four required checkpoints")
    fit = log_fit([p["tokens"] for p in selected], [p["types"] for p in selected])
    fit.update({"k": fit["coefficient"], "beta": fit["slope"]})
    train = [p for p in selected if p["fraction"] < 1]
    predictive = log_fit([p["tokens"] for p in train], [p["types"] for p in train])
    full = selected[-1]
    prediction = predictive["coefficient"] * full["tokens"] ** predictive["slope"]
    predictive.update({"k": predictive["coefficient"], "beta": predictive["slope"],
                       "held_out_fraction": 1.0, "actual_types": full["types"],
                       "predicted_types": prediction,
                       "relative_error": (prediction - full["types"]) / full["types"]})
    return {"four_checkpoint_fit": fit, "fit_10_30_50_predict_100": predictive}
