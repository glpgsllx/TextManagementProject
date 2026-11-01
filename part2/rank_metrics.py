"""Ranking agreement metrics (used when no relevance judgments exist): overlap@k and Kendall tau-b.

Why these two metrics are needed
--------------------------------
The assignment gives us a corpus but no relevance judgments (qrels), so none of the
**judgment-dependent** metrics — P@10, MAP, NDCG — can be computed at all. All we can
measure is "how similar are two rankings", and that splits into two layers:

    overlap@k  looks only at the sets: of the top k results on each side, how many are
               the same document.
               It answers "are the two sides retrieving the same documents", ignoring
               their internal order entirely.
    Kendall τ  looks only at the order: take the documents both sides retrieved and
               check whether every pairwise precedence agrees.
               It answers "do the two sides order that shared set of documents the same
               way".

Both metrics must be reported together: high overlap with low τ means roughly the same
documents were picked but they disagree sharply on which comes first; low overlap with
high τ means few documents are shared, yet their relative order agrees closely.

We use tau-b rather than tau-a: a scoring function produces **ties** (documents with
equal scores), and tau-a keeps those ties in the denominator, which systematically
depresses τ; tau-b subtracts the tied pairs on each side from the denominator (see the
Wikipedia definition of the "Kendall rank correlation coefficient", where n1/n2 are
exactly the numbers of tied pairs on the two sides).

On significance: only when neither side has ties (n1 = n2 = 0) does the sampling
variance of τ admit the classic z_A = (nc − nd) / sqrt(n(n−1)(2n+5)/18) approximation
(a result that holds for strict rankings); only then do we report a p value. When ties
are present we give no p, rather than leaning on an approximation that does not hold to
claim "significance".
"""

import math
from itertools import combinations


def rank_map(ranked):
    """[(doc_id, score)] → {doc_id: rank}; ranks start at 1 and cover every candidate document.

    The input must be **already sorted with a deterministic tiebreak**: descending
    score, ties broken by ascending doc_id. With a deterministic tiebreak the order of
    tied documents is reproducible; without one, the order inside a tied block is just
    dict iteration order (stable in Python but meaningless), and noise would be
    mistaken for a "ranking difference".
    """
    return {doc_id: position for position, (doc_id, _) in enumerate(ranked, 1)}


def overlap_at_k(ranked_a, ranked_b, k):
    """Size of the intersection between the top k results of the two sides (0 … k).

    Note that this is the **size of the intersection of two sets**, not a Jaccard index:
    the denominator is fixed at k, so "7/10" reads as "7 of the top 10 appear on both
    sides", which is intuitive and directly comparable across queries.
    """
    set_a = {doc_id for doc_id, _ in ranked_a[:k]}
    set_b = {doc_id for doc_id, _ in ranked_b[:k]}
    return len(set_a & set_b)


def kendall_tau_b(rank_x, rank_y):
    """Kendall tau-b: the denominator excludes the tied pairs on each side. Both dictionaries must cover the same set of documents.

    Symbol by symbol (n = number of documents compared):
        n0 = n(n−1)/2              all document pairs, the denominator of tau-a
        n1 = pairs tied only on the x side
        n2 = pairs tied only on the y side
        nc = concordant pairs: both sides put a before b (or both put a after b)
        nd = discordant pairs: one side says a is ahead, the other says b is ahead
        τ  = (nc − nd) / sqrt((n0 − n1)(n0 − n2))

    τ = +1 means perfect agreement, −1 perfect disagreement, 0 no association. Returns
    None when it cannot be computed (fewer than 2 documents, or a zero denominator
    because everything is tied within each side — which happens once idf is clamped and
    many documents share the same score).
    """
    docs = list(rank_x)
    n = len(docs)
    if n < 2:
        return None
    n0 = n * (n - 1) // 2
    n1 = n2 = nc = nd = 0
    for a, b in combinations(docs, 2):
        dx = rank_x[a] - rank_x[b]
        dy = rank_y[a] - rank_y[b]
        if dx == 0:
            n1 += 1
        if dy == 0:
            n2 += 1
        if dx == 0 or dy == 0:
            continue  # tied on one side: this pair carries no direction, counted into n1/n2 then skipped
        if (dx > 0) == (dy > 0):
            nc += 1
        else:
            nd += 1
    denom = math.sqrt((n0 - n1) * (n0 - n2))
    if denom == 0:
        return None
    tau = (nc - nd) / denom

    z = None
    if n1 == 0 and n2 == 0:
        variance = n * (n - 1) * (2 * n + 5) / 18.0
        if variance > 0:
            z = (nc - nd) / math.sqrt(variance)
    p_two_sided = math.erfc(abs(z) / math.sqrt(2)) if z is not None else None
    return {"tau": tau, "n": n, "n0": n0, "n1": n1, "n2": n2, "nc": nc, "nd": nd,
            "z": z, "p_two_sided": p_two_sided}


def compare_pair(ranked_a, ranked_b, ks=(5, 10, 20, 50), pool=50):
    """Compare two rankings, reporting in one pass overlap@k, tau-b (union / intersection), and whether the top-1 result matches.

    pool (50 by default) is the "deep pool": the top pool results are taken from each
    side.

        union τ         computed over the **union** of the two sides' top pool results.
                        The largest sample, but it also contains documents that one side
                        ranked very low and the other never retrieved at all, so
                        membership differences leak into the order comparison and
                        negative values arise easily.
        intersection τ  computed only over documents that made the top pool on **both**
                        sides. The membership is fixed, so this is a pure order
                        comparison, but the sample shrinks and is inherently biased
                        toward documents both sides agree on, so the reading runs high.

    Report both: union τ is the conservative estimate of "overall agreement",
    intersection τ is "ordering agreement on the shared documents".

    On documents that "one side never retrieved at all": this is not an anomaly, it is
    the norm. BM25's Okapi idf clamps to 0 every term with df > N/2, and terms with
    idf 0 are skipped outright, so **a document matching only such a term never appears
    in BM25's results at all**, yet it may well appear in TF-IDF's top 50 (where
    idf = ln(N/df) is merely small, not 0). Such a document has no rank on the side
    where it is missing, and is uniformly recorded as "past the last rank of that side"
    (that is, len(results on that side) + 1, all tied with one another) — the standard
    censored handling, meaning "not retrieved = behind every retrieved document".
    n_missing_a / n_missing_b report how many documents fall into this category; if
    either is non-zero, bear in mind that τ is mixing in membership differences.
    """
    rank_a, rank_b = rank_map(ranked_a), rank_map(ranked_b)
    top_a = [doc_id for doc_id, _ in ranked_a[:pool]]
    top_b = [doc_id for doc_id, _ in ranked_b[:pool]]

    union = set(top_a) | set(top_b)
    intersection = set(top_a) & set(top_b)
    # documents one side failed to retrieve: recorded as past that side's last rank (tied)
    missing_a = len(rank_a) + 1
    missing_b = len(rank_b) + 1

    tau_union = kendall_tau_b({d: rank_a.get(d, missing_a) for d in union},
                              {d: rank_b.get(d, missing_b) for d in union})
    tau_intersection = (kendall_tau_b({d: rank_a[d] for d in intersection},
                                      {d: rank_b[d] for d in intersection})
                        if len(intersection) >= 2 else None)

    return {
        "overlap": {f"@{k}": overlap_at_k(ranked_a, ranked_b, k) for k in ks},
        "tau_b_union": tau_union,
        "tau_b_intersection": tau_intersection,
        "n_union": len(union),
        "n_intersection": len(intersection),
        "n_missing_a": sum(1 for d in union if d not in rank_a),
        "n_missing_b": sum(1 for d in union if d not in rank_b),
        "top1_same": bool(top_a) and bool(top_b) and top_a[0] == top_b[0],
    }


def self_check():
    """Self-check of the τ implementation: identical order 1, reversed order −1, hand-computed 4/6. A wrong coefficient shows up immediately."""
    docs = list(range(1, 9))
    same = kendall_tau_b({d: i for i, d in enumerate(docs, 1)},
                         {d: i for i, d in enumerate(docs, 1)})
    reversed_ = kendall_tau_b({d: i for i, d in enumerate(docs, 1)},
                              {d: 9 - i for i, d in enumerate(docs, 1)})
    x = {"a": 1, "b": 2, "c": 3, "d": 4}
    y = {"a": 1, "b": 3, "c": 2, "d": 4}   # one pair swapped: nc = 5, nd = 1, n0 = 6
    hand = kendall_tau_b(x, y)
    assert abs(same["tau"] - 1.0) < 1e-12, same
    assert abs(reversed_["tau"] + 1.0) < 1e-12, reversed_
    assert abs(hand["tau"] - 4 / 6) < 1e-12, hand
    return {"identical": same["tau"], "reversed": reversed_["tau"],
            "hand_check": hand["tau"]}
