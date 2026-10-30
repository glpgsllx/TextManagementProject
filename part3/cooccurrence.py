"""Window-based PMI for one aspect term per year, and how its top words drift."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from text_processing import get_tokenizer


# Written here because NLTK's list needs a separate download. Negations such as "not"
# are removed too: they mark polarity, not what reviewers associate with an aspect.
STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further had
has have having he her here hers herself him himself his how i if in into is it its itself just
me more most my myself no nor not now of off on once only or other our ours ourselves out over
own same she should so some such than that the their theirs them themselves then there these
they this those through to too under until up very was we were what when where which while who
whom why will with would you your yours yourself yourselves
""".split())


@dataclass
class YearCorpus:
    """Token and pair occurrences tagged by review, so resampling only reweights reviews."""

    target: int
    token_ids: np.ndarray
    token_reviews: np.ndarray
    pair_ids: np.ndarray
    pair_reviews: np.ndarray
    review_businesses: np.ndarray


def is_candidate(word, exclude=()):
    return word.isalpha() and len(word) >= 3 and word not in STOPWORDS and word not in exclude


def window_neighbors(tokens, target, window):
    """Every word within window positions of each target occurrence, target itself excluded."""
    neighbors = []
    for i, token in enumerate(tokens):
        if token == target:
            nearby = tokens[max(0, i - window):i] + tokens[i + 1:i + window + 1]
            neighbors.extend(word for word in nearby if word != target)
    return neighbors


def pmi(pair_count, word_count, target_count, total):
    """log2 of P(t, w) / (P(t) P(w)), each probability estimated over total tokens."""
    return np.log2(pair_count * total / (word_count * target_count))


def tokenize_by_year(reviews):
    tokenizer = get_tokenizer("custom")
    by_year = defaultdict(list)
    for review in reviews:
        by_year[review["year"]].append((review["business_id"], tokenizer.tokenize(review["text"])))
    return by_year


def merge_periods(by_year, periods):
    """Pool yearly documents into named ranges such as {"2014-2016": (2014, 2016)}."""
    return {name: [doc for year in by_year if start <= year <= end for doc in by_year[year]]
            for name, (start, end) in periods.items()}


def build_corpus(documents, target, window, vocabulary):
    """Flatten one year's (business, tokens) documents into arrays for rank_words()."""
    businesses = {}
    token_ids, token_reviews, pair_ids, pair_reviews, review_businesses = [], [], [], [], []
    for review, (business, tokens) in enumerate(documents):
        review_businesses.append(businesses.setdefault(business, len(businesses)))
        token_ids.extend(vocabulary.setdefault(token, len(vocabulary)) for token in tokens)
        token_reviews.extend([review] * len(tokens))
        neighbors = window_neighbors(tokens, target, window)
        pair_ids.extend(vocabulary[word] for word in neighbors)
        pair_reviews.extend([review] * len(neighbors))
    return YearCorpus(vocabulary.setdefault(target, len(vocabulary)), np.array(token_ids), np.array(token_reviews),
                      np.array(pair_ids, dtype=int), np.array(pair_reviews, dtype=int), np.array(review_businesses))


def rank_words(corpus, allowed, min_count, min_businesses, weights=None):
    """Return eligible word ids ranked by PMI, with their scores and supporting counts.

    weights is how many times each review counts; None means the observed corpus once.
    """
    if weights is None:
        weights = np.ones(len(corpus.review_businesses), dtype=int)
    size = len(allowed)
    words = np.bincount(corpus.token_ids, weights[corpus.token_reviews], minlength=size)
    pairs = np.bincount(corpus.pair_ids, weights[corpus.pair_reviews], minlength=size)
    present = weights[corpus.pair_reviews] > 0
    links = np.unique(np.column_stack((corpus.pair_ids[present],
                                       corpus.review_businesses[corpus.pair_reviews[present]])), axis=0)
    businesses = np.bincount(links[:, 0], minlength=size)
    eligible = np.flatnonzero(allowed & (pairs >= min_count) & (businesses >= min_businesses))
    if len(eligible) == 0 or words[corpus.target] == 0:
        return np.array([], dtype=int), np.array([]), pairs, businesses
    scores = pmi(pairs[eligible], words[eligible], words[corpus.target], words.sum())
    # Word id breaks ties so that equal scores keep a reproducible order.
    order = np.lexsort((eligible, -scores))
    return eligible[order], scores[order], pairs, businesses


def jaccard(first, second):
    first, second = set(first), set(second)
    return len(first & second) / len(first | second) if first | second else 0.0


def sampling_stability(corpus, allowed, top_n, min_count, min_businesses, rng, resamples):
    """Mean Top-N overlap between the observed year and bootstrap copies of the same year.

    Reviews are the same people writing in the same year, so this is the overlap expected
    from sampling noise alone. Year-to-year overlap well below it suggests real drift.
    """
    observed = rank_words(corpus, allowed, min_count, min_businesses)[0][:top_n]
    reviews = len(corpus.review_businesses)
    overlaps = []
    for _ in range(resamples):
        weights = rng.multinomial(reviews, np.full(reviews, 1 / reviews))
        overlaps.append(jaccard(observed, rank_words(corpus, allowed, min_count, min_businesses, weights)[0][:top_n]))
    return float(np.mean(overlaps))


def analyze_term(by_period, target, window, top_n, min_count, min_businesses, rng, resamples, exclude=()):
    """Return the Top-N rows, per-period drift rows and each period's full word ranking."""
    vocabulary = {}
    periods = sorted(by_period)
    corpora = {period: build_corpus(by_period[period], target, window, vocabulary) for period in periods}
    names = sorted(vocabulary, key=vocabulary.get)
    allowed = np.array([is_candidate(word, exclude) for word in names])
    top_rows, drift_rows, rankings, tops = [], [], {}, {}
    for period in periods:
        ranked, scores, pairs, businesses = rank_words(corpora[period], allowed, min_count, min_businesses)
        rankings[period] = {names[word]: {"rank": rank, "pmi": float(score), "pairs": int(pairs[word])}
                            for rank, (word, score) in enumerate(zip(ranked, scores), 1)}
        tops[period] = [names[word] for word in ranked[:top_n]]
        top_rows.extend({"term": target, "period": period, "rank": rank, "word": names[word],
                         "pmi": round(float(score), 4), "cooccurrences": int(pairs[word]),
                         "businesses": int(businesses[word])}
                        for rank, (word, score) in enumerate(zip(ranked[:top_n], scores[:top_n]), 1))
    for previous, period in zip([None] + periods, periods):
        drift_rows.append({
            "term": target, "period": period,
            "target_count": sum(tokens.count(target) for _, tokens in by_period[period]),
            "eligible_words": len(rankings[period]),
            "jaccard_previous": "" if previous is None else round(jaccard(tops[previous], tops[period]), 4),
            "jaccard_first": round(jaccard(tops[periods[0]], tops[period]), 4),
            "same_period_resample_jaccard": round(sampling_stability(
                corpora[period], allowed, top_n, min_count, min_businesses, rng, resamples), 4),
            "new_since_previous": "" if previous is None else " ".join(w for w in tops[period] if w not in tops[previous]),
        })
    return top_rows, drift_rows, rankings


def tracked_words(rankings, target, words):
    """Rank and PMI of chosen words over time; a word that misses the filters has blanks."""
    rows = []
    for word in words:
        for period, ranking in sorted(rankings.items()):
            entry = ranking.get(word, {})
            rows.append({"term": target, "word": word, "period": period, "rank": entry.get("rank", ""),
                         "pmi": round(entry["pmi"], 4) if entry else "", "cooccurrences": entry.get("pairs", "")})
    return rows


def theme_rates(by_year, target, themes, window, rng, resamples):
    """Theme words near the target per 100 target mentions, with a bootstrap 95% interval."""
    rows = []
    for year in sorted(by_year):
        mentions = np.array([tokens.count(target) for _, tokens in by_year[year]])
        for theme, words in themes.items():
            hits = np.array([sum(word in words for word in window_neighbors(tokens, target, window))
                             for _, tokens in by_year[year]])
            draws = rng.integers(0, len(mentions), (resamples, len(mentions)))
            rates = 100 * hits[draws].sum(axis=1) / mentions[draws].sum(axis=1)
            low, high = np.percentile(rates, [2.5, 97.5])
            rows.append({"term": target, "year": year, "theme": theme, "mentions": int(mentions.sum()),
                         "cooccurrences": int(hits.sum()), "per_100_mentions": round(100 * hits.sum() / mentions.sum(), 3),
                         "ci_low": round(float(low), 3), "ci_high": round(float(high), 3)})
    return rows


def window_sensitivity(by_year, target, windows, baseline, top_n, min_count, min_businesses, exclude=()):
    """Top-N agreement between each window size and the baseline window, per year."""
    rows = []
    vocabulary = {}
    for year in sorted(by_year):
        corpora = {w: build_corpus(by_year[year], target, w, vocabulary) for w in windows}
        names = sorted(vocabulary, key=vocabulary.get)
        allowed = np.array([is_candidate(word, exclude) for word in names])
        tops = {w: rank_words(corpora[w], allowed, min_count, min_businesses)[0][:top_n] for w in windows}
        for w in windows:
            rows.append({"term": target, "year": year, "window": w,
                         "jaccard_with_baseline": round(jaccard(tops[w], tops[baseline]), 4)})
    return rows
