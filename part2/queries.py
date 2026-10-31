"""Query set: 50 diverse queries, run through both scoring functions.

Why 50
------
The assignment brief itself only asks for "a set of at least five diverse queries on
both implementations" (3.3.2), but five is the **pass line, not the target**. Manning,
Raghavan and Schütze put it this way in Introduction to Information Retrieval, §8.1:

    "The test document collection and suite of information needs have to be of a
     reasonable size: you need to average performance over fairly large test sets,
     as results are highly variable over different documents and information needs.
     As a rule of thumb, 50 information needs has usually been found to be a
     sufficient minimum."

So 50 information needs is the "sufficient minimum" found by experience. That rule was
written for evaluation with relevance judgements (performance has to be averaged over
many queries, because a single query fluctuates enormously); this assignment has no
judgements, so we can only measure rank agreement. The underlying point, that a single
query is highly variable, holds all the same: on our earlier six-query set, overlap@10
between TF-IDF and BM25 jumped anywhere between 1/10 and 7/10, which is exactly what
that rule of thumb is about. Hence 50.

How the 50 are distributed
--------------------------
The 50 cannot just be 50 words grabbed at random. The query set has to support
attributing *where* the differences come from, so it is laid out along five axes; each
query carries exactly one primary axis, plus a purpose saying what it is meant to show:

    category     business category / topic search (12): venue types that genuinely
                 occur in the Yelp corpus, one to three words
    aspect       aspect + sentiment (12): the review dimensions people judge on -
                 service, waiting time, price, attitude
    attribute    attribute constraints (10): compound conditions such as gluten free
                 or dog friendly
    df_spectrum  single terms spread over the df spectrum (10): from df=13,139 down to
                 df=1, included specifically to test whether the rarity of the query
                 term is the variable that explains the models' disagreement
    edge         edge cases / special inputs (6): all stop words, negative-IDF terms,
                 repeated terms, contractions, zero-hit terms - these add no
                 statistical power, but they verify that the implementation does not
                 break, and they demonstrate clamping

The df band (df_band) is **not hard-coded into the table**; it is computed at run time
from the df actually measured in the index. A hard-coded band stops being true the
moment the corpus changes, whereas a measured band stays self-consistent after a corpus
swap. The band boundaries are given in DF_BANDS.

Why the edge cases matter (the six under `edge` are the ones worth reading):
    the and of            all three terms have df close to N, so TF-IDF's idf -> 0 and
                          BM25's Okapi idf clamps to 0: both sides should score every
                          document at 0, testing how cleanly the degenerate case is
                          handled
    the food / pizza and pasta
                          12 terms in the whole corpus have a negative raw Okapi idf
                          (df > (N+1)/2); a query carrying one of them is where the
                          clamping ablation happens live
    pizza pizza           the same term twice, testing whether term-frequency weight
                          and the query vector get distorted by double counting
    don't waste your money the contraction expands to do / not / waste / your / money,
                          testing whether the tokenizer's contraction expansion behaves
                          the same on the query side as on the document side
    zzyzx                 df = 0, no hits: tests whether the empty result is graceful
                          (no crash, metrics marked n/a)
"""

from text_processing import get_tokenizer

# df band boundaries: a query is binned by the df of its **most frequent** term.
# The larger the df, the more common the term and the weaker its discriminative power.
DF_BANDS = [
    ("df >= 10000 (very high)", 10000, None),
    ("3000 <= df < 10000 (high)", 3000, 10000),
    ("1000 <= df < 3000 (medium)", 1000, 3000),
    ("300 <= df < 1000 (low)", 300, 1000),
    ("0 < df < 300 (rare)", 1, 300),
    ("df = 0 (zero-hit)", 0, 1),
]

QUERIES = [
    # ---- category: business category / topic search (12) ----
    {"id": "Q01", "axis": "category", "text": "pizza",
     "purpose": "A single venue-type term, df=1,941: the most typical topic-search baseline"},
    {"id": "Q02", "axis": "category", "text": "sushi",
     "purpose": "A second venue-type term, df=967: same band as pizza, different topic, "
                "to check that the conclusions do not hinge on one particular term"},
    {"id": "Q03", "axis": "category", "text": "thai food",
     "purpose": "Venue type plus the generic term food (df=13,139), to see whether a "
                "high-frequency generic term is diluted away by IDF"},
    {"id": "Q04", "axis": "category", "text": "taco tuesday",
     "purpose": "A compound need with a time qualifier (Tuesday specials); neither term "
                "has a high df"},
    {"id": "Q05", "axis": "category", "text": "coffee shop",
     "purpose": "Venue type plus a place word: shop is high-frequency, so this probes "
                "the interference of a high-frequency generic term"},
    {"id": "Q06", "axis": "category", "text": "breakfast burrito",
     "purpose": "Meal period plus a single dish; the two dfs differ by roughly a factor "
                "of ten, probing the combination effect under very uneven df"},
    {"id": "Q07", "axis": "category", "text": "ice cream",
     "purpose": "A dessert venue type; ice doubles as adjective and noun, probing "
                "ambiguity of part of speech"},
    {"id": "Q08", "axis": "category", "text": "bbq chicken",
     "purpose": "An abbreviated venue type (bbq), checking whether the tokenizer "
                "handles abbreviations consistently on both sides"},
    {"id": "Q09", "axis": "category", "text": "barber shop",
     "purpose": "A low-frequency venue type (barber, df=41): a long-tail query with few "
                "candidates, where the rankings should agree closely"},
    {"id": "Q10", "axis": "category", "text": "dentist appointment",
     "purpose": "A low-frequency service business (dentist, df=142): long tail like "
                "barber, serving as a counterpart to it"},
    {"id": "Q11", "axis": "category", "text": "oil change",
     "purpose": "Automotive service; change is a high-frequency polyseme, probing the "
                "pull of an ambiguous term"},
    {"id": "Q12", "axis": "category", "text": "brake repair",
     "purpose": "Extremely low frequency (brake, df=21) with very few candidates: an "
                "extreme sample at the long-tail end"},

    # ---- aspect: aspect + sentiment (12) ----
    {"id": "Q13", "axis": "aspect", "text": "terrible service",
     "purpose": "Negative sentiment plus an aspect term: service (df=11,347) is a "
                "high-frequency generic term, so this checks whether the sentiment term "
                "can outweigh it"},
    {"id": "Q14", "axis": "aspect", "text": "rude staff",
     "purpose": "Negative sentiment plus a medium-frequency aspect term (staff, df=5,668)"},
    {"id": "Q15", "axis": "aspect", "text": "friendly staff",
     "purpose": "Same aspect term as Q14 with opposite sentiment polarity: a minimal "
                "pair differing only in the sentiment term"},
    {"id": "Q16", "axis": "aspect", "text": "slow service",
     "purpose": "The speed dimension plus a high-frequency aspect term: same aspect "
                "term as Q13, different dimension term"},
    {"id": "Q17", "axis": "aspect", "text": "long wait",
     "purpose": "Waiting time: wait has df=3,506 and tends to show up in short Yelp "
                "reviews, probing length bias"},
    {"id": "Q18", "axis": "aspect", "text": "overpriced",
     "purpose": "A single term on the price dimension, df=339: low frequency and "
                "semantically unambiguous"},
    {"id": "Q19", "axis": "aspect", "text": "best pizza in town",
     "purpose": "A known disagreement: TF-IDF's top hit is a short review that never "
                "mentions pizza (best + town beat pizza), and in is itself a "
                "negative-IDF term, so one query does two jobs"},
    {"id": "Q20", "axis": "aspect", "text": "worst experience ever",
     "purpose": "Three terms, all negative: experience and ever are both high-frequency "
                "generic terms, so this asks which one dominates when generic terms "
                "stack up"},
    {"id": "Q21", "axis": "aspect", "text": "cold food",
     "purpose": "The temperature/quality dimension plus the high-frequency generic term "
                "food, as a counterpart to the food in Q03"},
    {"id": "Q22", "axis": "aspect", "text": "wrong order",
     "purpose": "A specific kind of service failure: order is polysemous (a food order "
                "versus a sequence)"},
    {"id": "Q23", "axis": "aspect", "text": "dirty bathroom",
     "purpose": "The hygiene dimension: bathroom is low-frequency and dirty "
                "medium-low, making the query low-frequency overall"},
    {"id": "Q24", "axis": "aspect", "text": "great atmosphere",
     "purpose": "Positive sentiment plus an abstract noun: abstract terms are scattered "
                "across reviews, so this looks at the shape of what gets retrieved"},

    # ---- attribute: attribute constraints (10) ----
    {"id": "Q25", "axis": "attribute", "text": "gluten free pancakes",
     "purpose": "A three-term compound constraint (dietary restriction plus a dish): "
                "the information need closest to a real search"},
    {"id": "Q26", "axis": "attribute", "text": "vegan options",
     "purpose": "Dietary restriction plus a generic term: vegan has df=182, very low "
                "frequency"},
    {"id": "Q27", "axis": "attribute", "text": "dog friendly patio",
     "purpose": "A three-term attribute, pet friendly plus outdoor seating: patio has "
                "df=455"},
    {"id": "Q28", "axis": "attribute", "text": "outdoor seating",
     "purpose": "A two-term attribute constraint: semantically the same as Q27 (outdoor "
                "seating) but worded differently, probing sensitivity to wording"},
    {"id": "Q29", "axis": "attribute", "text": "drive thru",
     "purpose": "A service-format attribute: thru is the colloquial spelling of through, "
                "checking whether the tokenizer treats it consistently on both sides"},
    {"id": "Q30", "axis": "attribute", "text": "free wifi",
     "purpose": "Free plus an amenity: free has df=1,478 and wifi df=75, two attribute "
                "terms with very different df"},
    {"id": "Q31", "axis": "attribute", "text": "happy hour",
     "purpose": "A promotion attribute: happy has df=2,460 and doubles as a sentiment "
                "adjective, probing semantic drift in adjectives"},
    {"id": "Q32", "axis": "attribute", "text": "kid friendly",
     "purpose": "An audience attribute plus a sentiment term: shares friendly with Q15 "
                "(friendly staff), forming a counterpart to it"},
    {"id": "Q33", "axis": "attribute", "text": "wheelchair accessible",
     "purpose": "An accessibility attribute: both terms are low-frequency, a long-tail "
                "attribute search"},
    {"id": "Q34", "axis": "attribute", "text": "takeout order",
     "purpose": "A service-format attribute: shares the polysemous order with Q22, "
                "probing how an ambiguous term behaves in different contexts"},

    # ---- df_spectrum: single terms spread over the df spectrum (10) ----
    {"id": "Q35", "axis": "df_spectrum", "text": "food",
     "purpose": "df=13,139, the top of the spectrum: it appears in roughly one review in "
                "three, so it discriminates least"},
    {"id": "Q36", "axis": "df_spectrum", "text": "service",
     "purpose": "df=11,347: another very-high-frequency generic term like food, serving "
                "as a replication of it"},
    {"id": "Q37", "axis": "df_spectrum", "text": "staff",
     "purpose": "df=5,668, the high band"},
    {"id": "Q38", "axis": "df_spectrum", "text": "wait",
     "purpose": "df=3,506, the lower edge of the high band"},
    {"id": "Q39", "axis": "df_spectrum", "text": "salad",
     "purpose": "df=1,993, the medium band"},
    {"id": "Q40", "axis": "df_spectrum", "text": "burger",
     "purpose": "df=1,552, the medium band: same band as salad, different term"},
    {"id": "Q41", "axis": "df_spectrum", "text": "rude",
     "purpose": "df=917, the low band: a sentiment term"},
    {"id": "Q42", "axis": "df_spectrum", "text": "patio",
     "purpose": "df=455, the low band: a noun"},
    {"id": "Q43", "axis": "df_spectrum", "text": "espresso",
     "purpose": "df=76, the rare band: very few candidates, so the two models should "
                "agree closely"},
    {"id": "Q44", "axis": "df_spectrum", "text": "xylophone",
     "purpose": "df=1, the most extreme point on the spectrum (only one document in the "
                "whole corpus mentions it), probing behaviour at extreme rarity"},

    # ---- edge: edge cases / special inputs (6) ----
    {"id": "Q45", "axis": "edge", "text": "the and of",
     "purpose": "All three terms have df > 20,000: TF-IDF's idf -> 0 and BM25's Okapi "
                "idf is clamped to 0 throughout, so both sides should score every "
                "document at 0 (empty result set or all scores tied)"},
    {"id": "Q46", "axis": "edge", "text": "the food",
     "purpose": "the has a raw Okapi idf of -2.03: the harshest case in the clamping "
                "ablation (measured top-10 overlap of 1/10)"},
    {"id": "Q47", "axis": "edge", "text": "pizza and pasta",
     "purpose": "and has a raw idf of -2.12: top-10 overlap is 6/10 before and after "
                "clamping, and the top hit differs"},
    {"id": "Q48", "axis": "edge", "text": "pizza pizza",
     "purpose": "A repeated term: TF-IDF's query vector should not be inflated by "
                "double counting, and BM25 treats the query terms as deduplicated"},
    {"id": "Q49", "axis": "edge", "text": "don't waste your money",
     "purpose": "The contraction expands to do / not / waste / your / money (your is "
                "high-frequency), testing whether not is then scored as an ordinary "
                "term"},
    {"id": "Q50", "axis": "edge", "text": "zzyzx",
     "purpose": "A zero-hit query with df = 0: both sides should return an empty result, "
                "with metrics marked n/a rather than crashing"},
]


def df_band(max_df):
    """Place a df into one of the DF_BANDS buckets and return the bucket name."""
    for name, low, high in DF_BANDS:
        if high is None:
            if max_df >= low:
                return name
        elif low <= max_df < high:
            return name
    return DF_BANDS[-1][0]


def profile_query(index, text):
    """Build a measured profile of one query: tokenization, per-term df, df band,
    and the number of candidate documents.

    Every df is computed on the fly rather than stored in QUERIES, so the profile
    follows automatically when the corpus or the tokenizer changes, and the table can
    never end up claiming df=1941 while the real value has moved on.
    """
    terms = get_tokenizer(index.tokenizer).tokenize(text)
    dfs = [{"term": term, "df": index.df(term)} for term in sorted(set(terms))]
    hit_df = [item["df"] for item in dfs if item["df"]]
    candidates = set()
    for item in dfs:
        if item["df"]:
            candidates.update(index.postings[item["term"]])
    return {
        "text": text,
        "terms": terms,
        "n_terms": len(terms),
        "n_unique_terms": len({term: 1 for term in terms}),
        "term_df": dfs,
        "df_min": min(hit_df) if hit_df else 0,
        "df_max": max(hit_df) if hit_df else 0,
        "df_band": df_band(max(hit_df) if hit_df else 0),
        "n_candidates": len(candidates),
    }


def query_axes():
    """Count the queries on each axis, to check that the query set is not skewed
    onto a single axis."""
    counts = {}
    for query in QUERIES:
        counts[query["axis"]] = counts.get(query["axis"], 0) + 1
    return counts
