"""Okapi BM25 retrieval (Assignment 3.3.2), with three variants: course, Okapi, and Lucene.

All three variants share a single "term-frequency saturation + length normalization"
factor; only the idf formula differs:

    score(Q, d) = Σ_{t∈Q}  idf(t)  ·  (k1 + 1) · tf  /  (tf + k1 · (1 − b + b · dl / avgdl))

Symbol by symbol:
    tf    how many times the term occurs in document d
    dl    length of document d after tokenization; avgdl is the corpus-wide average
          length, computed once at index time
    k1    how fast term frequency saturates: the larger tf is, the closer this factor
          gets to its ceiling (k1 + 1) — 20 occurrences are not 20 times the evidence
    b     strength of length normalization: b = 0 ignores length entirely, b = 1
          penalizes by length in full; the default is 0.75
    idf   inverse document frequency — the only place where the three variants differ,
          described below

Three versions of idf:

    course  ln(N / df)
            The course-slide version (Lecture 9.1, pp. 18–19), and the same one used
            in TF-IDF. It is 0 when df = N; when the term is absent from the index
            (df = 0) it is treated as 0 and the whole term is skipped.

    okapi   max(0, ln((N − df + 0.5) / (df + 0.5)))
            The standard formula of Robertson & Zaragoza 2009 (the degenerate form
            with R = r = 0, i.e. no relevance-feedback information).
            Note that it **can go negative**: when df > (N + 1) / 2 the numerator is
            smaller than the denominator. A negative value means "a document loses
            score for containing this query term", which defies common sense, so it
            must be clamped to 0. On this corpus 12 terms are affected; the most
            frequent of them, "and" (df = 33,401 with N = 37,401), has a raw idf of
            −2.12.

    okapi_raw (experimental, not delivered)   ln((N − df + 0.5) / (df + 0.5)), i.e.
            the original Okapi formula with clamping removed.
            Used only for the ablation experiment, to quantify what clamping actually
            changes. See "On the unclamped idf" below.

    lucene  ln(1 + (N − df + 0.5) / (df + 0.5))
            The actual implementation of BM25Similarity in Lucene 8.x / 9.x (and
            therefore in Elasticsearch 7 and 8). The official documentation puts it
            this way: "Implemented as log(1 + (docCount - docFreq + 0.5)/(docFreq + 0.5))".
            The + 1 is deliberate on Lucene's part: it rules out negative idf without
            having to clamp hard the way Okapi does. It almost coincides with the
            course version when df ≪ N (ln((N+1)/(df+0.5)) ≈ ln(N/df)); the difference
            only surfaces as df approaches N — the course version tends to 0, the
            Lucene version to ln2 = 0.693.

On the unclamped idf (experiments only)
---------------------------------------
`okapi_idf_unclamped` is the original Okapi formula with max(0, ·) removed, so it
returns negative values. It is **deliberately not registered** in `IDF_BY_VARIANT`,
which means `bm25_search(index, q, variant=...)` cannot reach it whatever you pass,
and a script that loops over the variants will not stumble onto it by accident. To use
it in an experiment you have to spell it out:

    bm25_search(index, q, idf_fn=okapi_idf_unclamped)

Once `idf_fn` is supplied it takes over idf completely and `variant` plays no part.
The default delivered path (no idf_fn) only ever runs the three variants registered
in `IDF_BY_VARIANT`.

A note on scale: current Lucene mainline (10.x) has dropped the constant (k1 + 1),
which is the same for every term. That constant does not affect ranking, only the
score scale, so we keep the (k1 + 1) that the course slides and the Okapi textbook
have in common, letting the three variants differ cleanly in idf alone.

Queries use de-duplicated terms only (consistent with the way Lucene handles
TermQuery): the (k3 + 1)·tf_q / (k3 + tf_q) query-term saturation factor that the
slides prepare for long queries is not implemented, because the queries in this
assignment have 2–4 terms with tf_q = 1 each, which makes that factor identically 1.
Add it if long queries ever become necessary (using a whole review as the query, say).

Scoring walks the postings term-at-a-time: for each query term it scans one posting
list, so only documents matching at least one query term show up as candidate
documents — there is no need to sweep the whole collection. Terms whose idf comes out
as 0 are skipped outright: their contribution is 0 for every document, and skipping
them also saves reading one more posting list.
"""

import math

from text_processing import get_tokenizer

COURSE = "course"
OKAPI = "okapi"
LUCENE = "lucene"

DEFAULT_K1 = 1.2
DEFAULT_B = 0.75


def course_idf(df, n_docs):
    """Course-slide idf = ln(N / df). Returns 0 when df = 0 (term absent from the index)."""
    return math.log(n_docs / df) if df else 0.0


def okapi_idf(df, n_docs):
    """Okapi idf = max(0, ln((N − df + 0.5) / (df + 0.5))); negative values clamped to 0."""
    return max(0.0, math.log((n_docs - df + 0.5) / (df + 0.5)))


def lucene_idf(df, n_docs):
    """Lucene idf = ln(1 + (N − df + 0.5) / (df + 0.5)); non-negative by construction."""
    return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))


def okapi_idf_unclamped(df, n_docs):
    """Experimental: the original Okapi formula without clamping, negative when df > (N + 1) / 2.

    It exists solely for the ablation experiment — only by obtaining negative idf
    values can we quantify how much the clamping step actually changes the outcome.
    It is **not** registered in IDF_BY_VARIANT, so the variant argument of
    bm25_search cannot select it; it must be passed explicitly through idf_fn=.
    Do not use it on the delivered path: a negative idf makes "the document contains
    the query term" count against that document, a pathological behaviour that the
    textbook explicitly warns against.
    """
    return math.log((n_docs - df + 0.5) / (df + 0.5))


# Variants accepted on the delivered path. okapi_idf_unclamped is deliberately absent; see the module header.
IDF_BY_VARIANT = {COURSE: course_idf, OKAPI: okapi_idf, LUCENE: lucene_idf}


def term_weight(tf, dl, avgdl, k1=DEFAULT_K1, b=DEFAULT_B):
    """Term-frequency saturation + length normalization factor: (k1 + 1) · tf / (tf + k1 · (1 − b + b · dl / avgdl)).

    As tf → ∞ it converges to (k1 + 1): the marginal evidence of a repeat decays, so
    stuffing a document with keywords cannot inflate its score. When dl = avgdl the
    parenthesised term equals 1 and length normalization does nothing; the larger dl
    is, the larger the denominator and the lower the score.
    """
    return tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))


def bm25_search(index, query_text, k1=DEFAULT_K1, b=DEFAULT_B,
                variant=OKAPI, top_k=10, idf_fn=None):
    """Run a natural-language query, returning [(doc_id, score)] sorted by descending score, ties broken by ascending doc_id.

    query_text arrives as raw text and is tokenized internally with the same tokenizer
    that was used to build the index. variant selects course / okapi (the default) /
    lucene; k1 and b are tunable parameters (the assignment text asks us to "correctly
    handle the parameters k1 and b").

    idf_fn is an explicit experiment hook: once supplied it is used to compute idf and
    variant no longer participates. This is the only way to reach a variant that is not
    registered in IDF_BY_VARIANT (such as the unclamped okapi_idf_unclamped). It
    defaults to None, which goes through the variant table; do not pass it on the
    delivered path.
    """
    if idf_fn is not None:
        idf = idf_fn
    else:
        if variant not in IDF_BY_VARIANT:
            raise ValueError(f"unsupported idf variant: {variant}, available: {sorted(IDF_BY_VARIANT)}")
        idf = IDF_BY_VARIANT[variant]

    terms = set(get_tokenizer(index.tokenizer).tokenize(query_text))
    n_docs = len(index.doc_lengths)
    scores = {}
    for term in terms:
        plist = index.postings.get(term)
        if not plist:
            continue  # term absent from the index: df = 0, its contribution to every document is 0
        weight = idf(len(plist), n_docs)
        if weight == 0.0:
            continue  # a term with df = N (or clamped to 0 by Okapi) discriminates nothing
        for doc_id, tf in plist.items():
            scores[doc_id] = (scores.get(doc_id, 0.0)
                              + weight * term_weight(tf, index.doc_lengths[doc_id],
                                                     index.avg_doc_length, k1, b))

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return ranked[:top_k]
