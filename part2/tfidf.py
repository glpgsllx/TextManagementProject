"""TF-IDF + cosine retrieval (Assignment 3.3.2), with two variants: lnc.ltc and ltc.ltc.

Written in SMART triple notation as ddd.qqq: the triple on the left describes how the
**document vector** is weighted, the triple on the right how the **query vector** is
weighted (that is the order Manning's textbook uses). One letter per position:

    position 1 (term frequency)      n = raw tf count   l = 1 + ln(tf)
    position 2 (idf or not)          n = do not apply   t = ln(N / df)
    position 3 (normalization)       n = none           c = cosine (divide the vector by its own norm ‖·‖)

So the two variants differ only in whether the document side is multiplied by idf:

    lnc.ltc   document weight = 1 + ln(tf), query weight = (1 + ln tf) · ln(N / df)
              idf appears exactly once along the whole path. This is the scheme taught
              in class, and the default combination recommended by Manning. A side
              benefit: the document vector is independent of collection-wide df, so
              ‖d‖ is fixed for any given document and never has to be recomputed when
              new documents arrive and some term's df changes (friendly to incremental
              indexing).

    ltc.ltc   document weight = (1 + ln tf) · ln(N / df), query weight as above
              This is the literal reading of "build every document into a TF-IDF
              vector, then take the cosine against the query vector", and the version
              of the TF-IDF cosine printed in the body of most textbooks. The price is
              that the same idf is applied once on the document and once on the query,
              which amounts to idf² in the numerator, so rare terms are amplified twice.

Because the two document vectors differ, the cosine denominator ‖d‖ differs too; both
are computed at index time and stored on the index object as doc_norms_lnc /
doc_norms_ltc. Query time does no full-collection scan: for each query term it walks
that term's posting list once and accumulates (query weight / ‖q‖) · (document weight /
‖d‖) into that document's score (that is, term-at-a-time), so only documents matching
at least one query term appear among the candidate documents.

Edge cases: idf = ln(N / df) is 0 when df = N (words like "and" or "the" that occur in
every document automatically get weight 0), and df = 0 (term absent from the index) is
treated as 0, so an obscure word in the query adds nothing to any document and we never
have to evaluate ln(N / 0).
"""

import math
from collections import Counter

from text_processing import get_tokenizer

from .inverted_index import (
    inverse_document_frequency,
    log_tf_weight,
    tfidf_weight,
)

LNC_LTC = "lnc.ltc"
LTC_LTC = "ltc.ltc"


def query_weights(index, query_terms):
    """Query-side ltc weight: for each query term, w = (1 + ln tf_q) · ln(N / df).

    The query side is identical for both variants; the difference lies only on the
    document side. When a term is absent from the index, df = 0, idf is 0, and so is
    the weight — such terms can simply be dropped at search time, since they contribute
    0 to every document.
    """
    n_docs = len(index.doc_lengths)
    return {
        term: tfidf_weight(tf, inverse_document_frequency(index.df(term), n_docs))
        for term, tf in Counter(query_terms).items()
    }


def tfidf_search(index, query_text, top_k=10, variant=LNC_LTC):
    """Run a natural-language query, returning [(doc_id, score)]; scores are cosine values, in descending order.

    query_text arrives as raw text and is tokenized internally with the same tokenizer
    that was used at index time, so queries and documents travel through the same
    preprocessing pipeline. variant selects "lnc.ltc" (the default) or "ltc.ltc".
    """
    if variant == LNC_LTC:
        doc_norms, doc_weight = index.doc_norms_lnc, log_tf_weight
    elif variant == LTC_LTC:
        doc_norms, doc_weight = index.doc_norms_ltc, tfidf_weight
    else:
        raise ValueError(f"unsupported weighting scheme: {variant}, available: {LNC_LTC} or {LTC_LTC}")

    terms = get_tokenizer(index.tokenizer).tokenize(query_text)
    weights = {term: w for term, w in query_weights(index, terms).items() if w}
    if not weights:
        return []
    query_norm = math.sqrt(sum(weight * weight for weight in weights.values()))

    n_docs = len(index.doc_lengths)
    scores = {}
    for term, query_weight in weights.items():
        idf = inverse_document_frequency(index.df(term), n_docs)
        for doc_id, tf in index.postings.get(term, {}).items():
            norm = doc_norms[doc_id]
            if norm == 0.0:
                continue  # empty document: the cosine denominator is 0, no similarity to speak of
            scores[doc_id] = (
                scores.get(doc_id, 0.0) + query_weight * doc_weight(tf, idf) / norm
            )

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    # Dividing by ‖q‖ is what makes this a true cosine; it is the same factor for every
    # document, so it does not change the ranking, only how readable the values are.
    return [(doc_id, score / query_norm) for doc_id, score in ranked[:top_k]]
