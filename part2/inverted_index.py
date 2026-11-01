"""Inverted index (Assignment 3.3.1).

One document is one Yelp review, and doc_id is the review_id. Terms come from the
`custom` method of the shared `text_processing` tokenizer (stop words kept, no
stemming), the same convention as part1.

The index is three dictionaries plus one number, with no further wrapping:

    index.postings[term][doc_id] = tf     how many times a term occurs in a given review
    index.doc_lengths[doc_id] = dl        how many terms a review has after tokenization
    index.doc_norms_lnc[doc_id] = ‖d‖     norm of the document vector under lnc weights (the cosine denominator)
    index.doc_norms_ltc[doc_id] = ‖d‖     norm of the document vector under ltc weights
    index.avg_doc_length = avgdl          corpus-wide average document length, computed once at index time

Two ‖d‖ tables are kept because TF-IDF has two classic weighting schemes, and their
document vectors — hence their norms — are different:

    lnc.ltc  document weight = 1 + ln tf (no idf)
    ltc.ltc  document weight = (1 + ln tf) · ln(N / df)

The cosine denominator must come from the same weighting as the numerator; take the
wrong one and every score is wrong. See part2/tfidf.py for details.

df (the number of documents containing a term) is not stored on its own: it is the
length of the postings list, computed directly by `index.df(term)`. Term positions are
not stored either: this assignment has no phrase queries, so they would never be used.

Building runs in two passes. The first tokenizes document by document, accumulating
postings and dl; the second computes ‖d‖ — the version that involves idf depends on the
full df table and can only run after the first pass ends. With ‖d‖ in hand, computing a
cosine at query time only means walking the query terms, never another full scan.

A complete build takes about 8 seconds, so the index is pickled to disk as soon as it is
built (`save_index`) and read back with `load_index` afterwards, which takes seconds.
The corpus file name, byte size, record count and tokenizer are recorded alongside it;
if any of them fails to match on load, an error prompts a rebuild, so you never run an
old index against a new corpus.
"""

import math
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from text_processing import get_tokenizer

DEFAULT_TOKENIZER = "custom"


def inverse_document_frequency(df, n_docs):
    """idf = ln(N / df), where N is the total number of documents and df the number containing the term.

    Returns 0 when df = 0 (the query term is absent from the index), so that term adds
    nothing to any document. When df = N (every document contains it), idf is 0 as well:
    the term discriminates nothing, so its weight is naturally 0.
    """
    return math.log(n_docs / df) if df else 0.0


def log_tf_weight(tf, idf=None):
    """Document-side lnc weight = 1 + ln tf, with no idf factor.

    The idf argument plays no part in the computation; it is there only so that this
    function has the same signature as tfidf_weight, letting the scoring loop call
    either one as weight(tf, idf). Applying idf once, on the query side, is enough:
    applying it in both places amounts to idf², which makes rare terms heavier than the
    design intends.
    """
    return 1.0 + math.log(tf)


def tfidf_weight(tf, idf):
    """TF-IDF term weight w = (1 + ln tf) · idf.

    We use 1 + ln tf rather than raw tf: a term that occurs 20 times should not carry 20
    times the weight of one that occurs once, and the sublinear scaling stops repeated
    terms in long documents from dominating. Computing ‖d‖ at index time and scoring at
    query time share this single definition, so any change only has to be made here.
    """
    return (1.0 + math.log(tf)) * idf


@dataclass
class InvertedIndex:
    """An inverted index. Its fields are the whole index; there is no hidden state."""

    postings: dict
    doc_lengths: dict
    doc_norms_lnc: dict
    doc_norms_ltc: dict
    avg_doc_length: float
    total_tokens: int
    tokenizer: str

    def df(self, term):
        """Number of documents containing term; 0 when term is absent from the index."""
        return len(self.postings.get(term, ()))


def compute_doc_norms(postings, doc_ids, weight, n_docs):
    """Second pass: accumulate ‖d‖² = Σ_t w(t, d)² posting by posting, then take the square root.

    weight decides which weighting scheme the norm is computed under: pass
    log_tf_weight for lnc and tfidf_weight for ltc. doc_ids is used to zero every
    document's accumulator up front, so empty documents (dl = 0) still appear in the
    result with ‖d‖ = 0; a scoring function that sees 0 can skip them and never divides
    by zero.
    """
    squared = {doc_id: 0.0 for doc_id in doc_ids}
    for plist in postings.values():
        idf = inverse_document_frequency(len(plist), n_docs)
        for doc_id, tf in plist.items():
            value = weight(tf, idf)
            squared[doc_id] += value * value
    return {doc_id: math.sqrt(value) for doc_id, value in squared.items()}


def build_inverted_index(reviews, tokenizer_name=DEFAULT_TOKENIZER):
    """Build an inverted index over a batch of reviews.

    reviews is any iterable whose items carry at least the fields review_id and text
    (part1's documents have exactly that shape). The return value can be handed
    straight to a scoring function: BM25 uses postings / doc_lengths /
    avg_doc_length, and the TF-IDF cosine additionally uses the two doc_norms tables.
    """
    tokenizer = get_tokenizer(tokenizer_name)
    postings = {}
    doc_lengths = {}
    total_tokens = 0
    for review in reviews:
        doc_id = review["review_id"]
        terms = tokenizer.tokenize(review["text"])
        doc_lengths[doc_id] = len(terms)
        total_tokens += len(terms)
        for term, tf in Counter(terms).items():
            postings.setdefault(term, {})[doc_id] = tf
    if not doc_lengths:
        raise ValueError("no documents were read, cannot build an index")
    n_docs = len(doc_lengths)
    return InvertedIndex(
        postings=postings,
        doc_lengths=doc_lengths,
        doc_norms_lnc=compute_doc_norms(postings, doc_lengths, log_tf_weight, n_docs),
        doc_norms_ltc=compute_doc_norms(postings, doc_lengths, tfidf_weight, n_docs),
        avg_doc_length=total_tokens / n_docs,
        total_tokens=total_tokens,
        tokenizer=tokenizer_name,
    )


def save_index(index, path, source=None):
    """Pickle the index to disk, returning the file size in bytes.

    source records where the index came from: corpus file name, byte size, record count
    and tokenizer. It is stored so those values can be checked when the index is read
    back; see load_index.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": dict(source or {}),
        "postings": index.postings,
        "doc_lengths": index.doc_lengths,
        "doc_norms_lnc": index.doc_norms_lnc,
        "doc_norms_ltc": index.doc_norms_ltc,
        "avg_doc_length": index.avg_doc_length,
        "total_tokens": index.total_tokens,
        "tokenizer": index.tokenizer,
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=4)
    return path.stat().st_size


def load_index(path, source=None):
    """Read an index back from disk.

    When source is supplied, its items are checked one by one first; a mismatch on any
    item means the corpus or the tokenizer has changed, so we raise straight away and
    let the caller rebuild, rather than carrying on with a stale index.
    """
    path = Path(path)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    expected = dict(source or {})
    saved = payload.get("source", {})
    mismatch = {key: (saved.get(key), value)
                for key, value in expected.items() if saved.get(key) != value}
    if mismatch:
        raise ValueError(f"index does not match the current corpus, rebuild it with --rebuild: {mismatch}")
    return InvertedIndex(
        postings=payload["postings"],
        doc_lengths=payload["doc_lengths"],
        doc_norms_lnc=payload["doc_norms_lnc"],
        doc_norms_ltc=payload["doc_norms_ltc"],
        avg_doc_length=payload["avg_doc_length"],
        total_tokens=payload["total_tokens"],
        tokenizer=payload["tokenizer"],
    )
