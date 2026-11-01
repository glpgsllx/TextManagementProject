"""Command-line entry point for a single query: try a query ad hoc and see what the two
implementations put at the top.

    python -m part2.search_cli "gluten free pancakes"
    python -m part2.search_cli "pizza" --model bm25 --variant course --top-k 5
    python -m part2.search_cli "terrible service" --all          # both implementations side by side
    python -m part2.search_cli "best pizza in town" --no-snippet # just doc_id and score

By default it prints the document id, score, document length and the beginning of the
review text. To read a review in full, look its doc_id up in the corpus; long text is
not printed here, and pasting complete result sets into the report is not allowed
either.
"""

import argparse
import json
from pathlib import Path

from .bm25 import COURSE, DEFAULT_B, DEFAULT_K1, LUCENE, OKAPI, bm25_search
from .inverted_index import load_index
from .query_experiment import load_index_from_disk
from .tfidf import LNC_LTC, LTC_LTC, tfidf_search

ROOT = Path(__file__).resolve().parents[1]


def load_review_text(path):
    """doc_id -> review text. Read only when snippets are needed; one pass takes about
    two seconds."""
    texts = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            review = json.loads(line)
            texts[review["review_id"]] = review["text"]
    return texts


def search(index, text, model, variant, k1, b, top_k):
    if model == "tfidf":
        return tfidf_search(index, text, top_k=top_k,
                            variant=LNC_LTC if variant is None else variant)
    if model == "bm25":
        return bm25_search(index, text, top_k=top_k,
                           variant=OKAPI if variant is None else variant, k1=k1, b=b)
    raise SystemExit(f"unrecognized model: {model}")


def show(label, index, ranked, texts, snippet):
    print(f"--- {label}: {len(ranked)} hits ---")
    for position, (doc_id, score) in enumerate(ranked, 1):
        line = f"{position:2d}. {doc_id}  score={score:.4f}  dl={index.doc_lengths[doc_id]}"
        if snippet and texts:
            body = " ".join(texts.get(doc_id, "").split())[:90]
            line += f"  | {body}"
        print(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="the raw query text")
    parser.add_argument("--model", default="tfidf", choices=["tfidf", "bm25"])
    parser.add_argument("--variant", default=None,
                        help=f"tfidf: {LNC_LTC} / {LTC_LTC}; "
                             f"bm25: {OKAPI} / {COURSE} / {LUCENE}")
    parser.add_argument("--k1", type=float, default=DEFAULT_K1)
    parser.add_argument("--b", type=float, default=DEFAULT_B)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--all", action="store_true",
                        help="run both implementations and show them side by side")
    parser.add_argument("--no-snippet", action="store_true",
                        help="skip reading the corpus and hide the review text")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/meridian_id")
    parser.add_argument("--index-file", type=Path, default=ROOT / "part2/.cache/index.pkl")
    parser.add_argument("--tokenizer", default="custom")
    args = parser.parse_args()

    index = load_index_from_disk(args.index_file, args.input_dir, args.tokenizer)
    texts = {} if args.no_snippet else load_review_text(args.input_dir / "review.json")

    if args.all:
        show(f"TF-IDF {LNC_LTC}", index,
             tfidf_search(index, args.query, top_k=args.top_k, variant=LNC_LTC),
             texts, not args.no_snippet)
        show(f"TF-IDF {LTC_LTC}", index,
             tfidf_search(index, args.query, top_k=args.top_k, variant=LTC_LTC),
             texts, not args.no_snippet)
        show(f"BM25 {OKAPI} (k1={args.k1}, b={args.b})", index,
             bm25_search(index, args.query, top_k=args.top_k, variant=OKAPI,
                         k1=args.k1, b=args.b),
             texts, not args.no_snippet)
        return

    ranked = search(index, args.query, args.model, args.variant, args.k1, args.b,
                    args.top_k)
    label = f"{args.model} {args.variant or ('lnc.ltc' if args.model == 'tfidf' else OKAPI)}"
    if args.model == "bm25":
        label += f" (k1={args.k1}, b={args.b})"
    show(label, index, ranked, texts, not args.no_snippet)


if __name__ == "__main__":
    main()
