"""Behaviour of the BM25 parameters k1 and b on the real corpus.

Usage (from the repository root, with the index already built):

    python -m part2.param_behaviour

The grid is anchored on published recommendations:
    k1 in {0, 1.2, 1.6, 2.0, 3.0}: 1.2 and 2.0 bound the range recommended by
        Manning et al. (IIR 11.4.3) and Robertson & Zaragoza (2009, 3.5); 1.6 is its
        midpoint; 0 removes term frequency; 3.0 lies beyond the range.
    b in {0, 0.5, 0.75, 0.8, 1}: 0.5 and 0.8 bound the range of Robertson & Zaragoza;
        0.75 is the IIR value and the Lucene default; 0 and 1 are the extremes.

For every setting this script reports, over the 50 queries:
    agreement with the default (k1 = 1.2, b = 0.75): overlap@10 and Kendall tau-b
        over the union of the two top-50 lists (same measures as query_experiment);
    term-frequency profile of the top 10: mean tf of the matched query terms, and,
        for queries with at least two distinct indexed terms, the share of top-10
        documents that contain every distinct query term;
    length profile of the top 10: median dl / avgdl and the share of documents
        longer than avgdl.
It also writes the closed-form curves the two parameters imply. Without relevance
judgements none of this says which setting is better; it checks that k1 and b act
on term frequency and document length as the formula says.

Outputs: part2/results/param_behaviour.json and param_behaviour.md.
The existing grid in query_experiment.py is left unchanged.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from text_processing import get_tokenizer

from .bm25 import DEFAULT_B, DEFAULT_K1, bm25_search
from .queries import QUERIES
from .query_experiment import load_index_from_disk
from .rank_metrics import compare_pair

ROOT = Path(__file__).resolve().parents[1]

K1_VALUES = (0.0, 1.2, 1.6, 2.0, 3.0)
B_VALUES = (0.0, 0.5, 0.75, 0.8, 1.0)
POOL = 50
TOP = 10


def saturation_curve(k1, tfs):
    """Term-frequency factor at dl = avgdl divided by its limit k1 + 1: tf / (tf + k1).
    With k1 = 0 every matched term contributes the same (value 1)."""
    return [1.0 if k1 == 0 else tf / (tf + k1) for tf in tfs]


def length_curve(b, ratios, k1=DEFAULT_K1, tf=1):
    """Term-frequency factor for a fixed tf as a function of dl / avgdl."""
    return [tf * (k1 + 1) / (tf + k1 * (1 - b + b * r)) for r in ratios]


def top_profile(index, terms, top_ids):
    """tf and length profile of a list of top documents for the given query terms."""
    terms = [t for t in terms if t in index.postings]
    if not top_ids or not terms:
        return None
    tfs, full, ratios = [], [], []
    for doc_id in top_ids:
        matched = [index.postings[t][doc_id] for t in terms if doc_id in index.postings[t]]
        if matched:
            tfs.append(statistics.fmean(matched))
        full.append(len(matched) == len(terms))
        ratios.append(index.doc_lengths[doc_id] / index.avg_doc_length)
    return {"mean_tf": statistics.fmean(tfs) if tfs else None,
            "all_terms_share": sum(full) / len(full),
            "multi_term": len(terms) >= 2,
            "ratios": ratios}


def _mean(values):
    values = [v for v in values if v is not None]
    return round(statistics.fmean(values), 4) if values else None


def run(index, queries, k1_values=K1_VALUES, b_values=B_VALUES):
    tokenizer = get_tokenizer(index.tokenizer)
    n_docs = len(index.doc_lengths)
    settings = [(k1, b) for k1 in k1_values for b in b_values]
    if (DEFAULT_K1, DEFAULT_B) not in settings:
        settings.append((DEFAULT_K1, DEFAULT_B))
    rankings = {}
    for k1, b in settings:
        rankings[(k1, b)] = {q["id"]: bm25_search(index, q["text"], k1=k1, b=b, top_k=n_docs)
                             for q in queries}
    rows = []
    for k1, b in settings:
        per_query = []
        for q in queries:
            ranked = rankings[(k1, b)][q["id"]]
            default = rankings[(DEFAULT_K1, DEFAULT_B)][q["id"]]
            cmp = compare_pair(default, ranked, ks=(TOP, POOL), pool=POOL)
            terms = sorted(set(tokenizer.tokenize(q["text"])))
            prof = top_profile(index, terms, [d for d, _ in ranked[:TOP]])
            per_query.append({"id": q["id"], "overlap@10": cmp["overlap"][f"@{TOP}"],
                              "tau": cmp["tau_b_union"]["tau"] if cmp["tau_b_union"] else None,
                              "profile": prof})
        ratios = [r for p in per_query if p["profile"] for r in p["profile"]["ratios"]]
        rows.append({
            "k1": k1, "b": b, "is_default": (k1, b) == (DEFAULT_K1, DEFAULT_B),
            "overlap@10": _mean([p["overlap@10"] for p in per_query]),
            "tau_union": _mean([p["tau"] for p in per_query]),
            "top10_mean_tf": _mean([p["profile"]["mean_tf"] for p in per_query if p["profile"]]),
            "top10_all_terms_share": _mean([p["profile"]["all_terms_share"] for p in per_query
                                            if p["profile"] and p["profile"]["multi_term"]]),
            "n_multi_term_queries": sum(1 for p in per_query
                                        if p["profile"] and p["profile"]["multi_term"]),
            "top10_median_dl_ratio": round(statistics.median(ratios), 4) if ratios else None,
            "top10_share_longer_than_avgdl": _mean([1.0 if r > 1 else 0.0 for r in ratios]),
        })
    tfs = list(range(1, 21))
    dl_ratios = [0.25 * i for i in range(1, 17)]
    curves = {"tf": tfs,
              "saturation": {str(k1): [round(v, 4) for v in saturation_curve(k1, tfs)]
                             for k1 in k1_values},
              "dl_ratio": dl_ratios,
              "length": {str(b): [round(v, 4) for v in length_curve(b, dl_ratios)]
                         for b in b_values}}
    return rows, curves


def write_markdown(path, meta, rows):
    lines = ["# BM25 parameter behaviour (k1 x b)", "",
             f"Generated by part2/param_behaviour.py; {meta['n_queries']} queries, "
             f"N = {meta['n_docs']}, avgdl = {meta['avgdl']:.4f}, {meta['elapsed_seconds']} s.",
             "Agreement is against the default k1 = 1.2, b = 0.75 (tau-b over the union of "
             "the two top-50 lists). Profiles are over each setting's own top 10.", "",
             "| k1 | b | overlap@10 | tau union | mean tf of matched terms | share with all terms "
             "(multi-term queries) | median dl/avgdl | share dl > avgdl |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        tag = " (default)" if r["is_default"] else ""
        lines.append(f"| {r['k1']} | {r['b']}{tag} | {r['overlap@10']} | {r['tau_union']} | "
                     f"{r['top10_mean_tf']} | {r['top10_all_terms_share']} | "
                     f"{r['top10_median_dl_ratio']} | {r['top10_share_longer_than_avgdl']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/meridian_id")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "part2/results")
    parser.add_argument("--index-file", type=Path, default=ROOT / "part2/.cache/index.pkl")
    parser.add_argument("--tokenizer", default="custom")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    index = load_index_from_disk(args.index_file, args.input_dir, args.tokenizer)
    rows, curves = run(index, QUERIES)
    meta = {"n_queries": len(QUERIES), "n_docs": len(index.doc_lengths),
            "avgdl": index.avg_doc_length, "k1_values": list(K1_VALUES),
            "b_values": list(B_VALUES),
            "elapsed_seconds": round(time.perf_counter() - started, 1)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "param_behaviour.json").write_text(
        json.dumps({"meta": meta, "grid": rows, "curves": curves}, indent=2), encoding="utf-8")
    write_markdown(args.output_dir / "param_behaviour.md", meta, rows)
    print(f"Done in {meta['elapsed_seconds']}s")


if __name__ == "__main__":
    main()