"""Query experiment runner: 50 queries x several retrieval conditions -> JSON / CSV / Markdown.

Usage (from the repository root, with the index already built):

    python -m part2.query_experiment     # full run: 50 queries x 6 base + 16-cell grid
    python -m part2.query_experiment --skip-grid     # base conditions only
    python -m part2.query_experiment --queries Q01,Q19,Q46   # only the named queries
    python -m part2.query_experiment --axes edge     # only one axis

Outputs (written to part2/results/ by default):
    query_experiment_runs.json       raw artifacts: top-50 and latency per condition per query
    query_experiment_summary.json    metrics: per-query + aggregates (by axis, by df band)
    query_experiment_summary.md      human-readable tables
    query_experiment_overlap.csv     long-format table for plotting

This script is the comparative half of the 3.3.2 analysis. Section 3.3.2 asks us to
"Run a set of at least five diverse queries on **both implementations**"; we read that
as two complementary questions: "how does each implementation behave on its own" and
"how and why do the two differ". The former is
covered by impl_analysis.py (a per-implementation profile, which can tell whether the
implementation behaves correctly without any qrels). This script does the latter: it
explains **why the two implementations diverge and which variables drive that
divergence**.

It measures ranking agreement, so **without relevance judgments it cannot say which
implementation is better** -- that is its ceiling, not a flaw.

Experiment groups (defined one by one below; the README table is exactly these)
-----------------------------------------------------------------------------
Each experiment group is a "paired comparison of two retrieval conditions" that differ
in **exactly one respect** (see the note on each group), so that any difference in
overlap / tau can be attributed to that single factor:

    E1  divergence-mechanism analysis of the two implementations: TF-IDF cosine vs BM25
    E2  within TF-IDF: whether the document side carries idf (lnc.ltc vs ltc.ltc)
    E3  within BM25: Okapi idf vs the lecture-slide idf
    E4  within BM25: Okapi idf vs the Lucene idf
    E5  within BM25: lecture-slide idf vs Lucene idf (the two should almost overlap)
    E6  clamping ablation: clamped vs unclamped Okapi (passed in through the idf_fn argument; only
        queries containing negative-idf terms move)
    E7  parameter sensitivity: a 4x4 k1 x b grid, every cell compared against the
        default (1.2, 0.75)

There are only two measures (rank_metrics.py documents them in full): overlap@k looks at
set overlap, Kendall tau-b at order agreement. There are no relevance judgments, so
P@10 / MAP / NDCG are simply not computed -- if they cannot be computed honestly, we do
not pretend to compute them.
"""

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

from .bm25 import (
    COURSE,
    DEFAULT_B,
    DEFAULT_K1,
    LUCENE,
    OKAPI,
    bm25_search,
    okapi_idf_unclamped,
)
from .inverted_index import load_index
from .queries import QUERIES, profile_query
from .rank_metrics import compare_pair, self_check
from .tfidf import LNC_LTC, LTC_LTC, tfidf_search

ROOT = Path(__file__).resolve().parents[1]

# Deep pool: take the top POOL documents on each side; overlap@50 and tau are computed
# over its union / intersection.
POOL = 50
KS = (5, 10, 20, 50)


# ------------------------------------------------------------- retrieval conditions

def _run_tfidf(index, text, top_k, variant):
    return tfidf_search(index, text, top_k=top_k, variant=variant)


def _run_bm25(index, text, top_k, variant=OKAPI, k1=DEFAULT_K1, b=DEFAULT_B,
              idf_fn=None):
    return bm25_search(index, text, top_k=top_k, variant=variant, k1=k1, b=b,
                       idf_fn=idf_fn)


def base_conditions():
    """The six base conditions: both implementations plus their variant ablations."""
    return [
        {"name": "tfidf_lnc_ltc", "family": "tfidf",
         "label": "TF-IDF cosine lnc.ltc",
         "note": "document side uses 1+ln(tf) without idf (the lecture version, and the "
                 "Manning textbook default)",
         "run": lambda index, text, k: _run_tfidf(index, text, k, LNC_LTC)},
        {"name": "tfidf_ltc_ltc", "family": "tfidf",
         "label": "TF-IDF cosine ltc.ltc",
         "note": "document side also carries idf (the version most textbooks show in the "
                 "main text; the numerator is effectively idf squared)",
         "run": lambda index, text, k: _run_tfidf(index, text, k, LTC_LTC)},
        {"name": "bm25_okapi", "family": "bm25",
         "label": "BM25 okapi (k1=1.2, b=0.75)",
         "note": "idf = max(0, ln((N-df+0.5)/(df+0.5))), deliverable configuration, "
                 "default parameters",
         "run": lambda index, text, k: _run_bm25(index, text, k, OKAPI)},
        {"name": "bm25_course", "family": "bm25",
         "label": "BM25 course",
         "note": "idf = ln(N/df), verbatim from Lecture 9.1 slides, the same idf as "
                 "TF-IDF",
         "run": lambda index, text, k: _run_bm25(index, text, k, COURSE)},
        {"name": "bm25_lucene", "family": "bm25",
         "label": "BM25 lucene",
         "note": "idf = ln(1+(N-df+0.5)/(df+0.5)), the idf of Lucene's "
                 "BM25Similarity (Lucene 8 onward, including 10.5.2)",
         "run": lambda index, text, k: _run_bm25(index, text, k, LUCENE)},
        {"name": "bm25_okapi_unclamped", "family": "bm25",
         "label": "BM25 okapi unclamped (experimental interface)",
         "note": "passed in through the idf_fn argument, idf may be negative; used only to "
                 "quantify the effect of clamping, not part of the deliverable",
         "run": lambda index, text, k: _run_bm25(index, text, k, OKAPI,
                                                 idf_fn=okapi_idf_unclamped)},
    ]


def grid_conditions(k1_values=(0.0, 0.9, 1.2, 2.0), b_values=(0.0, 0.3, 0.75, 1.0)):
    """The k1 x b parameter grid (the assignment asks us to "correctly handle the
    parameters k1 and b").

    Every cell is compared against the default (1.2, 0.75), so exactly one parameter
    dimension differs at a time. k1=0 is a degenerate point (the term-frequency
    saturation factor is constant 1, leaving only a sum over idf), and b=0 disables
    length normalization entirely.
    """
    return [
        {"name": f"bm25_k1_{k1}_b_{b}", "family": "bm25_grid",
         "label": f"BM25 okapi k1={k1}, b={b}", "note": f"parameter grid: k1={k1}, b={b}",
         "k1": k1, "b": b, "is_default": (k1 == DEFAULT_K1 and b == DEFAULT_B),
         "run": lambda index, text, k, _k1=k1, _b=b: _run_bm25(
             index, text, k, OKAPI, k1=_k1, b=_b)}
        for k1 in k1_values for b in b_values
    ]


def group_pairs():
    """An experiment group is a pair of conditions. Each pair differs in exactly one
    respect; see the note on each group."""
    pairs = [
        ("E1", "tfidf_lnc_ltc", "bm25_okapi",
         "Cross-implementation comparison (the per-implementation profile is in "
         "impl_analysis.py): the difference is the whole scoring function"),
        ("E2", "tfidf_lnc_ltc", "tfidf_ltc_ltc",
         "TF-IDF internal ablation: whether the document side carries idf; everything "
         "else identical"),
        ("E3", "bm25_okapi", "bm25_course",
         "BM25 internal ablation: swap only the idf (clamped Okapi vs slide ln(N/df))"),
        ("E4", "bm25_okapi", "bm25_lucene",
         "BM25 internal ablation: swap only the idf (clamped Okapi vs Lucene +1 form)"),
        ("E5", "bm25_course", "bm25_lucene",
         "BM25 internal ablation: swap only the idf (slide vs Lucene; expected to "
         "almost overlap)"),
        ("E6", "bm25_okapi", "bm25_okapi_unclamped",
         "Clamping ablation: remove only max(0, .); expected to move only queries "
         "containing terms with df>N/2"),
    ]
    return pairs


# ------------------------------------------------------------------ batch execution

def run_condition(index, condition, query_text, top_k):
    """Run one condition once, returning (ranking, elapsed milliseconds)."""
    started = time.perf_counter()
    ranked = condition["run"](index, query_text, top_k)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return ranked, round(elapsed_ms, 2)


def run_query(index, conditions, query):
    """One query x all conditions, keeping the **full ranking** in memory for the
    metrics.

    Each condition runs twice, and the two passes serve different purposes -- do not
    collapse them into one:

        full ranking   top_k = the whole corpus, giving every candidate document its
                       true rank. tau compares ranks over the union of the two sides'
                       top-50, so if we only kept the first 50 results, a document that
                       "missed the top 50 on this side but made it on the other" would
                       have no rank here and tau could not be computed at all (a plain
                       KeyError).
        top-10         only 10 results; the latency of this pass is what can honestly be
                       reported as "per-query latency" -- real retrieval asks for the top
                       10 documents, nobody fully ranks all 37,401 for a query.
    """
    n_docs = len(index.doc_lengths)
    out = {}
    for condition in conditions:
        ranked_full, ms_full = run_condition(index, condition, query["text"], n_docs)
        _, ms_10 = run_condition(index, condition, query["text"], 10)
        out[condition["name"]] = {"ranked": ranked_full, "ms": ms_10,
                                  "ms_full_sort": ms_full}
    return out


def pair_metrics_for_query(query_runs, pairs, ks=KS, pool=POOL):
    """Compute all experiment groups' metrics for one query in one pass (the full
    rankings are still in memory at this point)."""
    return {group: compare_pair(query_runs[name_a]["ranked"],
                                query_runs[name_b]["ranked"], ks=ks, pool=pool)
            for group, name_a, name_b, _note in pairs}


# ------------------------------------------------------------------ aggregation

def _mean(values):
    values = [v for v in values if v is not None]
    return round(statistics.fmean(values), 4) if values else None


def aggregate(group, queries, ks=KS):
    """Collapse one experiment group's per-query metrics into a few rows: one set by
    axis, one set by df band."""
    per_query = group["per_query"]
    rows = []

    def summarize(label, ids):
        subset = [per_query[qid] for qid in ids if qid in per_query]
        if not subset:
            return None
        row = {"slice": label, "n_queries": len(subset)}
        for k in ks:
            row[f"overlap@{k}"] = _mean([m["overlap"][f"@{k}"] for m in subset])
        row["tau_union"] = _mean([m["tau_b_union"]["tau"] if m["tau_b_union"] else None
                                  for m in subset])
        row["tau_intersection"] = _mean(
            [m["tau_b_intersection"]["tau"] if m["tau_b_intersection"] else None
             for m in subset])
        row["top1_same_rate"] = _mean([1.0 if m["top1_same"] else 0.0
                                       for m in subset])
        row["n_intersection"] = _mean([m["n_intersection"] for m in subset])
        return row

    rows.append(summarize("all queries", [q["id"] for q in queries]))

    axes = {}
    for query in queries:
        axes.setdefault(query["axis"], []).append(query["id"])
    for axis, ids in sorted(axes.items()):
        row = summarize(f"axis={axis}", ids)
        if row:
            rows.append(row)

    bands = {}
    for query in queries:
        bands.setdefault(query.get("df_band", "n/a"), []).append(query["id"])
    for band, ids in sorted(bands.items()):
        row = summarize(f"band={band}", ids)
        if row:
            rows.append(row)

    return [row for row in rows if row]


def condition_summary(runs, conditions, queries):
    """Per-condition profile: mean latency, mean number of returned documents, median
    document length among the top 10."""
    rows = []
    for condition in conditions:
        name = condition["name"]
        per_query = runs[name]
        returned = [per_query[q["id"]]["n_returned"] for q in queries]
        ms = [per_query[q["id"]]["ms"] for q in queries]
        dls = [per_query[q["id"]]["top10_dl_median"] for q in queries
               if per_query[q["id"]]["top10_dl_median"] is not None]
        rows.append({"condition": name, "label": condition["label"],
                     "note": condition["note"],
                     "mean_ms": _mean(ms), "max_ms": round(max(ms), 2),
                     "mean_n_returned": _mean(returned),
                     "median_top10_dl": round(statistics.median(dls), 1) if dls else None,
                     "n_empty_results": sum(1 for n in returned if n == 0)})
    return rows


# ---------------------------------------------------------------------- output

def _fmt_tau(metrics_union, metrics_intersection):
    """tau is always printed in Markdown to 4 decimal places; write n/a when it cannot
    be computed."""
    union = metrics_union["tau"] if metrics_union else None
    intersection = metrics_intersection["tau"] if metrics_intersection else None
    return (f"{union:.4f}" if union is not None else "n/a",
            f"{intersection:.4f}" if intersection is not None else "n/a")


def write_markdown(path, meta, profiles, condition_rows, groups, aggregates,
                   grid_rows):
    """Write the human-readable summary tables."""
    lines = ["# Query Experiment Results", ""]
    lines += [
        f"- Corpus: {meta['reviews']} reviews, tokenizer {meta['tokenizer']}, "
        f"avgdl={meta['avgdl']}",
        f"- Queries: {meta['n_queries']} (the assignment requires >=5; the empirical "
        f"floor in IIR Section 8.1 is 50 information needs)",
        f"- Deep pool: top {meta['pool']} per side; overlap@k k = "
        + ", ".join(str(k) for k in meta["ks"]),
        f"- Ranking tiebreak: {meta['tiebreak']}",
        "- Metrics: overlap@k (set overlap, denominator fixed at k) + Kendall tau-b "
        "(order agreement, reported separately over the union and the intersection)",
        "",
    ]

    lines += ["## 1. Retrieval conditions", "",
              "| condition | note | mean ms | max ms | mean returned | "
              "median top-10 dl | empty results |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in condition_rows:
        lines.append(
            f"| `{row['condition']}` | {row['note']} | {row['mean_ms']} | "
            f"{row['max_ms']} | {row['mean_n_returned']} | {row['median_top10_dl']} | "
            f"{row['n_empty_results']} |")
    lines.append("")

    lines += ["## 2. Experiment groups and aggregated results", ""]
    for group, agg_rows in aggregates.items():
        if group.startswith("E7::"):
            continue  # the 16 parameter-grid cells are not expanded here; they all go
                      # into the compact table in section 3
        note = groups[group]["note"]
        name_a, name_b = groups[group]["pair"]
        lines += [f"### {group}: `{name_a}` vs `{name_b}`", "",
                  f"Differs only in: {note}", "",
                  "| slice | queries | " +
                  " | ".join(f"overlap@{k}" for k in meta["ks"]) +
                  " | tau union | tau intersection | top-1 same rate |",
                  "|---|---:|" + "---:|" * (len(meta["ks"]) + 3)]
        for row in agg_rows:
            cells = [str(row[f"overlap@{k}"]) for k in meta["ks"]]
            lines.append(
                f"| {row['slice']} | {row['n_queries']} | " + " | ".join(cells) +
                f" | {row['tau_union']} | {row['tau_intersection']} | "
                f"{row['top1_same_rate']} |")
        lines.append("")

    if grid_rows:
        lines += ["## 3. Parameter sensitivity grid (k1 x b, all compared against the "
                  "default k1=1.2 / b=0.75)", "",
                  "| k1 | b | overlap@10 | tau union | top-1 same rate | note |",
                  "|---:|---:|---:|---:|---:|---|"]
        for row in grid_rows:
            lines.append(
                f"| {row['k1']} | {row['b']} | {row['overlap@10']} | "
                f"{row['tau_union']} | {row['top1_same_rate']} | {row['note']} |")
        lines.append("")

    lines += ["## 4. Per-query profiles", "",
              "| query | text | axis | terms | df range | df band | candidate "
              "documents |",
              "|---|---|---|---|---|---|---:|"]
    for query, profile in profiles:
        dfs = [str(item["df"]) for item in profile["term_df"]]
        lines.append(
            f"| {query['id']} | {query['text']} | {query['axis']} | "
            f"{profile['n_unique_terms']} | {' / '.join(dfs)} | {profile['df_band']} | "
            f"{profile['n_candidates']} |")
    lines.append("")

    lines += ["## 5. E1 (the two implementations) per-query detail", "",
              "| query | text | axis | df band | overlap@5 | overlap@10 | overlap@20 | "
              "overlap@50 | tau union | tau intersection | top-1 same | TF-IDF missed | "
              "BM25 missed |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for qid, metrics in groups["E1"]["per_query"].items():
        tau_u, tau_i = _fmt_tau(metrics["tau_b_union"], metrics["tau_b_intersection"])
        lines.append(
            f"| {qid} | {metrics['text']} | {metrics['axis']} | {metrics['df_band']} | "
            f"{metrics['overlap']['@5']} | {metrics['overlap']['@10']} | "
            f"{metrics['overlap']['@20']} | {metrics['overlap']['@50']} | "
            f"{tau_u} | {tau_i} | {'yes' if metrics['top1_same'] else 'no'} | "
            f"{metrics['n_missing_a']} | {metrics['n_missing_b']} |")
    lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path, groups, ks=KS):
    """Long-format table: group x query x k -> overlap; plotting reads this directly."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "query_id", "text", "axis", "df_band", "k",
                         "overlap", "tau_union", "tau_intersection", "top1_same"])
        for group, payload in groups.items():
            for qid, metrics in payload["per_query"].items():
                tau_u = metrics["tau_b_union"]["tau"] if metrics["tau_b_union"] else ""
                tau_i = (metrics["tau_b_intersection"]["tau"]
                         if metrics["tau_b_intersection"] else "")
                for k in ks:
                    writer.writerow([group, qid, metrics["text"], metrics["axis"],
                                     metrics["df_band"], k, metrics["overlap"][f"@{k}"],
                                     tau_u, tau_i,
                                     "1" if metrics["top1_same"] else "0"])


# ------------------------------------------------------------------------- main

def load_index_from_disk(index_file, input_dir, tokenizer):
    """Load the index written to disk; a corpus/tokenizer mismatch raises immediately,
    so we never run a new corpus against a stale index."""
    import json as _json
    reviews_path = input_dir / "review.json"
    manifest = _json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    source = {"file": reviews_path.name, "bytes": reviews_path.stat().st_size,
              "reviews": manifest["files"]["review.json"]["records"],
              "tokenizer": tokenizer}
    return load_index(index_file, source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/meridian_id")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "part2/results")
    parser.add_argument("--index-file", type=Path, default=ROOT / "part2/.cache/index.pkl")
    parser.add_argument("--tokenizer", default="custom")
    parser.add_argument("--pool", type=int, default=POOL)
    parser.add_argument("--ks", type=int, nargs="+", default=list(KS))
    parser.add_argument("--queries", default="",
                        help="run only these query ids, comma-separated")
    parser.add_argument("--axes", default="",
                        help="run only these axes, comma-separated")
    parser.add_argument("--skip-grid", action="store_true",
                        help="skip the k1/b parameter grid")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet
    index = load_index_from_disk(args.index_file, args.input_dir, args.tokenizer)

    queries = QUERIES
    if args.queries:
        wanted = {q.strip() for q in args.queries.split(",") if q.strip()}
        queries = [q for q in queries if q["id"] in wanted]
    if args.axes:
        wanted = {a.strip() for a in args.axes.split(",") if a.strip()}
        queries = [q for q in queries if q["axis"] in wanted]
    if not queries:
        raise SystemExit("no query selected; check --queries / --axes")

    # Profiles: tokenization, df and candidate counts per query. Measured, not hardcoded.
    profiles = []
    for query in queries:
        profile = profile_query(index, query["text"])
        query["df_band"] = profile["df_band"]
        profiles.append((query, profile))

    if verbose:
        print(f"tau self-check: {self_check()}")
        print(f"{6 + (0 if args.skip_grid else 16)} conditions x {len(queries)} queries")

    conditions = base_conditions()
    grid = [] if args.skip_grid else grid_conditions()
    pairs = group_pairs()
    for condition in grid:
        if condition["is_default"]:
            continue  # the default-parameter cell is bm25_okapi itself; comparing it
                      # against itself is pointless
        pairs.append((f"E7::{condition['name']}", "bm25_okapi", condition["name"],
                      f"parameter sensitivity: change only k1={condition['k1']} / "
                      f"b={condition['b']}"))

    all_conditions = conditions + grid
    runs = {condition["name"]: {} for condition in all_conditions}
    groups = {group: {"pair": [name_a, name_b], "note": note, "per_query": {}}
              for group, name_a, name_b, note in pairs}

    # Run query by query: the full ranking lives in memory for just one round; once the
    # metrics are computed only the top POOL entries are kept for output.
    started = time.perf_counter()
    for position, query in enumerate(queries, 1):
        query_runs = run_query(index, all_conditions, query)
        qid = query["id"]
        for name, payload in query_runs.items():
            ranked = payload["ranked"]
            runs[name][qid] = {
                "top": [[doc_id, round(score, 6)] for doc_id, score
                        in ranked[:args.pool]],
                "n_returned": len(ranked),
                "ms": payload["ms"],
                "ms_full_sort": payload["ms_full_sort"],
                "top10_dl_median": (
                    statistics.median([index.doc_lengths[doc_id]
                                       for doc_id, _ in ranked[:10]])
                    if ranked else None),
            }
        for group, metrics in pair_metrics_for_query(
                query_runs, pairs, ks=tuple(args.ks), pool=args.pool).items():
            metrics["axis"] = query["axis"]
            metrics["text"] = query["text"]
            metrics["df_band"] = query.get("df_band", "")
            name_a, name_b = groups[group]["pair"]
            metrics["n_returned_a"] = runs[name_a][qid]["n_returned"]
            metrics["n_returned_b"] = runs[name_b][qid]["n_returned"]
            groups[group]["per_query"][qid] = metrics
        if verbose and position % 10 == 0:
            print(f"  ... {position}/{len(queries)} queries")
    elapsed = time.perf_counter() - started

    aggregates = {group: aggregate(groups[group], queries, ks=tuple(args.ks))
                  for group in groups}
    condition_rows = condition_summary(runs, conditions, queries)

    grid_rows = []
    for condition in grid:
        group = groups.get(f"E7::{condition['name']}")
        row = aggregate(group, queries, ks=tuple(args.ks))[0] if group else None
        if row:
            grid_rows.append({"k1": condition["k1"], "b": condition["b"],
                              "overlap@10": row.get("overlap@10"),
                              "tau_union": row["tau_union"],
                              "top1_same_rate": row["top1_same_rate"],
                              "note": ("default parameters (the same condition as "
                                       "bm25_okapi)" if condition["is_default"] else "")})
        elif condition["is_default"]:
            grid_rows.append({"k1": condition["k1"], "b": condition["b"],
                              "overlap@10": "—", "tau_union": "—",
                              "top1_same_rate": "—",
                              "note": "default parameters; compared against itself this "
                                      "is trivially 10/10, tau=1"})

    meta = {
        "generated_by": "python -m part2.query_experiment",
        "reviews": len(index.doc_lengths),
        "tokenizer": index.tokenizer,
        "avgdl": round(index.avg_doc_length, 4),
        "n_queries": len(queries),
        "query_axes": {axis: sum(1 for q in queries if q["axis"] == axis)
                       for axis in sorted({q["axis"] for q in queries})},
        "pool": args.pool,
        "ks": args.ks,
        "tiebreak": "score descending, ties broken by doc_id ascending",
        "elapsed_seconds": round(elapsed, 1),
        "citation_query_count": (
            "IIR (Manning/Raghavan/Schütze 2008) §8.1: 'As a rule of thumb, 50 "
            "information needs has usually been found to be a sufficient minimum.'"),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "query_experiment_runs.json").write_text(
        json.dumps({"meta": meta, "runs": runs}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8")
    (args.output_dir / "query_experiment_summary.json").write_text(
        json.dumps({"meta": meta,
                    "profiles": [dict(query, profile=profile)
                                 for query, profile in profiles],
                    "conditions": condition_rows,
                    "groups": {g: {"pair": groups[g]["pair"], "note": groups[g]["note"],
                                   "aggregate": aggregates[g],
                                   "per_query": groups[g]["per_query"]}
                               for g in groups},
                    "grid": grid_rows},
                   ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8")
    write_csv(args.output_dir / "query_experiment_overlap.csv", groups,
              ks=tuple(args.ks))
    write_markdown(args.output_dir / "query_experiment_summary.md", meta, profiles,
                   condition_rows, groups, aggregates, grid_rows)

    if verbose:
        print(f"\nDone in {elapsed:.1f}s; results written to {args.output_dir}")
        for group in ["E1", "E2", "E3", "E4", "E5", "E6"]:
            if group not in aggregates:
                continue
            row = aggregates[group][0]
            print(f"{group} {groups[group]['pair']}: overlap@10={row['overlap@10']} "
                  f"overlap@50={row['overlap@50']} tau_union={row['tau_union']} "
                  f"top1_same={row['top1_same_rate']}")


if __name__ == "__main__":
    main()
