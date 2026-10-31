"""Per-implementation analysis: a behavioural profile of each of the two
delivered implementations across the 50 queries.

Why this script exists (and how the work is split with query_experiment.py)
--------------------------------------------------------------------------
Assignment 3.3.2 says "Run a set of at least five diverse queries on **both
implementations**" — what it asks for is that **each** of the two
implementations be run and analysed on its own; it does **not** ask for a
head-to-head contest between them. Hence:

    this script  = the per-implementation half: run each implementation and answer
                   "does this implementation behave correctly?"
    query_       = the comparative half: put the two implementations side by side and
    experiment     answer "why do they disagree?". There are no relevance
                   judgments, so what it measures is ranking agreement — it
                   **cannot** tell which one is better.

Every metric computed here needs **no relevance judgments**, yet each speaks
directly to whether the implementation behaves the way its formula predicts:

    n_returned      results returned (n_returned). BM25 clamping down to 0 can
                    make a whole query return nothing, and this column catches
                    that.
    margin_rel      relative margin between the top-1 and the second result,
                    (s1−s2)/|s1|. Near 0 means the winner did not really win —
                    it was picked arbitrarily out of a pile of ties, so
                    changing the tiebreak changes the top-1.
    n_tied_top1     documents tied with the top-1 result. Same story: the more
                    of them, the less you can trust the top-1.
    decay_10        s10/s1, the score concentration across the top ten. Near 1
                    means the scores barely separate.
    length_bias     median length of the top-10 documents ÷ corpus mean length.
                    1.0 = biased neither long nor short; >1 favours long
                    documents, <1 favours short ones. Used to verify that
                    normalization really is in effect.
    ms              latency (ms) for retrieving the top 10 of a single query.

On top of that, the `length_bias` column is backed by a **b sweep** (k1 fixed
at 1.2, b = 0 / 0.3 / 0.75 / 1.0): if length normalization is genuinely doing
its job, the median length of the top-10 should decrease monotonically with b.
That expectation is **falsifiable** — if it fails, something is wrong with the
implementation.

How to run (from the repository root, with the index already built):

    python -m part2.impl_analysis                    # all 50 queries
    python -m part2.impl_analysis --queries Q01,Q45  # just a few
    python -m part2.impl_analysis --quiet

Outputs (part2/results/):
    impl_analysis.md    three tables: profile / b sweep / spot-check of
                        representative queries
    impl_analysis.json  raw per-query metrics, for plotting or re-checking
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from .bm25 import DEFAULT_B, DEFAULT_K1, OKAPI, bm25_search
from .inverted_index import load_index
from .queries import QUERIES, df_band, profile_query
from .tfidf import LNC_LTC, tfidf_search

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "part2" / "results"
CORPUS = ROOT / "data" / "meridian_id" / "review.json"

# Representative queries used for the spot-check: one from each of the five
# axes, plus the df=0 one.
SPOTLIGHT = ["Q01", "Q25", "Q35", "Q43", "Q45", "Q50"]
SPOTLIGHT_TOPN = 5
SNIPPET_CHARS = 120

# Tie threshold: a relative difference within 1e-9 counts as a tie (doubles all
# the way through, so this only screens out floating-point noise).
TIE_REL_TOL = 1e-9


# ------------------------------------------------------------ conditions

def analysis_conditions():
    """The conditions profiled here: the two delivered implementations plus a
    BM25 b sweep (b=0.75 coincides with the delivered configuration)."""
    conditions = [
        {"name": "tfidf_lnc_ltc", "label": "TF-IDF cosine lnc.ltc",
         "note": "delivered implementation 1: 1+ln(tf) on the document side, "
                 "cosine normalization",
         "run": lambda index, text, k: tfidf_search(index, text, top_k=k,
                                                    variant=LNC_LTC)},
    ]
    for b in (0.0, 0.3, 0.75, 1.0):
        is_default = abs(b - DEFAULT_B) < 1e-12
        conditions.append({
            "name": f"bm25_b_{b}",
            "label": (f"BM25 k1={DEFAULT_K1}, b={b}"
                      + (" (delivered configuration)" if is_default else "")),
            "note": f"b sweep of delivered implementation 2: b={b}",
            "b": b, "is_default": is_default,
            "run": lambda index, text, k, _b=b: bm25_search(
                index, text, top_k=k, variant=OKAPI, k1=DEFAULT_K1, b=_b),
        })
    return conditions


# --------------------------------------------------------- per-query metrics

def query_metrics(index, condition, query, texts):
    """Run one query and compute every behavioural metric of this condition on
    that query."""
    n_docs = len(index.doc_lengths)
    started = time.perf_counter()
    ranked_full = condition["run"](index, query["text"], n_docs)
    ms_full = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    condition["run"](index, query["text"], 10)
    ms = (time.perf_counter() - started) * 1000

    top10 = ranked_full[:10]
    scores = [score for _doc, score in top10]
    s1 = scores[0] if scores else None
    s2 = scores[1] if len(scores) > 1 else None
    s10 = scores[-1] if len(scores) >= 10 else None

    n_tied = 0
    if s1 is not None:
        tol = TIE_REL_TOL * max(1.0, abs(s1))
        n_tied = sum(1 for _doc, score in ranked_full if abs(score - s1) <= tol)

    dls = [index.doc_lengths[doc_id] for doc_id, _ in top10]
    return {
        "text": query["text"],
        "axis": query["axis"],
        "df_band": df_band(profile_query(index, query["text"])["df_max"]),
        "n_returned": len(ranked_full),
        "top1_score": s1,
        "top2_score": s2,
        "margin_rel": ((s1 - s2) / abs(s1)) if s1 and s2 is not None else None,
        "n_tied_top1": n_tied,
        "decay_10": (s10 / s1) if s1 and s10 is not None and s1 else None,
        "top10_dl_median": statistics.median(dls) if dls else None,
        "top10_dl_mean": round(statistics.fmean(dls), 1) if dls else None,
        "length_bias": (round(statistics.median(dls) / index.avg_doc_length, 4)
                        if dls else None),
        "ms": round(ms, 2),
        "ms_full_sort": round(ms_full, 2),
        "top": [{"doc": doc_id, "score": round(score, 6),
                 "dl": index.doc_lengths[doc_id],
                 # the source text may contain pipes; unescaped they break the
                 # Markdown table
                 "snippet": (texts.get(doc_id) or "").replace("\n", " ")
                            .replace("|", r"\|")[:SNIPPET_CHARS]}
                for doc_id, score in top10],
    }


def run_all(index, conditions, queries, texts, verbose=True):
    """Run the full condition x query grid."""
    runs = {}
    total = len(conditions) * len(queries)
    done = 0
    for condition in conditions:
        per_query = {}
        for query in queries:
            per_query[query["id"]] = query_metrics(index, condition, query, texts)
            done += 1
            if verbose and done % 50 == 0:
                print(f"  ... {done}/{total}")
        runs[condition["name"]] = per_query
    return runs


# ------------------------------------------------------------- aggregation

BAND_ORDER = ["df=0", "rare", "low", "medium", "high", "very high"]
NUMERIC_KEYS = ["n_returned", "margin_rel", "n_tied_top1", "decay_10",
                "top10_dl_median", "length_bias", "ms"]


def _median(values):
    values = [v for v in values if v is not None]
    return round(statistics.median(values), 4) if values else None


def aggregate(per_query, queries, slice_by):
    """Aggregate along one dimension, slice by slice.
    slice_by: 'all' | 'axis' | 'df_band'."""
    profiles = {query["id"]: per_query[query["id"]] for query in queries}
    if slice_by == "all":
        groups = {"all": [query["id"] for query in queries]}
    else:
        groups = {}
        for query in queries:
            key = profiles[query["id"]][slice_by]
            groups.setdefault(key, []).append(query["id"])
    rows = []
    for key, qids in groups.items():
        row = {"slice": key, "n_queries": len(qids)}
        for metric in NUMERIC_KEYS:
            row[metric] = _median([profiles[qid][metric] for qid in qids])
        rows.append(row)
    if slice_by == "df_band":
        # slice names look like "1000 <= df < 3000 (medium)", so sort by the
        # band name inside the parentheses; otherwise the row order of a table
        # would differ between runs.
        def band_rank(name):
            short = name.split("(")[-1].rstrip(")") if "(" in name else name
            return BAND_ORDER.index(short) if short in BAND_ORDER else 99

        rows.sort(key=lambda row: band_rank(row["slice"]))
    elif slice_by == "axis":
        rows.sort(key=lambda row: row["slice"])
    return rows


def build_profiles(runs, queries, condition_names):
    """{condition name: {'all': [...], 'axis': [...], 'df_band': [...]}}"""
    out = {}
    for name in condition_names:
        out[name] = {key: aggregate(runs[name], queries, key)
                     for key in ("all", "axis", "df_band")}
    return out


def b_scan(profiles):
    """b sweep: length_bias and the median top-10 length should decrease
    monotonically with b."""
    rows = []
    for name in profiles:
        if not name.startswith("bm25_b_"):
            continue
        row = profiles[name]["all"][0]
        rows.append({"b": float(name.split("_")[-1]), "label": name,
                     "length_bias": row["length_bias"],
                     "top10_dl_median": row["top10_dl_median"],
                     "margin_rel": row["margin_rel"], "ms": row["ms"]})
    rows.sort(key=lambda row: row["b"])
    biases = [row["length_bias"] for row in rows if row["length_bias"] is not None]
    monotonic = all(biases[i] > biases[i + 1] for i in range(len(biases) - 1))
    return {"rows": rows, "monotonic_decreasing": monotonic}


def notable(runs, queries, condition_names):
    """Queries worth calling out individually: no hits, ties at top-1, and
    scores that barely separate."""
    out = {}
    for name in condition_names:
        zero_hit, tied, flat = [], [], []
        for query in queries:
            metrics = runs[name][query["id"]]
            if metrics["n_returned"] == 0:
                zero_hit.append({"id": query["id"], "text": query["text"]})
            if metrics["n_tied_top1"] and metrics["n_tied_top1"] > 1:
                tied.append({"id": query["id"], "text": query["text"],
                             "n_tied_top1": metrics["n_tied_top1"]})
            if metrics["decay_10"] is not None and metrics["decay_10"] > 0.95:
                flat.append({"id": query["id"], "text": query["text"],
                             "decay_10": metrics["decay_10"]})
        out[name] = {"zero_hit": zero_hit, "tied_top1": tied, "flat_top10": flat}
    return out


# ------------------------------------------------------------------- output

def write_markdown(path, meta, profiles, scan, spotlight, anomalies, conditions):
    labels = {c["name"]: c["label"] for c in conditions}
    deliverables = [c["name"] for c in conditions
                    if c["name"] == "tfidf_lnc_ltc" or c.get("is_default")]
    metric_cols = [("results returned", "n_returned"),
                   ("top-1 relative margin", "margin_rel"),
                   ("ties at top-1", "n_tied_top1"),
                   ("s10/s1", "decay_10"),
                   ("top-10 median length", "top10_dl_median"),
                   ("length bias (÷ avgdl)", "length_bias"),
                   ("latency (ms)", "ms")]

    def fmt(value):
        # one number style for every row: integral values without ".0"
        if value is None:
            return "—"
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return f"{value}"

    lines = ["# Per-implementation analysis (behavior of each implementation "
             "on the 50 queries, assignment 3.3.2)", "",
             f"Corpus of {meta['n_docs']:,} documents, mean length "
             f"{meta['avgdl']}, {meta['n_queries']} queries, "
             f"{meta['generated_at']}.", "",
             "What this section answers is whether **each implementation "
             "behaves correctly on its own** — not which of the two is better. "
             "Without relevance judgments (qrels), that cannot be measured.",
             "",
             "Column meanings: *results returned* = documents with a non-zero "
             "score; *top-1 relative margin* = (s1 - s2) / s1; *ties at top-1* = "
             "documents sharing the top score; *s10/s1* = score of the 10th hit "
             "divided by the 1st; *length bias* = top-10 median length ÷ avgdl. "
             "Each cell is the median over the queries in the slice; the slice "
             "groups are separated by a label row.",
             ""]

    group_titles = {"all": "**all queries**", "axis": "**by axis**",
                    "df_band": "**by df band**"}
    for name in deliverables:
        lines += [f"## {labels[name]}", "",
                  "| slice | queries | " + " | ".join(c for c, _k in metric_cols) + " |",
                  "|---|---:|" + "---:|" * len(metric_cols)]
        for key in ("all", "axis", "df_band"):
            if key != "all":
                lines.append(f"| {group_titles[key]} |" + " |" * (len(metric_cols) + 1))
            for row in profiles[name][key]:
                cells = [fmt(row[k]) for _c, k in metric_cols]
                lines.append(f"| {row['slice']} | {row['n_queries']} | "
                             + " | ".join(cells) + " |")
        lines.append("")

    lines += ["## b sweep: is length normalization really penalizing long "
              "documents?", "",
              "k1 is fixed at 1.2; only b moves. The falsifiable expectation "
              "is: **the larger b gets, the smaller the median document length "
              "of the top-10 becomes**.", "",
              "| b | top-10 median length | length bias (÷ corpus mean) | "
              "top-1 relative margin | latency (ms) |",
              "|---:|---:|---:|---:|---:|"]
    for row in scan["rows"]:
        lines.append(f"| {row['b']} | {row['top10_dl_median']} | "
                     f"{row['length_bias']} | {row['margin_rel']} | {row['ms']} |")
    lines += ["",
              f"Monotonically decreasing: "
              f"{'**yes**, matches the expectation' if scan['monotonic_decreasing'] else '**no**, refuted, needs investigating'}.",
              ""]

    lines += ["## Spot-check of representative queries (top-5, relevance judged "
              "by eye)", "",
              "For each query we list the top 5 documents returned by each "
              "implementation, together with a snippet of the source text. "
              "**This is the only place where you can see directly whether the "
              "retrieved results are actually right**, and it relies on human "
              "judgment rather than on metrics.", ""]
    for qid in spotlight:
        lines += [f"### {qid}", ""]
        for name in deliverables:
            metrics = spotlight[qid][name]
            lines += [f"**{labels[name]}** "
                      f"({metrics['n_returned']:,} results returned):", "",
                      "| # | document | score | length | snippet |",
                      "|---:|---|---:|---:|---|"]
            for position, item in enumerate(metrics["top"][:SPOTLIGHT_TOPN], 1):
                snippet = item["snippet"] or "(no text)"
                lines.append(f"| {position} | `{item['doc'][:12]}…` | "
                             f"{item['score']} | {item['dl']} | {snippet} |")
            lines.append("")

    lines += ["## Anomalies that need separate comment", ""]
    for name in deliverables:
        info = anomalies[name]
        lines += [f"### {labels[name]}", ""]
        lines.append(f"- **zero hits** (0 results returned): "
                     + (", ".join(f"`{x['id']}` {x['text']}" for x in info["zero_hit"])
                        or "none"))
        lines.append(f"- **ties at top-1** (the top-1 result is not the unique "
                     f"highest score): "
                     + (", ".join(f"`{x['id']}` ({x['n_tied_top1']} documents tied)"
                                  for x in info["tied_top1"][:8])
                        or "none"))
        lines.append(f"- **scores barely separated** (s10/s1 > 0.95): "
                     + (", ".join(f"`{x['id']}`" for x in info["flat_top10"][:8])
                        or "none"))
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- main

def load_texts():
    texts = {}
    with CORPUS.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            texts[record["review_id"]] = record.get("text") or ""
    return texts


def main(argv=None):
    parser = argparse.ArgumentParser(description="Per-implementation analysis")
    parser.add_argument("--queries", help="run only the given queries, "
                                          "comma-separated, e.g. Q01,Q45")
    parser.add_argument("--skip-run", action="store_true",
                        help="impl_analysis.json already exists, only "
                             "regenerate the Markdown")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    verbose = not args.quiet
    RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS / "impl_analysis.json"
    md_path = RESULTS / "impl_analysis.md"

    queries = QUERIES
    if args.queries:
        wanted = {item.strip() for item in args.queries.split(",")}
        queries = [q for q in QUERIES if q["id"] in wanted]

    if verbose:
        print(f"Loading corpus text ({CORPUS.name})...")
    texts = load_texts()

    source = {"file": "review.json", "bytes": CORPUS.stat().st_size,
              "reviews": len(texts), "tokenizer": "custom"}
    index = load_index(ROOT / "part2" / ".cache" / "index.pkl", source)
    if verbose:
        print(f"Index: {len(index.doc_lengths):,} documents, "
              f"mean length {index.avg_doc_length}")

    conditions = analysis_conditions()
    names = [c["name"] for c in conditions]

    if verbose:
        print(f"Running {len(conditions)} conditions x {len(queries)} queries...")
    started = time.perf_counter()
    runs = run_all(index, conditions, queries, texts, verbose)
    elapsed = time.perf_counter() - started

    profiles = build_profiles(runs, queries, names)
    scan = b_scan(profiles)
    anomalies = notable(runs, queries, names)
    spotlight = {qid: {name: runs[name][qid] for name in names}
                 for qid in SPOTLIGHT if qid in runs[names[0]]}

    payload = {
        "meta": {"n_docs": len(index.doc_lengths),
                 "avgdl": index.avg_doc_length,
                 "n_queries": len(queries),
                 "conditions": [{"name": c["name"], "label": c["label"],
                                 "note": c["note"]} for c in conditions],
                 "elapsed_seconds": round(elapsed, 1),
                 "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")},
        "profiles": profiles, "b_scan": scan,
        "anomalies": anomalies, "spotlight": spotlight,
        "per_query": runs,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    write_markdown(md_path, payload["meta"], profiles, scan, spotlight,
                   anomalies, conditions)

    if verbose:
        print(f"Done in {elapsed:.1f}s -> {md_path}")
        print(f"b sweep monotonically decreasing: {scan['monotonic_decreasing']}")
        for row in scan["rows"]:
            print(f"  b={row['b']}: length bias {row['length_bias']}, "
                  f"top-10 median length {row['top10_dl_median']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
