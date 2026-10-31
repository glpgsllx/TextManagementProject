"""Lucene baseline comparison: real Apache Lucene 10.5.2 vs our two hand-written implementations.

The closing sentence of assignment 3.3.2 reads:

    "You may also run the same queries and compare your results with those
     returned by a standard library such as Lucene, e.g., the built-in
     ranking modules."

That sentence is an optional bonus item, so this module leaves every existing
experiment group in `query_experiment.py` untouched and instead **adds** a
separate set, E8-E11, used as a third-party reference column.

The four Lucene configurations we run (two indexes x two similarities)
---------------------------------------------------------------------
    ours_tok   Feeds Lucene the tokens produced by our own tokenizer
               (WhitespaceAnalyzer simply splits on whitespace and applies no
               further processing). The token sequences are identical on both
               sides, so whatever difference remains is purely the **scoring
               implementation**: the IDF formula, length normalization, norm
               quantization, floating-point precision.
    std        StandardAnalyzer over the raw text, i.e. Lucene out of the box.
               Its difference from ours_tok is exactly the contribution of the
               **tokenizer** (including the English stop-word list).

    On each index we run BM25Similarity(1.2, 0.75) and ClassicSimilarity
    (Lucene's built-in TF-IDF).

Why the tokenizer layer is isolated
-----------------------------------
The earlier `data/lucene_probe` run issued only 6 queries and re-tokenized with
StandardAnalyzer, so "formula differences" and "tokenizer differences" were
tangled together and an overlap of 4/10 could not be attributed to either side.
Here both sides consume exactly the same tokens, so the "hand-written vs
Lucene" figure can be attributed directly to the formula implementation.

How to run
----------
    python -m part2.lucene_experiment                  # full: export corpus -> compile/run Java -> compare
    python -m part2.lucene_experiment --skip-run       # Java already ran, only recompute the comparison
    python -m part2.lucene_experiment --queries Q01,Q45

Requires a JDK (auto-detected under `C:\\Program Files\\Java\\jdk-*`) and the
three Lucene 10.5.2 jars (default `C:/tmp/lucene-probe/lib`). Output is written
to `part2/results/lucene_comparison.*`.
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

from .bm25 import COURSE, DEFAULT_B, DEFAULT_K1, LUCENE, OKAPI, bm25_search
from .inverted_index import load_index
from .queries import QUERIES
from .rank_metrics import compare_pair
from .tfidf import LNC_LTC, tfidf_search

ROOT = Path(__file__).resolve().parents[1]
JAVA_SRC = Path(__file__).resolve().parent / "lucene" / "LuceneCompare.java"

POOL = 50          # deep pool, same as query_experiment.py
KS = (5, 10, 20, 50)
LUCENE_TOPK = 200  # ask Lucene for extra depth, re-sort by tiebreak, then keep the top POOL
EXPLAIN_DEPTH = 10  # explain output for the top 10 documents per query (recovers the dl Lucene actually used)

OUR_CONDITIONS = {
    "tfidf_lnc_ltc": lambda index, text, k: tfidf_search(index, text, top_k=k,
                                                         variant=LNC_LTC),
    "bm25_okapi": lambda index, text, k: bm25_search(index, text, top_k=k,
                                                    variant=OKAPI),
    "bm25_course": lambda index, text, k: bm25_search(index, text, top_k=k,
                                                     variant=COURSE),
    "bm25_lucene": lambda index, text, k: bm25_search(index, text, top_k=k,
                                                     variant=LUCENE),
}

# E8-E11: each group again narrows the difference down to a single factor
GROUP_PAIRS = [
    ("E8", "bm25_lucene", "lucene_bm25_ours_tok",
     "Hand-written replication of Lucene's IDF vs real Lucene BM25: same tokens, "
     "same k1/b, only the fidelity of the replication is under test"),
    ("E9", "bm25_okapi", "lucene_bm25_ours_tok",
     "Deliverable BM25 (Okapi clamped IDF + (k1+1)) vs real Lucene BM25: same "
     "tokens, differing in IDF and in the constant"),
    ("E10", "tfidf_lnc_ltc", "lucene_classic_ours_tok",
     "Our TF-IDF cosine lnc.ltc vs Lucene's built-in ClassicSimilarity: same "
     "tokens, two different formulas"),
    ("E11", "lucene_bm25_ours_tok", "lucene_bm25_std",
     "Lucene BM25 on both sides, only the tokenizer is swapped (our custom vs "
     "Lucene StandardAnalyzer)"),
]


# ---------------------------------------------------------------- environment

def find_java_home():
    """Locate a JDK: check JAVA_HOME first, then scan Program Files for jdk-*."""
    import os
    env = os.environ.get("JAVA_HOME")
    if env and (Path(env) / "bin" / "javac.exe").exists():
        return Path(env)
    for base in (Path("C:/Program Files/Java"), Path("D:/Program Files/Java")):
        if not base.exists():
            continue
        candidates = sorted([p for p in base.iterdir()
                             if p.is_dir() and p.name.startswith("jdk")],
                            reverse=True)
        for candidate in candidates:
            if (candidate / "bin" / "javac.exe").exists():
                return candidate
    return None


def find_lucene_lib():
    for candidate in (Path("C:/tmp/lucene-probe/lib"),
                      Path("C:/tmp/lucene-probe"),
                      ROOT / "part2/lucene/lib"):
        if candidate.exists() and list(candidate.glob("lucene-core-*.jar")):
            return candidate
    return None


# ---------------------------------------------------------------- corpus export

def export_corpus(input_dir: Path, work_dir: Path, tokenizer_name: str, verbose=True):
    """Export ids.txt / texts_tok.txt / texts_raw.txt.

    The tokens written to texts_tok.txt must match the index tokens **exactly,
    one for one**; otherwise the "same tokens" premise collapses and none of
    the downstream differences can be attributed. We therefore re-tokenize with
    the very tokenizer that was used when the index was built.
    """
    from text_processing import get_tokenizer

    work_dir.mkdir(parents=True, exist_ok=True)
    ids_path = work_dir / "ids.txt"
    tok_path = work_dir / "texts_tok.txt"
    raw_path = work_dir / "texts_raw.txt"
    if ids_path.exists() and tok_path.exists() and raw_path.exists():
        if verbose:
            print(f"Corpus already exported, skipping: {work_dir}")
        return ids_path, tok_path, raw_path

    tokenizer = get_tokenizer(tokenizer_name)
    reviews_path = input_dir / "review.json"
    started = time.perf_counter()
    written = 0
    with (open(ids_path, "w", encoding="utf-8", newline="\n") as ids_out,
          open(tok_path, "w", encoding="utf-8", newline="\n") as tok_out,
          open(raw_path, "w", encoding="utf-8", newline="\n") as raw_out):
        for line in open(reviews_path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record["text"]
            tokens = tokenizer.tokenize(text)
            # collapse newlines/tabs into spaces: Lucene reads one document per line, the format must stay clean
            flat = re.sub(r"[\r\n\t]+", " ", text)
            tokens_flat = [re.sub(r"[\r\n\t]+", " ", t) for t in tokens]
            tokens_flat = [t for t in tokens_flat if t and not t.isspace()]
            ids_out.write(record["review_id"] + "\n")
            tok_out.write(" ".join(tokens_flat) + "\n")
            raw_out.write(flat + "\n")
            written += 1
    if verbose:
        print(f"Exported {written} documents to {work_dir} in {time.perf_counter() - started:.1f}s")
    return ids_path, tok_path, raw_path


def export_queries(work_dir: Path, queries, tokenizer_name: str):
    """queries.tsv: qid \\t raw query \\t our tokens."""
    from text_processing import get_tokenizer

    tokenizer = get_tokenizer(tokenizer_name)
    path = work_dir / "queries.tsv"
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for query in queries:
            tokens = [t for t in tokenizer.tokenize(query["text"])
                      if t and not t.isspace()]
            handle.write(f"{query['id']}\t{query['text']}\t{' '.join(tokens)}\n")
    return path


# ---------------------------------------------------------------- run Java

def run_java(work_dir: Path, out_path: Path, java_home: Path, lucene_lib: Path,
             verbose=True):
    classes = work_dir / "classes"
    classes.mkdir(parents=True, exist_ok=True)
    javac = java_home / "bin" / "javac.exe"
    java = java_home / "bin" / "java.exe"
    classpath = str(lucene_lib / "*")
    classpath = classpath.replace("\\", "/")

    if verbose:
        print(f"Compiling {JAVA_SRC.name} ...")
    subprocess.run(
        [str(javac), "-encoding", "UTF-8", "-nowarn", "-cp", classpath,
         "-d", str(classes), str(JAVA_SRC)],
        check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")

    if verbose:
        print("Building the index and searching (slowest step, roughly 1-3 minutes)...")
    started = time.perf_counter()
    result = subprocess.run(
        [str(java), "-Xmx4g", "-Dfile.encoding=UTF-8", "-Dstdout.encoding=UTF-8",
         "-cp", f"{classpath};{classes}",
         "LuceneCompare", str(work_dir), str(out_path), str(LUCENE_TOPK),
         str(EXPLAIN_DEPTH)],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        sys.stderr.write(result.stdout[-4000:] + "\n" + result.stderr[-4000:] + "\n")
        raise SystemExit(f"Lucene probe failed with exit code {result.returncode}")
    if verbose:
        print(f"Lucene finished in {time.perf_counter() - started:.1f}s")
    return out_path


# ---------------------------------------------------------------- read back and align

def load_lucene(path: Path):
    """Read the Lucene output into {config: {query id: [(doc_id, score), ...]}}
    and re-sort it with our own tiebreak (score descending, ties broken by
    ascending doc_id).

    Why re-sort: Lucene breaks score ties by internal doc id (= insertion
    order) while we break them by ascending review_id string. If we did not
    unify this, the ordering *inside* a tie block would be counted as a ranking
    difference and would inflate tau.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    runs = {cfg: {} for cfg in
            ["lucene_bm25_ours_tok", "lucene_classic_ours_tok",
             "lucene_bm25_std", "lucene_classic_std"]}
    meta = {"topK": payload["topK"], "indexes": payload["indexes"],
            "lucene": payload["lucene"], "java": payload.get("java")}
    for query in payload["queries"]:
        qid = query["id"]
        for cfg in runs:
            block = query["results"].get(cfg, {})
            ranked = [(item["id"], item["score"]) for item in block.get("top", [])]
            ranked.sort(key=lambda item: (-item[1], item[0]))
            runs[cfg][qid] = {"ranked": ranked, "totalHits": block.get("totalHits", 0),
                              "explain": block.get("explain", [])}
    return runs, meta


def lucene_dl_from_explain(explain):
    """Pull out of the raw explain text the dl Lucene actually scored with (it is flagged as approximate)."""
    match = re.search(r"([0-9]+(?:\.0)?) = dl, length of field", explain or "")
    if not match:
        return None
    return float(match.group(1))


def score_check(index, lucene_runs, queries, verbose=True):
    """Score-level check: recompute Lucene's BM25 score from the tf / df / avgdl in our own index.

    This is the hardest piece of evidence for the question "can our hand-written
    code stand in for Lucene". Attribution runs in three steps, and the residual
    left by one step is exactly what the next step must explain:

        Step 1  Recompute with **the true dl we computed** -> residual A.
        Step 2  Recompute with **the dl Lucene reports in explain** -> residual B.
        Step 3  Residual B should hold nothing but floating-point error (Lucene
                is float32 end to end, we are float64 end to end), on the order
                of 1e-7; if B ~ 0, then IDF, tf saturation, length
                normalization, N and avgdl are understood by us **term for
                term** exactly as Lucene implements them.
        Residual A - residual B = the loss caused by Lucene quantizing document
                length (the norm stores an approximate length).

    We additionally measure the "dl quantization loss" itself: the relative
    difference between the dl Lucene uses and the true dl.
    """
    import math
    from text_processing import get_tokenizer

    tokenizer = get_tokenizer(index.tokenizer)
    n_docs = len(index.doc_lengths)
    rows = []
    duplicate_checks = []
    for query in queries:
        qid = query["id"]
        tokenizer = get_tokenizer(index.tokenizer)
        tokens_all = tokenizer.tokenize(query["text"])
        terms = set(tokens_all)
        per_term = []
        for term in terms:
            plist = index.postings.get(term)
            if not plist:
                continue
            df = len(plist)
            per_term.append((term, plist, math.log(1 + (n_docs - df + 0.5) / (df + 0.5)),
                             tokens_all.count(term)))
        if not per_term:
            continue
        has_duplicate = len(tokens_all) != len(terms)
        block = lucene_runs["lucene_bm25_ours_tok"][qid]
        for position, (doc_id, actual) in enumerate(block["ranked"][:10]):
            dl = index.doc_lengths[doc_id]
            explains = block.get("explain", [])
            dl_lucene = (lucene_dl_from_explain(explains[position])
                         if position < len(explains) else None)

            def score_with(length, weighted=False):
                total = 0.0
                for _term, plist, idf, multiplicity in per_term:
                    tf = plist.get(doc_id)
                    if not tf:
                        continue
                    total += idf * tf / (tf + DEFAULT_K1 * (
                        1 - DEFAULT_B + DEFAULT_B * length / index.avg_doc_length)) \
                        * (multiplicity if weighted else 1)
                return total

            exact = score_with(dl)
            as_lucene = score_with(dl_lucene) if dl_lucene else None
            rows.append({
                "query": qid, "doc": doc_id, "lucene": actual,
                "dl_ours": dl, "dl_lucene": dl_lucene,
                "pred_dl_exact": exact, "pred_dl_lucene": as_lucene,
                "rel_err_exact": abs(exact - actual) / actual if actual else None,
                "rel_err_lucene_dl": (abs(as_lucene - actual) / actual
                                      if as_lucene is not None and actual else None),
                "dl_rel_loss": (abs(dl - dl_lucene) / dl
                                if dl_lucene is not None and dl else None),
            })
            if has_duplicate and position == 0 and as_lucene is not None:
                # duplicate-term query: also compute the un-deduplicated score to see whether it is exactly doubled
                weighted = score_with(dl_lucene, weighted=True)
                duplicate_checks.append({
                    "query": qid, "text": query["text"], "doc": doc_id,
                    "tokens": tokens_all,
                    "score_dedup": as_lucene, "score_with_duplicates": weighted,
                    "lucene_actual": actual,
                    "ratio": weighted / as_lucene if as_lucene else None})

    def summarize(key):
        values = [row[key] for row in rows if row[key] is not None]
        if not values:
            return None
        values = sorted(values)
        return {"n": len(values), "median": round(statistics.median(values), 10),
                "p90": round(values[int(len(values) * 0.9)], 10),
                "max": round(values[-1], 10)}

    out = {"n_compared": len(rows),
           "rel_err_dl_exact": summarize("rel_err_exact"),
           "rel_err_dl_lucene_actual": summarize("rel_err_lucene_dl"),
           "dl_quantization_loss": summarize("dl_rel_loss"),
           "duplicate_term_queries": duplicate_checks,
           "sample": rows[:8]}
    if verbose:
        print(f"Score-level check: compared {len(rows)} (query, document) pairs")
        print(f"  residual A (recomputed with the true dl)        : {out['rel_err_dl_exact']}")
        print(f"  residual B (recomputed with Lucene's actual dl) : {out['rel_err_dl_lucene_actual']}")
        print(f"  relative dl quantization loss                   : {out['dl_quantization_loss']}")
    return out


# ---------------------------------------------------------------- metrics

def compute_groups(index, our_runs, lucene_runs, queries, ks=KS, pool=POOL):
    """Compute the per-query metrics of E8-E11. Our conditions give a full ranking, Lucene gives top-K."""
    groups = {group: {"pair": [name_a, name_b], "note": note, "per_query": {}}
              for group, name_a, name_b, note in GROUP_PAIRS}
    for query in queries:
        qid = query["id"]
        for group, name_a, name_b, _note in GROUP_PAIRS:
            ranked_a = (our_runs[name_a][qid] if name_a in our_runs
                        else lucene_runs[name_a][qid]["ranked"])
            ranked_b = (our_runs[name_b][qid] if name_b in our_runs
                        else lucene_runs[name_b][qid]["ranked"])
            metrics = compare_pair(ranked_a, ranked_b, ks=ks, pool=pool)
            metrics["axis"] = query["axis"]
            metrics["text"] = query["text"]
            metrics["df_band"] = query.get("df_band", "")
            groups[group]["per_query"][qid] = metrics
    return groups


def _mean(values):
    values = [v for v in values if v is not None]
    return round(statistics.fmean(values), 4) if values else None


def aggregate(group, queries, ks=KS):
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
        row["top1_same_rate"] = _mean([1.0 if m["top1_same"] else 0.0 for m in subset])
        rows.append(row)

    summarize("all queries", [q["id"] for q in queries])
    axes = {}
    for query in queries:
        axes.setdefault(query["axis"], []).append(query["id"])
    for axis in sorted(axes):
        summarize(f"axis={axis}", axes[axis])
    return rows


def write_markdown(path, meta, groups, aggregates, checks, index_meta):
    lines = ["# Lucene baseline comparison (the \"compare with Lucene\" item of assignment 3.3.2)", "",
             f"- Lucene {meta['lucene']} / Java {meta['java']}; we ask each configuration for the top "
             f"{meta['topK']} hits and then re-sort them with our own tiebreak",
             f"- Our implementation: {index_meta['reviews']} documents, tokenizer "
             f"{index_meta['tokenizer']}, avgdl={index_meta['avgdl']}", ""]

    lines += ["## 1. Statistics of the two Lucene indexes", "",
              "| index | analyzer | numDocs | sumTotalTermFreq | avgdl |",
              "|---|---|---:|---:|---:|"]
    for name, stats in meta["indexes"].items():
        lines.append(f"| `{name}` | {stats['analyzer']} | {stats['numDocs']} | "
                     f"{stats['sumTotalTermFreq']} | {round(stats['avgdl'], 4)} |")
    lines.append("")

    lines += ["## 2. Experiment groups and aggregate results", ""]
    for group, rows in aggregates.items():
        payload = groups[group]
        lines += [f"### {group}: `{payload['pair'][0]}` vs `{payload['pair'][1]}`", "",
                  f"Difference confined to: {payload['note']}", "",
                  "| slice | queries | " + " | ".join(f"overlap@{k}" for k in KS) +
                  " | tau (union) | tau (intersection) | top-1 agreement rate |",
                  "|---|---:|" + "---:|" * (len(KS) + 3)]
        for row in rows:
            cells = [str(row[f"overlap@{k}"]) for k in KS]
            lines.append(f"| {row['slice']} | {row['n_queries']} | " +
                         " | ".join(cells) +
                         f" | {row['tau_union']} | {row['tau_intersection']} | "
                         f"{row['top1_same_rate']} |")
        lines.append("")

    lines += ["## 3. Score-level check (BM25, ours_tok index, top 10 documents of each query)", "",
              "We recompute Lucene's BM25 score from the tf / df / avgdl held in our own index "
              "and compare it with the score Lucene actually returned.",
              "Attribution runs in two steps: residual A uses the true dl we computed, residual B "
              "uses the dl Lucene reports in its explain output;",
              "A - B is therefore the loss caused by Lucene quantizing document length.", "",
              "| quantity | pairs compared | median | p90 | max |",
              "|---|---:|---:|---:|---:|"]
    for label, key in [("residual A: recomputed with the true dl", "rel_err_dl_exact"),
                       ("residual B: recomputed with Lucene's actual dl", "rel_err_dl_lucene_actual"),
                       ("relative loss from quantizing dl itself", "dl_quantization_loss")]:
        stats = checks[key]
        if stats:
            lines.append(f"| {label} | {stats['n']} | {stats['median']} | "
                         f"{stats['p90']} | {stats['max']} |")
    lines.append("")
    lines += ["Sample (first 8):", "",
              "| query | Lucene score | recomputed (true dl) | recomputed (Lucene dl) | true dl | Lucene dl |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in checks["sample"]:
        dl_luc = row["dl_lucene"] if row["dl_lucene"] is not None else "n/a"
        pred_luc = (f"{row['pred_dl_lucene']:.6f}"
                    if row["pred_dl_lucene"] is not None else "n/a")
        lines.append(f"| {row['query']} | {row['lucene']:.6f} | "
                     f"{row['pred_dl_exact']:.6f} | {pred_luc} | "
                     f"{row['dl_ours']} | {dl_luc} |")
    lines.append("")

    if checks.get("duplicate_term_queries"):
        lines += ["### 3.1 Known semantic difference: repeated query terms", "",
                  "Textbook BM25 is written as a sum over t in Q, Σ_{t∈Q}, i.e. the query is "
                  "treated as a **set of terms**, so `pizza pizza` and `pizza` express the same "
                  "information need and our implementation de-duplicates them.",
                  "Lucene's BooleanQuery does **not** de-duplicate: adding the same TermQuery "
                  "twice as SHOULD",
                  "exactly doubles the score. Measured below (top-1 document):", "",
                  "| query | tokens | de-duplicated score | score with duplicates | with dup / de-dup |",
                  "|---|---|---:|---:|---:|"]
        for row in checks["duplicate_term_queries"]:
            lines.append(f"| {row['query']} `{row['text']}` | {row['tokens']} | "
                         f"{row['score_dedup']:.6f} | {row['score_with_duplicates']:.6f} | "
                         f"{row['ratio']:.4f} |")
        lines.append("")
        lines += ["The Lucene results used for the comparison are explicitly de-duplicated on the "
                  "Java side (both sides then share the same query semantics),",
                  "so this behavior does not contaminate the overlap / tau figures above -- but it "
                  "is itself an implementation difference",
                  "that belongs in the report.", ""]

    lines += ["## 4. E9 (deliverable BM25 vs real Lucene BM25) per-query detail", "",
              "| query | text | axis | overlap@10 | overlap@50 | tau (union) | top-1 same | "
              "not retrieved by BM25 | not retrieved by Lucene |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for qid, metrics in groups["E9"]["per_query"].items():
        tau = metrics["tau_b_union"]["tau"] if metrics["tau_b_union"] else None
        tau_text = f"{tau:.4f}" if tau is not None else "n/a"
        lines.append(f"| {qid} | {metrics['text']} | {metrics['axis']} | "
                     f"{metrics['overlap']['@10']} | {metrics['overlap']['@50']} | "
                     f"{tau_text} | {'yes' if metrics['top1_same'] else 'no'} | "
                     f"{metrics['n_missing_a']} | {metrics['n_missing_b']} |")
    lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- main

def load_index_from_disk(index_file, input_dir, tokenizer):
    reviews_path = input_dir / "review.json"
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    source = {"file": reviews_path.name, "bytes": reviews_path.stat().st_size,
              "reviews": manifest["files"]["review.json"]["records"],
              "tokenizer": tokenizer}
    return load_index(index_file, source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/meridian_id")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "part2/results")
    parser.add_argument("--index-file", type=Path, default=ROOT / "part2/.cache/index.pkl")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "part2/.cache/lucene")
    parser.add_argument("--tokenizer", default="custom")
    parser.add_argument("--lucene-lib", type=Path, default=None)
    parser.add_argument("--java-home", type=Path, default=None)
    parser.add_argument("--queries", default="", help="run only the given query ids, comma-separated")
    parser.add_argument("--skip-run", action="store_true", help="do not re-run Java, reuse the existing output")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    verbose = not args.quiet

    queries = QUERIES
    if args.queries:
        wanted = {q.strip() for q in args.queries.split(",") if q.strip()}
        queries = [q for q in queries if q["id"] in wanted]

    java_home = args.java_home or find_java_home()
    lucene_lib = args.lucene_lib or find_lucene_lib()
    if java_home is None:
        raise SystemExit("No JDK found; point to one with --java-home")
    if lucene_lib is None:
        raise SystemExit("No Lucene jars found; point to them with --lucene-lib (needs the "
                         "lucene-core / -analysis-common / -queryparser jars)")
    if verbose:
        print(f"JDK: {java_home}\nLucene lib: {lucene_lib}")

    index = load_index_from_disk(args.index_file, args.input_dir, args.tokenizer)
    for query in queries:
        from .queries import profile_query
        query["df_band"] = profile_query(index, query["text"])["df_band"]

    export_corpus(args.input_dir, args.work_dir, args.tokenizer, verbose)
    export_queries(args.work_dir, queries, args.tokenizer)

    out_json = args.work_dir / "lucene_results.json"
    if not args.skip_run:
        run_java(args.work_dir, out_json, java_home, lucene_lib, verbose)
    lucene_runs, lucene_meta = load_lucene(out_json)

    # our four conditions: a full ranking (n_docs entries) so that tau reflects real rank positions
    n_docs = len(index.doc_lengths)
    our_runs = {name: {} for name in OUR_CONDITIONS}
    started = time.perf_counter()
    for query in queries:
        for name, runner in OUR_CONDITIONS.items():
            our_runs[name][query["id"]] = runner(index, query["text"], n_docs)
    if verbose:
        print(f"Our four conditions finished in {time.perf_counter() - started:.1f}s")

    groups = compute_groups(index, our_runs, lucene_runs, queries)
    aggregates = {group: aggregate(groups[group], queries) for group in groups}
    checks = score_check(index, lucene_runs, queries, verbose)

    index_meta = {"reviews": len(index.doc_lengths), "tokenizer": index.tokenizer,
                  "avgdl": round(index.avg_doc_length, 4)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "lucene_comparison.json").write_text(
        json.dumps({"meta": lucene_meta, "index": index_meta,
                    "groups": {g: {"pair": groups[g]["pair"], "note": groups[g]["note"],
                                   "aggregate": aggregates[g],
                                   "per_query": groups[g]["per_query"]}
                               for g in groups},
                    "score_check": checks}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8")
    write_markdown(args.output_dir / "lucene_comparison.md", lucene_meta, groups,
                   aggregates, checks, index_meta)

    if verbose:
        for group in ["E8", "E9", "E10", "E11"]:
            row = aggregates[group][0]
            print(f"{group} {groups[group]['pair']}: overlap@10={row['overlap@10']} "
                  f"overlap@50={row['overlap@50']} tau_union={row['tau_union']} "
                  f"top1_same={row['top1_same_rate']}")
        print(f"Output written to {args.output_dir}")


if __name__ == "__main__":
    main()
