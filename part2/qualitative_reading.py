"""W5 comparative analysis: qualitative reading of the 50 queries + clamped/unclamped comparison.

Of the five W5 items on the task checklist, this script covers item 4 (a qualitative
reading of the top results, one line per query) and item 2 (the clamped vs. unclamped
comparison); the remaining three are handled by other scripts:

    1  overlap@k + Kendall tau (per query)  -> query_experiment.py (E1-E7)
    3  Parameter sensitivity k1 x b          -> query_experiment.py (E7, full 4x4 grid)
    5  Three-way Lucene comparison           -> lucene_experiment.py (E8-E11)

Why the qualitative reading has to be written by a human, and why it cannot be taken as truth
--------------------------------------------------------------------------------------------
With no relevance judgments, there is no correct answer for "relevance". What follows is a
**rough human judgment**: a single rater looking only at the top-3 raw snippets. Its purpose
is to give the aggregate numbers a concrete counterpart, so the reader can see what actually
happens behind a figure such as overlap@10 = 5.5. It is **not ground truth** and must not be
used to compute P@10 or to claim that "BM25 is better". Wherever a quality claim is needed,
the report has to fall back on the honest position: with no judgments, only agreement (consistency) can be discussed.

Four patterns that recur across the readings (also the summary of all 50, quotable in the report):
    A. The TF-IDF top-1 result is almost always extremely short (4-20 words), a review that
       echoes the query terms word for word.
    B. The BM25 top-1 result is longer and more informative, but on high-df queries it drifts
       more easily into unrelated long reviews.
    C. The corpus itself contains duplicate and near-duplicate reviews, which hurt top-k
       diversity for both scoring implementations.
    D. On multi-term queries, TF-IDF can miss the core term (the Q19 top-1 result is about
       burgers, not pizza).

How to run (from the repository root):

    python -m part2.qualitative_reading

Output: part2/results/qualitative_reading.md
"""

import json
import time
from pathlib import Path

from .bm25 import DEFAULT_B, DEFAULT_K1, OKAPI, bm25_search, okapi_idf_unclamped
from .inverted_index import load_index
from .queries import QUERIES
from .tfidf import LNC_LTC, tfidf_search

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "part2" / "results"
CORPUS = ROOT / "data" / "meridian_id" / "review.json"

SNIPPET_CHARS = 90
TOP_N = 3

# Demo queries for the clamped/unclamped comparison. The first is the example named
# explicitly in the task checklist; the other two are the boundary cases in this query
# set that actually trigger clamping.
CLAMP_DEMOS = ["fish and chips", "the and of", "pizza and pasta"]

# ---------------------------------------------------------------- readings
# One line each. The wording is fixed as "what each of the two implementations picked up
# + what that difference means"; it never says which one is better (with no relevance
# judgments, that cannot be decided).
READINGS = {
    "Q01": "Both talk about pizza, but in completely different registers: the TF-IDF top-1 result is a 17-word short review, while BM25 returns a 49-word account of an actual ordering experience.",
    "Q02": "The top-1 result is the same document for both, but the BM25 2nd result is negative - neither implementation distinguishes sentiment polarity; both only match terms.",
    "Q03": "All three TF-IDF results are tightly on topic around \"Best Thai food in...\"; the BM25 top-1 result is an anecdote about how the writer's parents judge Thai food, which is clearly worse.",
    "Q04": "The two top-3 result sets are identical, the most consistent of the 50 queries (the query terms are all low-frequency, so candidates are few and scores are sparse).",
    "Q05": "All three TF-IDF results are ten-odd-word notes of the \"I like this coffee shop\" kind; the BM25 top-1 result is 74 words describing the layout of the shop - more information, but more diffuse.",
    "Q06": "The top-1 results differ but both hit the breakfast burrito; the BM25 2nd and 3rd results are tangents mentioned in passing inside long reviews.",
    "Q07": "The top-1 result is the same document for both; the BM25 results are longer overall (52/18/44 words vs. 18/19/20 words).",
    "Q08": "The TF-IDF 1st and 2nd results are **duplicate reviews of the same text** - exactly the \"two documents tied at the top\" case that was detected; BM25 instead puts a substantial 46-word BBQ review first.",
    "Q09": "The two implementations agree on the top-1 result, a positive review of a barber shop; only the order of the 2nd and 3rd results differs.",
    "Q10": "Both retrieve accounts of a dental visit, with top-1 lengths close to each other (35 / 33 words); the two diverge only in ranking.",
    "Q11": "The top two results are identical across both implementations, both positive accounts of an oil-change visit.",
    "Q12": "The top-1 result is shared (63 words asking about brake repair); the TF-IDF 2nd and 3rd results are \"iPhone repair\" and \"Chip Repair\" - they match only repair, i.e. a partial match.",
    "Q13": "The top-1 result is shared; the BM25 3rd result (57 words) drifts to an insurance company, unrelated to food and dining.",
    "Q14": "All three TF-IDF results literally contain rude/staff; the BM25 2nd result (40 words) contains neither term - a textbook case of drift.",
    "Q15": "The TF-IDF top-1 result is only 9 words (\"...and produce, friendly staff.\"); BM25 returns 20 words first and 279 words second.",
    "Q16": "The TF-IDF top-1 result is just 4 words, \"Great food, slow service....\" - a literal perfect match that carries no information; the BM25 top-1 result is 30 words.",
    "Q17": "The top-1 result is shared; the BM25 3rd result (98 words) complains throughout but never mentions waiting.",
    "Q18": "The top-1 result is shared; the BM25 3rd result (45 words) is about how many TVs the place has, unrelated to overpriced.",
    "Q19": "**The TF-IDF top-1 result is \"best burgers in town\" - the query matched best/in/town and missed the core term pizza**; retrieved count 22,546 vs. 7,751 for BM25 (\"in\" gets clamped away).",
    "Q20": "The top two results are identical across both; the 3rd differs, though both are negative experiences.",
    "Q21": "The TF-IDF 3rd result is \"Great food and cold beer\" (a cinema), matching cold and food separately; the BM25 2nd result (46 words) never mentions cold.",
    "Q22": "The two top-3 result sets overlap heavily; both are complaints about the wrong dish or the wrong order being delivered.",
    "Q23": "The top two are shared; the BM25 3rd result (41 words) is about a dirty, messy room - a hotel, not a restaurant.",
    "Q24": "All three TF-IDF results are ten-odd-word \"Great ... atmosphere\" lines; the BM25 top-1 result, 56 words, talks about pizza, only weakly related to atmosphere.",
    "Q25": "The top-1 result is shared (65 words); the BM25 3rd result (42 words) genuinely discusses gluten-free pancakes, more on topic than the TF-IDF 3rd result (8 words that only mention gluten free options).",
    "Q26": "The TF-IDF 1st and 2nd results are duplicate reviews of the same text (the tied scores that were detected); the BM25 top-1 result, 57 words, reviews a veggie burger.",
    "Q27": "The top-1 result is shared, and **it is negative** (\"they don't allow dogs\") - neither implementation understands that \"dog friendly\" is a positive request.",
    "Q28": "The two top-3 result sets are completely identical (the query terms are low-frequency, so candidates are few).",
    "Q29": "The top-1 result is shared and negative (a slow, rude drive thru); the BM25 2nd result (54 words) wanders off to Starbucks.",
    "Q30": "The BM25 top-1 result (87 words) is a negative review saying the place is unsuitable for working on WiFi; the TF-IDF top-1 result, 17 words, mentions free wifi positively.",
    "Q31": "The top-1 result is shared; the BM25 3rd result is a near-duplicate of the top-1 result (34 / 48 words, two versions of the same text).",
    "Q32": "The top two results are identical across both implementations, all containing kid friendly literally.",
    "Q33": "The TF-IDF top-1 result is the most on topic (\"Seating is accessible for a scooter/wheelchair\"); the BM25 2nd and 3rd results are duplicates of the same text.",
    "Q34": "The two top-3 result sets are completely identical.",
    "Q35": "The TF-IDF top-1 result is the 4-word \"Great food, slow service....\"; the BM25 top-1 result, 65 words, says the food is a poor excuse for Mexican cuisine - a negative review.",
    "Q36": "The TF-IDF top-1 result is 4 words and the 2nd is 5 words - all extremely short reviews; the BM25 top-1 result, 64 words, describes the worst customer service.",
    "Q37": "The two top-3 result sets are completely identical (18/20/21 words; all three come from the same batch of short reviews).",
    "Q38": "The TF-IDF top-1 result is the 14-word \"Worth the wait!\"; the BM25 top-1 result, 55 words, is about queueing for a table.",
    "Q39": "The two implementations swap top-1 and 2nd (TF1=BM2, TF2=BM1): the same documents, only in a different order.",
    "Q40": "All three TF-IDF results are ten-odd-word short reviews; the BM25 top-1 result, 135 words, compares burger places against each other.",
    "Q41": "The BM25 top-1 result (47 words) mentions a \"crabby teenager\" - genuinely related to rudeness, though the word rude never appears; the TF-IDF top-1 result contains rude literally.",
    "Q42": "The top-1 result is shared (26 words, positive); the BM25 2nd result, 146 words, recounts a dispute over reserving the patio.",
    "Q43": "A rare term (df=76), so both implementations have only 76 candidates; the TF-IDF top-1 result, 23 words, mentions the espresso bar directly, while the BM25 top-1 result, 117 words, only talks about drinks.",
    "Q44": "Only one document in the whole collection matches, and both implementations return that same one (484 words) - when df=1, any formula can only produce that result.",
    "Q45": "All three terms have their idf clamped to 0 under BM25, so it **returns 0 results**; TF-IDF returns 36,558. This query expresses no real information need, so whatever is returned is meaningless anyway.",
    "Q46": "BM25 clamps away \"the\" (retrieved count 13,139 = the df of food alone), while TF-IDF keeps 34,169; the BM25 top-1 result is the same as the Q35 top-1 result.",
    "Q47": "TF-IDF retrieves 33,613 (including and) versus 2,165 for BM25 (and clamped away), yet the top two results still agree - evidence that \"and\" contributes almost nothing to the ranking.",
    "Q48": "A repeated-term query: both implementations follow the textbook convention and treat the query as a **set**, de-duplicating it, so the results are identical to Q01 (Lucene does not de-duplicate, and its score would double).",
    "Q49": "The two top-3 result sets contain the same documents in a different order, both hitting the literal phrase \"don't waste your money\" precisely.",
    "Q50": "df=0, so both implementations return 0 results, as expected (a term absent from the collection does not take part in scoring).",
}


# ---------------------------------------------------------------- batch run

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


def top_n(index, run, text, texts, n=TOP_N):
    ranked = run(text, len(index.doc_lengths))
    out = []
    for doc_id, score in ranked[:n]:
        out.append({"doc": doc_id, "score": round(score, 6),
                    "dl": index.doc_lengths[doc_id],
                    "snippet": (texts.get(doc_id) or "").replace("\n", " ")
                              .replace("|", r"\|")[:SNIPPET_CHARS]})
    return {"n_returned": len(ranked), "top": out}


def collect(index, texts, verbose=True):
    """Top-3 results of the two delivered implementations for all 50 queries."""
    runs = {"tfidf": {}, "bm25": {}}
    for query in QUERIES:
        runs["tfidf"][query["id"]] = top_n(
            index, lambda t, k: tfidf_search(index, t, top_k=k, variant=LNC_LTC),
            query["text"], texts)
        runs["bm25"][query["id"]] = top_n(
            index, lambda t, k: bm25_search(index, t, top_k=k, variant=OKAPI,
                                            k1=DEFAULT_K1, b=DEFAULT_B),
            query["text"], texts)
    return runs


def clamp_demo(index, texts):
    """Clamped vs. unclamped comparison: the same query under both idf variants."""
    out = []
    for text in CLAMP_DEMOS:
        clamped = top_n(index, lambda t, k: bm25_search(
            index, t, top_k=k, variant=OKAPI, k1=DEFAULT_K1, b=DEFAULT_B), text, texts)
        unclamped = top_n(index, lambda t, k: bm25_search(
            index, t, top_k=k, variant=OKAPI, k1=DEFAULT_K1, b=DEFAULT_B,
            idf_fn=okapi_idf_unclamped), text, texts)
        out.append({"text": text, "clamped": clamped, "unclamped": unclamped})
    return out


def agreement(runs):
    """How many queries have identical top-1 / top-3 sets across the two implementations -
    a quantitative baseline for the readings."""
    same1 = same3 = 0
    for qid in runs["tfidf"]:
        a = [item["doc"] for item in runs["tfidf"][qid]["top"]]
        b = [item["doc"] for item in runs["bm25"][qid]["top"]]
        if a and b and a[0] == b[0]:
            same1 += 1
        if set(a) == set(b):
            same3 += 1
    return {"n": len(runs["tfidf"]), "same_top1": same1, "same_top3_set": same3}


# ---------------------------------------------------------------- output

def write_markdown(path, runs, demos, agree):
    lines = ["# Qualitative Reading: One Line Per Query (50 Queries)", "",
             "Rough human judgment (a single rater, looking only at the raw top-3 snippets of "
             "each implementation). **This is not ground truth**; it only supplies concrete "
             "examples to go with the aggregate numbers. With no relevance judgments, quality "
             "cannot be decided.", "",
             f"The two implementations share the **top-1 result on "
             f"{agree['same_top1']}/{agree['n']} queries**, and have an "
             f"**identical top-3 set on {agree['same_top3_set']}/{agree['n']} queries**.", "",
             "Four patterns that recur across the readings:", "",
             "- **A**: the TF-IDF top-1 result is almost always extremely short (4-20 words), "
             "a review that echoes the query terms word for word;",
             "- **B**: the BM25 top-1 result is longer and more informative, but on high-df "
             "queries it drifts more easily into unrelated long reviews;",
             "- **C**: the corpus itself contains duplicate and near-duplicate reviews, which "
             "hurt top-k diversity for both implementations;",
             "- **D**: on multi-term queries, TF-IDF can miss the core term (see Q19).", "",
             "## Reading, Query by Query", "",
             "| Query | Axis | TF-IDF top-1 (length) | BM25 top-1 (length) | Reading |",
             "|---|---|---:|---:|---|"]
    for query in QUERIES:
        qid = query["id"]
        tf = runs["tfidf"][qid]["top"]
        bm = runs["bm25"][qid]["top"]
        tf1 = f"{tf[0]['snippet']} ({tf[0]['dl']})" if tf else "(0 results)"
        bm1 = f"{bm[0]['snippet']} ({bm[0]['dl']})" if bm else "(0 results)"
        lines.append(f"| `{qid}` {query['text']} | {query['axis']} | {tf1} | "
                     f"{bm1} | {READINGS.get(qid, '')} |")
    lines += ["", "## Clamped vs. Unclamped", "",
              "The same query and the same k1/b, switching only whether the `max(0, .)` "
              "clamp is applied. The first one is the example named in the task checklist.",
              ""]
    for demo in demos:
        lines += [f"### `{demo['text']}`", "",
                  "| | Retrieved count | Top-1 (length) | 2nd (length) | 3rd (length) |",
                  "|---|---:|---|---|---|"]
        for label, key in [("Clamped (delivered configuration)", "clamped"),
                           ("Unclamped", "unclamped")]:
            block = demo[key]
            cells = []
            for item in block["top"]:
                cells.append(f"{item['snippet']} ({item['dl']})")
            while len(cells) < 3:
                cells.append("—")
            lines.append(f"| {label} | {block['n_returned']:,} | " + " | ".join(cells) + " |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    verbose = True
    RESULTS.mkdir(parents=True, exist_ok=True)
    texts = load_texts()
    source = {"file": "review.json", "bytes": CORPUS.stat().st_size,
              "reviews": len(texts), "tokenizer": "custom"}
    index = load_index(ROOT / "part2" / ".cache" / "index.pkl", source)
    started = time.perf_counter()
    runs = collect(index, texts)
    demos = clamp_demo(index, texts)
    agree = agreement(runs)
    md = RESULTS / "qualitative_reading.md"
    write_markdown(md, runs, demos, agree)
    print(f"Elapsed {time.perf_counter() - started:.1f}s -> {md}")
    print(f"Identical top-1 on {agree['same_top1']}/{agree['n']}, "
          f"identical top-3 set on {agree['same_top3_set']}/{agree['n']}")
    for demo in demos:
        print(f"  `{demo['text']}`: clamped {demo['clamped']['n_returned']:,} / "
              f"unclamped {demo['unclamped']['n_returned']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
