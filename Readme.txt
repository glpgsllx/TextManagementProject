================================================================================
AI6122 Text Data Management & Processing - Group Assignment
Source code README (Section 4.2 of the assignment brief)
================================================================================

What this repository contains
-----------------------------
This repository holds OUR OWN source code only, covering two parts of the
assignment:

  Part 1  Tokenization, Zipf's law, Heaps' law, tokenizer comparison
          (Section 3.2 of the assignment brief)
  Part 2  Inverted index, TF-IDF with cosine similarity, Okapi BM25, query
          experiments, Lucene cross-check (Sections 3.3.1 and 3.3.2)

The dataset (Yelp reviews) and all third-party libraries are NOT included, as
required by Section 4.2 of the brief. Everything here is plain Python or Java
source plus small reproducible result artefacts (CSV / Markdown / JSON tables
and a few figures). No dataset, no index cache, no binary, no library.

Repository layout:

  <repository root>/
    Readme.txt           this file
    text_processing/     shared tokenizer (custom regex tokenizer, Part 1+2)
    part1/               statistics, Zipf, Heaps, tokenizer comparison
    part2/               inverted index, TF-IDF, BM25, experiments, Lucene probe
    scripts/             builds the city subset from the raw Yelp corpus
    tests/               unit tests for the shared tokenizer
    requirements.txt     Python dependencies (root level, inherited by part1/2)
    README.md            top-level guide (what to run, in what order)

Submitting the Src Gxx.zip to NTULearn
--------------------------------------
Section 4.2 asks for "An Src Gxx.zip file containing Readme.txt and the
SourceCode folder". Build it in Windows PowerShell from the directory that
holds the checkout (xx = your group number). robocopy copies the working
tree and excludes the dataset, the git metadata and every cache:

  New-Item -ItemType Directory Src_Gxx\SourceCode | Out-Null
  robocopy TextManagementProject Src_Gxx\SourceCode /E /XD .git data .cache __pycache__ .venv /NFL /NDL
  Copy-Item TextManagementProject\Readme.txt Src_Gxx\
  Compress-Archive -Path Src_Gxx\* -DestinationPath Src_Gxx.zip

The archive must hold only Readme.txt and SourceCode/ - no dataset, no
third-party library, no .git directory, no index cache. It comes out at about
2.4 MB, mostly the per-query JSON results under part2/results/; if it is tens
of MB, data/ or part2/.cache/ was copied by mistake.

================================================================================
1. THIRD-PARTY LIBRARIES AND DOWNLOAD LINKS
================================================================================

Required (Part 1 and Part 2 both need them):

  Python 3.9 or newer (verified on Python 3.13.7)
      https://www.python.org/downloads/

  NLTK 3.9.4  - only used to obtain WhitespaceTokenizer / WordPunctTokenizer
                as reference tokenizers in the Part 1 comparison. No NLTK data
                package (no punkt, no stopword corpus) is needed.
      https://pypi.org/project/nltk/
      https://www.nltk.org/install.html

  regex 2026.5.9  - drop-in replacement for the standard library `re`, used by
                our custom tokenizer (Unicode property escapes).
      https://pypi.org/project/regex/

Required for the figures (Part 1 Zipf / Heaps plots, and the two Part 2
report figures drawn by delivery/make_figures.py):

  numpy 2.x         https://pypi.org/project/numpy/
  matplotlib 3.x    https://pypi.org/project/matplotlib/

Optional, only for the Lucene cross-check in Part 2 (Section 3.3.2 allows
comparing with a standard library; nothing in the deliverable depends on it):

  JDK 25 (any JDK 21+ works)
      https://jdk.java.net/25/
  Apache Lucene 10.5.2 - three JARs: lucene-core, lucene-queryparser,
      lucene-analysis-common
      https://lucene.apache.org/core/downloads.html
      or from Maven Central:
      https://repo1.maven.org/maven2/org/apache/lucene/lucene-core/10.5.2/
      https://repo1.maven.org/maven2/org/apache/lucene/lucene-queryparser/10.5.2/
      https://repo1.maven.org/maven2/org/apache/lucene/lucene-analysis-common/10.5.2/

None of the ranking functions use a built-in scoring module: both scoring
functions (TF-IDF cosine and BM25) are implemented from scratch in Python.
Lucene appears only in the optional cross-check, where its own BM25 and
ClassicSimilarity rankings serve as an external baseline to compare against.

================================================================================
2. INSTALLATION GUIDE
================================================================================

Step 1 - get the code

  git clone <repository-url>
  cd TextManagementProject

Step 2 - set up Python (from the repository root)

  python -m venv .venv
  .venv\Scripts\activate          (Windows)
  source .venv/bin/activate       (macOS / Linux)
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt        # nltk + regex
  python -m pip install -r part1/requirements.txt  # + numpy + matplotlib

  Note on pinned versions: requirements.txt pins nltk==3.10.4 and
  regex==2026.9.29 (the Part 1 pins). The reported numbers were produced with
  nltk 3.9.4 and regex 2026.5.9. This does not matter for Part 2: Part 2 never
  loads nltk, and regex is used only for basic patterns and Unicode property
  escapes, which both regex versions handle identically.

  No NLTK data download is required (`python -m nltk.downloader` is NOT needed).

Step 3 - put the dataset in place (dataset not shipped in this repository)

  Create the folder  data/meridian_id/  next to part1/ and part2/, and place
  three files in it:

    review.json     one JSON object per line, fields: review_id, text, ...
    business.json   one JSON object per line
    manifest.json   {"files": {"review.json": {"records": 37401}, ...}, ...}

  Our subset is the "Meridian, ID" city subset of the Yelp Open Dataset
  (37,401 reviews). Any city subset works - the code reads whatever is there
  and reports the counts it found; it never hard-codes 37,401 except as a
  cross-check against manifest.json and against the Part 1 statistics.

Step 4 (optional) - Lucene cross-check only

  Put the three Lucene JARs in one directory and point the experiment at it:

    python -m part2.lucene_experiment --lucene-lib C:\tmp\lucene-probe\lib --java-home "C:\Program Files\Java\jdk-25.0.3"

  --lucene-lib points at the directory holding the three Lucene JARs; --java-home
  points at the JDK. Both are optional. If omitted, the script looks for a JDK in
  the JAVA_HOME environment variable and then in C:\Program Files\Java\jdk-*, and
  for the Lucene JARs in C:\tmp\lucene-probe\lib or in this repository's
  part2/lucene/lib (put the JARs in the latter if you want them found
  automatically). It does NOT look at the PATH. If neither is found the script
  exits with a message telling you which flag to add.

  If Java or the JARs are missing, this command fails and nothing else is
  affected - all other results in the report come from the Python code alone.

Step 5 - verify the installation

  python -m unittest discover -s tests
  python -m unittest discover -s part1/tests
  python -m unittest discover -s part2/tests

  Expect 132 passing tests, 0 failures (7 shared tokenizer + 7 Part 1 +
  118 Part 2). The test files are part of the deliverable: they are how we
  defend the code line by line.

================================================================================
3. HOW TO RUN - COMMAND LINES, INPUT FORMAT, PARAMETERS
================================================================================

All commands are run from the repository root (the directory that
contains part1/ and part2/, i.e. the directory holding this Readme.txt).
Every script is a module entry point; there is no
GUI, no server, no configuration file.

--- Input format ---------------------------------------------------------------

  review.json     JSON Lines: one review per line, UTF-8. Only "review_id" and
                  "text" are read. Streamed line by line, so memory use does
                  not depend on file size.
  Queries         plain text strings, e.g. "gluten free pancakes". They are
                  tokenized with the SAME tokenizer used at index time.

--- Part 1 ---------------------------------------------------------------------

  python -m part1.run
      Writes part1/results/: tokenizer_comparison.csv, law_parameters.csv,
      heaps_samples.csv, tokenization_examples.csv and the Zipf/Heaps figures.
      Options: --input-dir, --output-dir, --seed (default 42).

  python -m part1 --text "I don't like the-best-food"
      Tokenizes one string with all three tokenizers and prints the tokens.

--- Part 2, step 1: build the index -------------------------------------------

  python -m part2.run
      Builds the inverted index from data/meridian_id/review.json and writes
      part2/results/index_statistics.json. The index is pickled to
      part2/.cache/index.pkl (~20 MB) so later runs reload it in ~2 s instead
      of re-tokenizing in ~11 s.
      Options:
        --rebuild         ignore the cached index and rebuild it
        --index-file PATH where to store the pickle
        --limit N         only first N reviews (quick smoke test; the result is
                          NOT cached and NOT used for any reported number)
        --tokenizer NAME  custom (default) / whitespace / wordpunct
        --input-dir PATH  --output-dir PATH

      The index carries df (per term), dl (per document), avgdl (computed once
      globally at indexing time, as Section 3.3.2 requires), and two sets of
      document norms for the two TF-IDF weighting schemes.

--- Part 2, step 2: run queries interactively ----------------------------------

  python -m part2.search_cli "gluten free pancakes" --all --top-k 3
      Options:
        query            (positional) the query text
        --model          tfidf (default) or bm25
        --variant        tfidf: lnc.ltc / ltc.ltc ; bm25: okapi / course / lucene
        --k1 FLOAT       BM25 k1, default 1.2
        --b  FLOAT       BM25 b,  default 0.75
        --top-k INT      how many hits to print, default 10
        --all            run both implementations side by side
        --no-snippet     do not load the corpus, print ids and scores only
        --input-dir / --index-file / --tokenizer  as above

--- Part 2, step 3: the query experiment (the Analysis section) ----------------

  python -m part2.query_experiment
      50 queries x 6 scoring conditions + a 4x4 k1/b grid. ~20-60 s.
      Writes part2/results/query_experiment_summary.md / .json / _runs.json /
      _overlap.csv.
      Options: --queries Q01,Q19  --axes category,aspect  --skip-grid
               --pool N (deep pool depth, default 50)  --ks 5,10,20,50  --quiet

  python -m part2.impl_analysis
      Per-implementation behavioural profile (recall count, first-place margin,
      tie count, top-10 length bias, b-sweep). ~3 s. -> impl_analysis.md/.json

  python -m part2.qualitative_reading
      One-sentence qualitative judgement per query + the clamped-vs-unclamped
      contrast. ~1 s. -> qualitative_reading.md

  python -m part2.lucene_experiment
      Optional cross-check against real Apache Lucene 10.5.2 (needs Java).
      ~20 s. -> lucene_comparison.md/.json
      Options: --queries, --skip-run (reuse the last Java output),
               --lucene-lib, --java-home

  python delivery/make_figures.py
      Draws the two report figures from the existing part2/results/ files
      (re-runs no experiment; needs matplotlib). ~5 s. ->
      delivery/figures/fig1_b_and_length.{pdf,png},
      delivery/figures/fig2_df_drives_disagreement.{pdf,png}

--- Recommended order ---------------------------------------------------------

  python -m part1.run                  # part2 cross-checks against these stats
  python -m part2.run                  # build index
  python -m part2.query_experiment     # the Analysis numbers
  python -m part2.impl_analysis
  python -m part2.qualitative_reading
  python -m part2.lucene_experiment    # optional, needs Java
  python delivery/make_figures.py      # report figures, needs matplotlib

  Running part1 first matters: part2.run reads
  part1/results/tokenizer_comparison.csv and reports whether the document
  count, token count and vocabulary size agree. If the file is absent, that
  cross-check is silently skipped.

================================================================================
4. SAMPLE OUTPUT AND HOW TO READ IT
================================================================================

--- 4.1  python -m part2.run  (abridged) ---------------------------------------

  {
    "tokenizer": "custom",
    "index_source": "built from corpus",
    "index_seconds": 10.9,
    "reviews": 37401,
    "terms": 38120,
    "total_tokens": 3783589,
    "avgdl": 101.1628,
    "postings": 2491298,
    "singleton_terms": 15862,
    "dl_min": 4, "dl_median": 69, "dl_max": 1005,
    "index_file_mb": 20.0,
    "top_df_terms": [{"term":"and","df":33401},{"term":"the","df":33073}, ...],
    "cross_check_part1": {"part1": {"reviews":37401,"tokens":3783589,"types":38120},
                          "index": {"reviews":37401,"tokens":3783589,"types":38120},
                          "matches": true}
  }

  How to read it:
    reviews / terms / total_tokens  size of the indexed collection
    avgdl                           average document length in tokens; computed
                                    once at indexing time and reused by BM25
    postings                        total entries in the inverted lists
                                    (= sum of df over all terms)
    singleton_terms                 terms appearing in exactly one review
                                    (15,862 of 38,120 - a long tail, which is
                                    why idf matters on this corpus)
    dl_min / dl_median / dl_max     document length spread; BM25's length
                                    normalisation works on this spread
    cross_check_part1.matches       true means Part 2 counted exactly the same
                                    documents, tokens and types as Part 1 did
                                    with the same tokenizer

--- 4.2  python -m part2.search_cli "gluten free pancakes" --all --top-k 3 -----

  --- TF-IDF lnc.ltc: 3 hits ---
   1. O_fr1k1BiqRhTNDzUSXNqw  score=0.4155  dl=65  | Amazing staff, out of this world food!! ...
   2. djIltsOkyK42we4R3YchDA  score=0.3838  dl=19  | Mediocre gluten free pancakes but ...
   3. zir_LSXZW67dYR8cZaMEVw  score=0.3825  dl=8   | Great Pizza! They have gluten free options too!
  --- TF-IDF ltc.ltc: 3 hits ---
   1. O_fr1k1BiqRhTNDzUSXNqw  score=0.5864  dl=65  | ...
   2. zir_LSXZW67dYR8cZaMEVw  score=0.5772  dl=8   | ...
   3. DptIUgXHB6REHOUNWkV_6w  score=0.5087  dl=42  | Even though it was lunch time, ...
  --- BM25 okapi (k1=1.2, b=0.75): 3 hits ---
   1. O_fr1k1BiqRhTNDzUSXNqw  score=21.0810  dl=65  | Amazing staff, out of this world food!! ...
   2. M36lgQfiQl-7aqd2ISg9rQ  score=19.3336  dl=69  | Gluten-Free In Boise!!! Best kept secret ...
   3. DptIUgXHB6REHOUNWkV_6w  score=18.9496  dl=42  | Even though it was lunch time, ...

  How to read it:
    The 22-character string is the Yelp review_id. `dl` is that review's length
    in tokens. The text after `|` is the beginning of the review.
    TF-IDF scores are cosine values in [0, 1] - comparable across queries.
    BM25 scores are unbounded sums - comparable only WITHIN one query.
    The absolute BM25 scale also differs from Lucene's by a constant factor
    (k1+1 = 2.2), because our implementation keeps the textbook constant that
    current Lucene drops. This changes no ranking, only the scale, so never
    compare our BM25 number with Lucene's number directly - compare the order.

--- 4.3  python -m part2.query_experiment  (tail of stdout) --------------------

  6 conditions x 50 queries
  E1 ['tfidf_lnc_ltc', 'bm25_okapi']: overlap@10=5.50 overlap@50=34.22
                                      tau_union=0.397 top1_same=0.50
  E2 ['tfidf_lnc_ltc', 'tfidf_ltc_ltc']: overlap@10=5.14 ... tau_union=0.248
  E3 ['bm25_okapi', 'bm25_course']:    overlap@10=9.24  ... tau_union=0.953
  E6 ['bm25_okapi', 'bm25_okapi_unclamped']: ... tau_union=0.961

  How to read it:
    overlap@k    out of the top k hits, how many are the same document in both
                 rankings (a SET measure, e.g. 5.50 means 5.5 of 10 on average
                 over the 50 queries)
    tau_union    Kendall tau-b over the union of both top-50 pools: +1 identical
                 order, 0 unrelated, -1 reversed. It mixes in membership
                 differences, so it is the conservative reading.
    top1_same    fraction of queries whose first hit is the same document
    These are AGREEMENT measures, not quality measures - the assignment gives
    no relevance judgements, so precision/recall/MAP/NDCG cannot be computed.

================================================================================
5. REPRODUCING THE NUMBERS IN THE REPORT
================================================================================

  Report figure/table                              Command
  -----------------------------------------------  ----------------------------
  Index statistics (Section 3.3.1)                 python -m part2.run
  Overlap@k and Kendall tau table (Analysis)       python -m part2.query_experiment
  k1 x b sensitivity grid                          python -m part2.query_experiment
  Per-implementation behaviour profile             python -m part2.impl_analysis
  Qualitative reading of the top hits              python -m part2.qualitative_reading
  Lucene cross-check (optional)                    python -m part2.lucene_experiment
  Figures 1 and 2                                  python delivery/make_figures.py

  All generated tables land in part2/results/ as both .md (readable) and .json
  (machine-readable). The report quotes from these files only.

================================================================================
6. KNOWN CAVEATS
================================================================================

* The dataset is not in this repository (Section 4.2 forbids shipping it). Without it, only
  the unit tests and the tokenizer demo run; every Part 2 command needs
  data/meridian_id/review.json.
* part2/.cache/ (the pickled index, ~20 MB) and part2/.cache/lucene/ (the
  Lucene index and the exported corpus) are build artefacts, not source. They
  are git-ignored and are regenerated by the commands above.
* Timings in this README were measured on a Windows 11 laptop with Python
  3.13.7; they are indicative only.
* BM25 absolute scores are 2.2x Lucene's by design (see 4.2). Only the ranking
  is comparable.
* With the Okapi idf variant, a query made only of terms whose df exceeds
  N/2 returns zero hits, because their idf is clamped to 0 and every term
  is then skipped. This is intended behaviour and is analysed in the report
  (query Q45 "the and of"). Use --variant lucene or course to avoid it.
================================================================================
