# Part 4: Finding comparisons between businesses

This is the Application required by Section 3.5 of the brief: a small
command-line NLP tool that reads the review corpus and reports the sentences in
which a reviewer compares the business under review with another business or
place.

## How to run

From the repository root (the same environment as Parts 1 and 3):

```bash
python -m part4.run
```

Results are written to `part4/results/` in about 12 seconds. Options:

```bash
python -m part4.run --input-dir data/meridian_id --output-dir part4/results --limit 0
```

`--limit N` scores only the first N reviews, which is useful for a quick look.

To reproduce the evaluation, redraw the samples and score them against the hand
labels that ship with the code:

```bash
python -m part4.sample        # writes part4/eval/candidates.csv and random_audit.csv
python -m part4.evaluate      # precision / recall / F1 against part4/eval/labels.csv
python -m unittest discover -s part4/tests -t .
```

`sample.py` writes `keys.csv` with the predictions, so `candidates.csv` can be
labelled without seeing the answer first; `labels.csv` is the annotation we
produced for the 120 sentences quoted in the report.

## Output

| File | Contents |
| --- | --- |
| `results/comparisons.csv` | every sentence classified as a business comparison |
| `results/superlatives.csv` | scoped superlatives ("best pizza in town") |
| `results/summary.json` | totals, comparisons per 1,000 sentences by star and year, cue families, top targets |
| `eval/candidates.csv`, `eval/keys.csv` | the 120-sentence sample, sentence only vs. prediction |
| `eval/labels.csv` | hand labels for that sample |
| `eval/random_audit.csv` | 40 sentences drawn without a cue filter, for the recall check |

`comparisons.csv` has one row per sentence with the columns
`review_id, business, year, stars, kind, families, targets, comparator, sentence`.
`families` lists the cue families that fired and `targets` the other businesses
named in the sentence.

## Definition

A sentence counts as a **business comparison** when it

1. contains a comparison cue (a comparative with *than*, *more/less ... than*,
   *compared to*, *same as*, *beats*, *prefer ... over*, *unlike*);
2. does not match a known non-comparison pattern (*other than that*, *more than
   happy*, *less than an hour*, *better than expected*); and
3. either names another business from the corpus, or leaves a target phrase that
   points at one (*the one downtown*, *other places*, *the restaurant down the
   street*).

A scoped superlative (*the best pizza in town*, *the best I have ever had*) is
returned as a second class, because it compares with every other business
implicitly but names none.

## Method

1. `sentences.py` splits each review at newlines and end punctuation, keeping
   abbreviations (`Dr.`, `a.m.`) and decimals (`4.50`) intact.
2. `patterns.py` holds the cue families, the non-comparison idioms and the
   target phrases.
3. `detect.py` normalises the sentence, matches the cues, extracts the
   comparator phrase, and matches the 1,045 corpus business names and about 90
   chain names longest-first. The reviewed business is removed from the targets,
   so "better than this Chipotle" does not count as a comparison for a Chipotle
   review.

Precision and recall are measured on 120 cue-bearing sentences labelled by hand
(precision 0.91, recall 0.79, F1 0.85 on the sample that ships with the code).
The tool is rule-based on purpose: every decision can be read off a pattern, and
no labelled training set was available.

## Limitations

- One annotator, 120 sentences: the confidence intervals in the report are wide.
- The lexicon only knows the Meridian businesses and the chain names in
  `detect.py`; comparisons with businesses outside the subset are missed unless
  the sentence also carries a place phrase.
- Anaphora is not resolved: *the others*, *those back home*, *as good as it
  gets* are not turned into targets.
- Sentence splitting is regular-expression based; a review without punctuation
  becomes one long sentence.
