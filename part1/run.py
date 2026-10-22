"""Run Part 1 end to end: python -m part1.run (from repository root)."""

from array import array
from collections import Counter
import argparse
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import regex

from text_processing import get_tokenizer

from .plots import make_figures
from .statistics import heaps_points, heaps_statistics, review_order, zipf_statistics
from .tokenizer import (
    APOSTROPHES, RULES, WORD_PATTERN, expand_word, is_emoji, is_emoticon,
)
from .boundary_cases import CASES


ROOT = Path(__file__).resolve().parents[1]
NAMES = ("custom", "whitespace", "wordpunct")


def dump_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_corpus(directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    businesses = set()
    documents = []
    reviews = set()
    dates = []
    for filename in ("business.json", "review.json"):
        digest = hashlib.sha256()
        count = 0
        with (directory / filename).open("rb") as f:
            for raw in f:
                digest.update(raw)
                row = json.loads(raw)
                count += 1
                if filename == "business.json":
                    if row["city"].strip().casefold() != "meridian" or row["state"].strip().upper() != "ID":
                        raise ValueError("Business outside selected Meridian, ID scope")
                    if row["business_id"] in businesses:
                        raise ValueError("Duplicate business_id")
                    businesses.add(row["business_id"])
                else:
                    if row["business_id"] not in businesses or row["review_id"] in reviews:
                        raise ValueError("Invalid review join or duplicate review_id")
                    if not isinstance(row["text"], str):
                        raise ValueError("Non-string review text")
                    reviews.add(row["review_id"])
                    dates.append(row["date"])
                    documents.append({k: row[k] for k in ("review_id", "business_id", "text", "date")})
        expected = manifest["files"][filename]
        if digest.hexdigest() != expected["sha256"] or count != expected["records"]:
            raise ValueError(f"Corpus checksum/count differs from export manifest: {filename}")
    return documents, {
        "city": "Meridian", "state": "ID", "businesses": len(businesses), "reviews": len(documents),
        "date_min": min(dates), "date_max": max(dates),
        "text_characters": sum(len(d["text"]) for d in documents),
        "empty_reviews": sum(not d["text"].strip() for d in documents),
        "source_sha256": {name: manifest["files"][name]["sha256"] for name in ("business.json", "review.json")},
        "validation": "Both file hashes/counts, unique IDs and all review-business joins verified.",
        "scope": "All selected-city reviews, all categories and dates; no language/ratings/open-status filter.",
    }


def build_examples(documents, tokenizers):
    patterns = {
        "negative_contraction": regex.compile(r"\b(?:don['’]t|won['’]t|can['’]t)\b", regex.I),
        "curly_apostrophe": regex.compile(r"\p{L}+’\p{L}+"),
        "hyphen_chain": regex.compile(r"\p{L}+(?:-\p{L}+){2,}"),
        "ascii_emoticon": regex.compile(r"[:;=]-?[)(DPp]"),
        "text_style_emoji_symbol": regex.compile(r"[®™‼⁉]"),
        "decimal_price": regex.compile(r"\$\d+\.\d+"),
        "ambiguous_contraction": regex.compile(r"\b(?:it['’]s|she['’]s|i['’]d)\b", regex.I),
    }
    examples = []
    for name, pattern in patterns.items():
        for document in documents:
            found = pattern.search(document["text"])
            if not found:
                continue
            text = document["text"]
            start = max(0, text.rfind(" ", 0, max(0, found.start() - 45)) + 1)
            end = text.find(" ", min(len(text), found.end() + 45))
            snippet = text[start:] if end == -1 else text[start:end]
            examples.append({"source": "real_corpus_excerpt", "feature": name,
                             "review_id": document["review_id"], "text": snippet})
            break
    for name, text, expected in CASES:
        examples.append({"source": "synthetic_boundary_test", "feature": name,
                         "text": text, "expected_custom_terms": expected})
    for item in examples:
        text = item["text"]
        methods = ("custom", "whitespace", "wordpunct")
        item["surface_tokens"] = {name: tokenizers[name].surface_tokens(text) for name in methods}
        item["normalized_terms"] = {name: tokenizers[name].tokenize(text) for name in methods}
        if "expected_custom_terms" in item:
            item["custom_passed"] = item["normalized_terms"]["custom"] == item["expected_custom_terms"]
    return examples


def analyze(documents):
    tokenizers = {name: get_tokenizer(name) for name in NAMES}
    raw_counts = {name: Counter() for name in NAMES}
    frequencies = {name: Counter() for name in NAMES}
    token_counts, document_types = [], []
    vocabulary_ids = {}
    expansion_counts, emoji_counts, emoticon_counts = Counter(), Counter(), Counter()
    emoji_reviews = emoticon_reviews = 0
    for i, document in enumerate(documents):
        text = document["text"]
        all_raw = {name: tokenizer.surface_tokens(text) for name, tokenizer in tokenizers.items()}
        custom_raw = all_raw["custom"]
        for name in NAMES:
            raw = all_raw[name]
            raw_counts[name].update(raw)
            terms = tokenizers[name].tokenize(text)
            frequencies[name].update(terms)
            if name == "custom":
                token_counts.append(len(terms))
                unique = set(terms)
                ids = []
                for term in sorted(unique):
                    if term not in vocabulary_ids:
                        vocabulary_ids[term] = len(vocabulary_ids)
                    ids.append(vocabulary_ids[term])
                document_types.append(array("I", ids))
        emojis = [t for t in custom_raw if is_emoji(t)]
        emoticons = [t for t in custom_raw if is_emoticon(t)]
        emoji_reviews += bool(emojis)
        emoticon_reviews += bool(emoticons)
        emoji_counts.update(emojis)
        emoticon_counts.update(emoticons)
        for token in custom_raw:
            canonical = token.translate(APOSTROPHES).lower()
            if WORD_PATTERN.fullmatch(canonical):
                expanded = expand_word(canonical)
                if len(expanded) > 1:
                    expansion_counts[canonical] += 1
        if (i + 1) % 10_000 == 0:
            print(f"Tokenized {i + 1:,}/{len(documents):,} reviews", flush=True)
    comparison = []
    for name in NAMES:
        frequency = frequencies[name]
        comparison.append({"method": name, "reviews": len(documents),
                           "surface_tokens": raw_counts[name].total(), "surface_types": len(raw_counts[name]),
                           "normalized_tokens": frequency.total(), "normalized_types": len(frequency),
                           "singleton_types": sum(v == 1 for v in frequency.values())})
    coverage = {"custom_emoji_reviews": emoji_reviews,
                "custom_emoji_sequence_counts": dict(emoji_counts.most_common()),
                "custom_emoticon_reviews": emoticon_reviews,
                "custom_emoticon_counts": dict(emoticon_counts.most_common()),
                "expanded_word_occurrences": expansion_counts.total(),
                "contraction_expansion_counts": dict(expansion_counts.most_common()),
                "emoji_note": "Actual Unicode matches include text-style symbols. Synthetic pictorial emoji cases remain separate."}
    return frequencies, comparison, token_counts, document_types, coverage, build_examples(documents, tokenizers)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/meridian_id")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "part1/results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    # Detailed working files stay local; results/ contains report-ready outputs.
    details = ROOT / "part1/.cache/run"
    details.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    documents, corpus = load_corpus(args.input_dir)
    print(f"Verified {corpus['businesses']:,} businesses and {corpus['reviews']:,} reviews", flush=True)
    frequencies, comparison, token_counts, document_types, coverage, examples = analyze(documents)
    for method, frequency in frequencies.items():
        ranked = sorted(frequency.items(), key=lambda item: (-item[1], item[0]))
        write_csv(details / f"term_frequencies_{method}.csv",
                  ({"rank": rank, "term": term, "frequency": count} for rank, (term, count) in enumerate(ranked, 1)),
                  ["rank", "term", "frequency"])
    _, zipf = zipf_statistics(frequencies["custom"])
    review_ids = [d["review_id"] for d in documents]
    points = heaps_points(review_ids, token_counts, document_types, args.seed)
    heaps = heaps_statistics(points)
    seed_results = []
    all_points = []
    sampling = []
    for seed in range(args.seed, args.seed + 5):
        current = points if seed == args.seed else heaps_points(review_ids, token_counts, document_types, seed)
        stats = heaps_statistics(current)
        seed_results.append({"seed": seed, **stats["four_checkpoint_fit"],
                             "prediction_relative_error": stats["fit_10_30_50_predict_100"]["relative_error"]})
        all_points.extend(current)
        ordering = review_order(review_ids, seed)
        sampling.append({"seed": seed, "checkpoints": [
            {"fraction": p["fraction"], "reviews": p["reviews"],
             "review_ids_sha256": hashlib.sha256("\n".join(review_ids[i] for i in ordering[:p["reviews"]]).encode()).hexdigest()}
            for p in current if p["required_checkpoint"]
        ]})
    if points[-1]["tokens"] != frequencies["custom"].total() or points[-1]["types"] != len(frequencies["custom"]):
        raise AssertionError("Heaps full sample does not conserve corpus token/type counts")
    if not all(item.get("custom_passed", True) for item in examples):
        raise AssertionError("A synthetic tokenizer boundary case failed")
    write_csv(output / "tokenizer_comparison.csv", comparison, list(comparison[0]))
    write_csv(output / "heaps_samples.csv", [p for p in points if p["required_checkpoint"]], list(points[0]))
    law_parameters = [
        {"law": "Zipf", "coefficient": zipf["primary_fit"]["coefficient"],
         "exponent": zipf["primary_fit"]["exponent_s"], "r2_log10": zipf["primary_fit"]["r2_log10"],
         "fit_scope": "ranks 10-10000; frequency >= 2"},
        {"law": "Heaps", "coefficient": heaps["four_checkpoint_fit"]["k"],
         "exponent": heaps["four_checkpoint_fit"]["beta"], "r2_log10": heaps["four_checkpoint_fit"]["r2_log10"],
         "fit_scope": "10%, 30%, 50%, 100% reviews; N is normalized tokens"},
    ]
    write_csv(output / "law_parameters.csv", law_parameters, list(law_parameters[0]))
    report_examples = []
    for item in examples:
        if item["source"] == "real_corpus_excerpt" or item["feature"] in ("required_noise", "zwj_skin_flag_keycap"):
            report_examples.append({"source": item["source"], "feature": item["feature"], "text": item["text"],
                                    **{name: json.dumps(terms, ensure_ascii=False)
                                       for name, terms in item["normalized_terms"].items()}})
    write_csv(output / "tokenization_examples.csv", report_examples, list(report_examples[0]))
    write_csv(details / "heaps_samples.csv", all_points, list(points[0]))
    write_csv(details / "heaps_parameters.csv", seed_results, list(seed_results[0]))
    dump_json(details / "tokenization_examples.json", examples)
    dump_json(details / "sampling_manifest.json", {"method": "sort by SHA256(seed + NUL + review_id), then review_id; nested prefixes; ceil(review_count*fraction)", "seeds": sampling})
    summary = {
        "corpus": corpus,
        "policy": {"surface_stage": "NFC, ordered regex; retain punctuation, discard dash delimiters",
                   "normalization": "NFC, normalize apostrophes, lowercase lexical terms, expand clear contractions; retain numbers/emoji/emoticons; drop other punctuation-only tokens",
                   "stopwords": "retained", "stemming": False, "lemmatization": False,
                   "ambiguous_forms": "preserve 's, 'd and ain't; no POS/sense guessing",
                   "baseline_fairness": "Same reviews, NFC and normalize_tokens function. Expansion requires a complete recognized word token."},
        "tokenizers": comparison, "zipf": zipf, "heaps": heaps,
        "heaps_seed_sensitivity": {"seeds": [r["seed"] for r in seed_results],
                                   "beta_min": min(r["beta"] for r in seed_results),
                                   "beta_max": max(r["beta"] for r in seed_results)},
        "coverage": coverage,
        "synthetic_tests": {"cases": len(CASES), "passed": sum(e.get("custom_passed", False) for e in examples), "excluded_from_corpus_statistics": True},
        "sampling": {"primary_seed": args.seed, "fractions": [0.1, 0.3, 0.5, 1.0],
                     "unit": "reviews for subset selection; actual token count for Heaps x-axis",
                     "full_sample": "100% of the selected Meridian corpus, not the full Yelp download"},
        "environment": {"python": platform.python_version(), **{p: version(p) for p in ("regex", "nltk", "numpy", "matplotlib")}},
        "code_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for folder in ("text_processing", "part1")
                        for path in sorted((ROOT / folder).glob("*.py"))},
        "references": {"assignment": "GroupAssignment-202609.pdf pp.2-3, section 3.2",
                       "normalization_slides": "../slides/Lecture2-Text Normalization.pdf pp.4,11-13",
                       "laws_slides": "../slides/Lecture8-Index.pdf pp.21-26"},
    }
    dump_json(details / "summary.json", summary)
    dump_json(details / "regex_rules.json", [{"priority": i, "kind": name, "pattern": pattern} for i, (name, pattern) in enumerate(RULES, 1)])
    make_figures(output, frequencies, comparison, zipf, points, heaps)
    print(json.dumps({"output": str(output.relative_to(ROOT)) if output.is_relative_to(ROOT) else str(output),
                      "tokenizers": comparison, "zipf": zipf["primary_fit"],
                      "heaps": heaps, "coverage_reviews": {"emoji": coverage["custom_emoji_reviews"], "emoticons": coverage["custom_emoticon_reviews"]},
                      "elapsed_seconds": round(time.monotonic() - started, 1)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
