import math
import unittest

import numpy as np

from part3.cooccurrence import (
    analyze_term, build_corpus, is_candidate, jaccard, merge_periods, pmi, rank_words,
    sampling_stability, theme_rates, tracked_words, window_neighbors, window_sensitivity,
)


class WindowTests(unittest.TestCase):
    def test_window_stops_at_review_edges_and_skips_target(self):
        tokens = "a b service c service d e f".split()
        self.assertEqual(window_neighbors(tokens, "service", 1), ["b", "c", "c", "d"])
        self.assertEqual(window_neighbors(["service"], "service", 5), [])

    def test_candidate_filter(self):
        self.assertTrue(is_candidate("slow"))
        self.assertFalse(any(is_candidate(w) for w in ("the", "ok", "she's", "10")))
        self.assertFalse(is_candidate("customer", ("customer",)))


class PmiTests(unittest.TestCase):
    def test_hand_computed_pmi(self):
        # 100 tokens; target 10 times, word 5 times, always together: log2(4 * 100 / (5 * 10)) = 3.
        self.assertAlmostEqual(pmi(4, 5, 10, 100), 3.0)

    def test_rank_words_matches_formula_and_filters(self):
        documents = [("b1", "service slow x y".split()), ("b2", "service slow rare z".split()),
                     ("b3", "slow service tasty q".split()), ("b1", "tasty tasty tasty tasty".split())]
        vocabulary = {}
        corpus = build_corpus(documents, "service", 1, vocabulary)
        allowed = np.ones(len(vocabulary), dtype=bool)
        ranked, scores, pairs, businesses = rank_words(corpus, allowed, 2, 3)
        self.assertEqual(list(ranked), [vocabulary["slow"]])
        self.assertAlmostEqual(float(scores[0]), math.log2(3 * 16 / (3 * 3)))
        self.assertEqual((pairs[vocabulary["slow"]], businesses[vocabulary["slow"]]), (3, 3))
        self.assertEqual(len(rank_words(corpus, allowed, 2, 4)[0]), 0)

    def test_weights_equal_duplicated_reviews(self):
        documents = [("b1", "service slow a".split()), ("b2", "service good b".split())]
        doubled = [documents[0], documents[0], documents[1]]
        vocabulary = {}
        corpus = build_corpus(documents, "service", 2, vocabulary)
        allowed = np.ones(len(vocabulary), dtype=bool)
        weighted = rank_words(corpus, allowed, 1, 1, np.array([2, 1]))[1]
        repeated = rank_words(build_corpus(doubled, "service", 2, dict(vocabulary)), allowed, 1, 1)[1]
        self.assertTrue(np.allclose(weighted, repeated))

    def test_target_absent_returns_nothing(self):
        vocabulary = {}
        corpus = build_corpus([("b1", "no match here".split())], "service", 2, vocabulary)
        self.assertEqual(len(rank_words(corpus, np.ones(len(vocabulary), dtype=bool), 1, 1)[0]), 0)


class DriftTests(unittest.TestCase):
    def setUp(self):
        self.by_year = {
            2015: [(f"b{i}", "great service slow food".split()) for i in range(3)],
            2016: [(f"b{i}", "bad service fast food".split()) for i in range(3)],
        }

    def test_jaccard(self):
        self.assertEqual(jaccard("abc", "bcd"), 0.5)
        self.assertEqual(jaccard((), ()), 0.0)

    def test_analyze_term_reports_new_words_and_stays_reproducible(self):
        run = lambda: analyze_term(self.by_year, "service", 1, 5, 1, 1, np.random.default_rng(1), 5)
        top, drift, rankings = run()
        self.assertEqual({r["word"] for r in top if r["period"] == 2015}, {"great", "slow"})
        self.assertEqual(drift[1]["new_since_previous"].split(), ["bad", "fast"] if top[2]["word"] == "bad" else ["fast", "bad"])
        self.assertEqual(drift[1]["jaccard_previous"], 0.0)
        self.assertEqual(run()[:2], (top, drift))
        rows = tracked_words(rankings, "service", ["slow"])
        self.assertEqual([r["cooccurrences"] for r in rows], [3, ""])

    def test_identical_copy_has_full_stability_with_constant_reviews(self):
        vocabulary = {}
        corpus = build_corpus(self.by_year[2015], "service", 1, vocabulary)
        allowed = np.ones(len(vocabulary), dtype=bool)
        self.assertEqual(sampling_stability(corpus, allowed, 5, 1, 1, np.random.default_rng(0), 3), 1.0)

    def test_merge_periods_conserves_reviews(self):
        merged = merge_periods(self.by_year, {"early": (2015, 2015), "all": (2015, 2016)})
        self.assertEqual((len(merged["early"]), len(merged["all"])), (3, 6))

    def test_theme_rates_per_hundred_mentions(self):
        rows = theme_rates(self.by_year, "service", {"Speed": frozenset({"slow", "fast"})}, 1, np.random.default_rng(0), 20)
        self.assertEqual([(r["mentions"], r["cooccurrences"], r["per_100_mentions"]) for r in rows], [(3, 3, 100.0)] * 2)

    def test_baseline_window_agrees_with_itself(self):
        rows = window_sensitivity(self.by_year, "service", (1, 2), 2, 5, 1, 1)
        self.assertTrue(all(r["jaccard_with_baseline"] == 1.0 for r in rows if r["window"] == 2))


if __name__ == "__main__":
    unittest.main()
