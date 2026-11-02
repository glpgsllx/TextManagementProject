"""Tests for the comparison detector (no external data needed)."""

import unittest

from part4.detect import BusinessComparisonDetector, comparator_span, name_key
from part4.sentences import split_sentences

BUSINESSES = ["Flying Pie Pizzaria", "Le Peep", "Pizza Hut", "Barrel 55", "Costa Vida"]


class SentenceSplitTests(unittest.TestCase):
    def test_splits_on_end_punctuation(self):
        self.assertEqual(
            split_sentences("The pizza was good. The service was better! Will we return?"),
            ["The pizza was good.", "The service was better!", "Will we return?"],
        )

    def test_keeps_abbreviations_and_decimals(self):
        sentences = split_sentences("We met Dr. Smith at 4.50 on Main St. and ate.")
        self.assertEqual(len(sentences), 1)

    def test_newline_is_a_boundary(self):
        self.assertEqual(split_sentences("Great food\nSlow service"), ["Great food", "Slow service"])


class DetectorTests(unittest.TestCase):
    def setUp(self):
        self.detector = BusinessComparisonDetector(BUSINESSES)

    def kind(self, sentence, own="Barrel 55"):
        result = self.detector.analyze(sentence, own)
        return None if result is None else result.kind

    def test_named_competitor_is_a_business_comparison(self):
        self.assertEqual(self.kind("Way better than Flying Pie Pizzaria."), "business")
        self.assertEqual(self.kind("Cheaper than Pizza Hut for the same pizza."), "business")

    def test_place_phrase_is_a_business_comparison(self):
        self.assertEqual(self.kind("Much better than the one downtown."), "business")
        self.assertEqual(self.kind("More options than other places in town."), "business")

    def test_scoped_superlative_is_its_own_class(self):
        self.assertEqual(self.kind("The best pizza in town."), "superlative")
        self.assertEqual(self.kind("The best burger I've ever had."), "superlative")

    def test_idioms_are_not_comparisons(self):
        for sentence in ("Other than that, it was fine.",
                         "They were more than happy to help.",
                         "We waited less than an hour.",
                         "The food was better than expected."):
            # No cue at all (None) or a cue that the idiom filter rejects.
            self.assertIn(self.kind(sentence), (None, "other"), sentence)

    def test_item_comparison_is_not_a_business_comparison(self):
        self.assertEqual(self.kind("The pizza was better than the pasta."), "other")

    def test_no_cue_means_no_result(self):
        self.assertIsNone(self.detector.analyze("The pizza was excellent.", "Barrel 55"))

    def test_comparator_span(self):
        self.assertEqual(comparator_span("much better than the place on main street"),
                         "the place on main street")

    def test_name_key_folds_case_and_punctuation(self):
        self.assertEqual(name_key("McDonald's"), "mcdonalds")
        self.assertEqual(name_key("Le Peep"), "le peep")


if __name__ == "__main__":
    unittest.main()
