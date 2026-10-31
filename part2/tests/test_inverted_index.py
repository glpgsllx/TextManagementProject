import math
import unittest

from part2.inverted_index import (
    build_inverted_index,
    inverse_document_frequency,
    tfidf_weight,
)

# A tiny corpus for hand-checking: 4 documents whose terms are all plain English words,
# so the tokenizer has nothing to expand.
TOY_CORPUS = [
    {"review_id": "d1", "text": "the pizza and the pasta"},
    {"review_id": "d2", "text": "pizza pizza"},
    {"review_id": "d3", "text": "pasta and salad"},
    {"review_id": "d4", "text": ""},
]


class InvertedIndexTests(unittest.TestCase):
    def setUp(self):
        self.index = build_inverted_index(TOY_CORPUS)

    def test_postings_hold_term_frequency(self):
        self.assertEqual(self.index.postings, {
            "the": {"d1": 2},
            "pizza": {"d1": 1, "d2": 2},
            "and": {"d1": 1, "d3": 1},
            "pasta": {"d1": 1, "d3": 1},
            "salad": {"d3": 1},
        })

    def test_document_lengths_and_global_averages(self):
        # dl counts tokens after tokenization; "the" twice in d1 counts as two tokens.
        self.assertEqual(self.index.doc_lengths, {"d1": 5, "d2": 2, "d3": 3, "d4": 0})
        self.assertEqual(self.index.total_tokens, 10)
        self.assertEqual(len(self.index.doc_lengths), 4)
        self.assertEqual(len(self.index.postings), 5)
        self.assertAlmostEqual(self.index.avg_doc_length, 2.5)

    def test_df_is_the_length_of_postings(self):
        self.assertEqual(self.index.df("pizza"), 2)
        self.assertEqual(self.index.df("the"), 1)
        self.assertEqual(self.index.df("missing"), 0)

    def test_lnc_norm_ignores_idf(self):
        # lnc: the document-side weight is 1 + ln tf, with no idf factor. N = 4.
        # d1 = (1+ln2)^2 + 3·1^2; d2 = (1+ln2)^2; d3 = 1^2 + 1^2 + 1^2; d4 is empty.
        w = 1 + math.log(2)
        self.assertAlmostEqual(self.index.doc_norms_lnc["d1"], math.sqrt(w * w + 3))
        self.assertAlmostEqual(self.index.doc_norms_lnc["d2"], w)
        self.assertAlmostEqual(self.index.doc_norms_lnc["d3"], math.sqrt(3))
        self.assertAlmostEqual(self.index.doc_norms_lnc["d1"], 2.4221369439067835)
        self.assertAlmostEqual(self.index.doc_norms_lnc["d2"], 1.6931471805599454)
        self.assertAlmostEqual(self.index.doc_norms_lnc["d3"], 1.7320508075688772)

    def test_ltc_norm_multiplies_idf(self):
        # ltc: the document-side weight is (1+ln tf)·idf, where terms with df=1 have
        # idf = ln4 and terms with df=2 have idf = ln2.
        # d1 = [(1+ln2)·ln4]² + 3·(ln2)²; d2 = [(1+ln2)·ln2]²; d3 = 2·(ln2)² + (ln4)².
        idf1, idf2 = math.log(4), math.log(2)
        self.assertAlmostEqual(
            self.index.doc_norms_ltc["d1"],
            math.sqrt(((1 + math.log(2)) * idf1) ** 2 + 3 * idf2 ** 2))
        self.assertAlmostEqual(self.index.doc_norms_ltc["d1"], 2.636419675937649)
        self.assertAlmostEqual(self.index.doc_norms_ltc["d2"], 1.1736001944781467)
        self.assertAlmostEqual(self.index.doc_norms_ltc["d3"], 1.6978569090206654)

    def test_the_two_norm_sets_are_different(self):
        # The two weightings produce different document vectors and therefore different
        # norms, so they must always be used as a matched pair when scoring.
        self.assertNotAlmostEqual(self.index.doc_norms_lnc["d1"],
                                  self.index.doc_norms_ltc["d1"])

    def test_every_document_has_a_norm_including_empty_ones(self):
        self.assertEqual(sorted(self.index.doc_norms_lnc), ["d1", "d2", "d3", "d4"])
        self.assertEqual(sorted(self.index.doc_norms_ltc), ["d1", "d2", "d3", "d4"])
        self.assertEqual(self.index.doc_norms_lnc["d4"], 0.0)
        self.assertEqual(self.index.doc_norms_ltc["d4"], 0.0)


class WeightTests(unittest.TestCase):
    def test_idf_handles_unknown_and_universal_terms(self):
        # df = 0 means the query term is absent from the index; df = N means the term
        # occurs in every document and therefore carries no discriminative power.
        self.assertEqual(inverse_document_frequency(0, 100), 0.0)
        self.assertEqual(inverse_document_frequency(100, 100), 0.0)
        self.assertAlmostEqual(inverse_document_frequency(50, 100), math.log(2))

    def test_tfidf_weight_is_sublinear_in_tf(self):
        idf = math.log(2)
        # At tf = 1 the weight is simply the idf itself.
        self.assertAlmostEqual(tfidf_weight(1, idf), idf)
        self.assertAlmostEqual(tfidf_weight(20, idf), (1 + math.log(20)) * idf)


class EmptyCorpusTests(unittest.TestCase):
    def test_building_without_documents_fails_loudly(self):
        with self.assertRaises(ValueError):
            build_inverted_index([])


if __name__ == "__main__":
    unittest.main()
