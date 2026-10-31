"""Hand-checked verification of TF-IDF cosine scoring (lnc.ltc and ltc.ltc).

The corpus is the same four short documents used in test_inverted_index.py, with N = 4:

    d1 "the pizza and the pasta"   the=2 pizza=1 and=1 pasta=1
    d2 "pizza pizza"               pizza=2
    d3 "pasta and salad"           pasta=1 and=1 salad=1
    d4 ""                          empty review

    df:   the=1 salad=1 | pizza=2 and=2 pasta=2
    idf:  df=1 → ln 4 = 1.3863; df=2 → ln 2 = 0.6931

Every test below first spells out the arithmetic, then compares it against the code
output value by value; these are the numbers we defend in the report.
"""

import math
import unittest

from part2.inverted_index import build_inverted_index
from part2.tfidf import LNC_LTC, LTC_LTC, query_weights, tfidf_search

TOY_CORPUS = [
    {"review_id": "d1", "text": "the pizza and the pasta"},
    {"review_id": "d2", "text": "pizza pizza"},
    {"review_id": "d3", "text": "pasta and salad"},
    {"review_id": "d4", "text": ""},
]

LOG2 = math.log(2)   # idf(pizza) = idf(and) = idf(pasta)
LOG4 = math.log(4)   # idf(the) = idf(salad)


class ToyRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.index = build_inverted_index(TOY_CORPUS)

    def test_query_weights_are_log_tf_times_idf(self):
        # Both variants agree on the query side: w = (1 + ln tf_q) · ln(N/df).
        # "the pizza": each term appears once in the query, so the tf factor is 1
        # and the weight is just the term's idf.
        self.assertAlmostEqual(query_weights(self.index, "the pizza".split()),
                               {"the": LOG4, "pizza": LOG2})
        # Repeating a term in the query gives tf = 2, so the weight is multiplied by
        # (1 + ln 2), not by 2.
        self.assertAlmostEqual(query_weights(self.index, "pizza pizza".split()),
                               {"pizza": (1 + math.log(2)) * LOG2})
        # A term absent from the index has df = 0 → idf is recorded as 0, and the term
        # takes no part in scoring.
        self.assertEqual(query_weights(self.index, ["sushi"]), {"sushi": 0.0})

    # ---------- lnc.ltc ----------

    def test_lnc_ltc_single_term_query(self):
        # With a single query term the idf cancels between numerator and denominator,
        # so the cosine depends only on document-side tf:
        # d2 contains nothing but pizza, w = 1 + ln 2, ‖d2‖_lnc = 1 + ln 2 → cosine = 1.
        # d1 contains pizza once → w = 1, ‖d1‖_lnc = √((1+ln2)² + 3) → cosine = 1/‖d1‖.
        results = tfidf_search(self.index, "pizza", variant=LNC_LTC)
        self.assertEqual([doc_id for doc_id, _ in results], ["d2", "d1"])
        self.assertAlmostEqual(results[0][1], 1.0)
        self.assertAlmostEqual(results[1][1], 1 / math.sqrt((1 + LOG2) ** 2 + 3))
        self.assertAlmostEqual(results[1][1], 0.4128585720620119)

    def test_lnc_ltc_two_term_query(self):
        # q = (ln2, ln2), ‖q‖ = √2·ln2 → after normalization each term carries 1/√2.
        # d2 holds pizza only: 1/√2 × 1 = 0.7071
        # d1 holds pizza and pasta once each: 1/√2 × (1+1)/‖d1‖_lnc = 1/√2 × 2/2.4221
        # d3 holds pasta only: 1/√2 × 1/√3 = 0.4082
        results = tfidf_search(self.index, "pizza pasta", variant=LNC_LTC)
        self.assertEqual([doc_id for doc_id, _ in results], ["d2", "d1", "d3"])
        self.assertAlmostEqual(results[0][1], 1 / math.sqrt(2))
        self.assertAlmostEqual(results[2][1], 1 / math.sqrt(6))
        self.assertAlmostEqual(results[1][1], 0.583870191952087)

    def test_lnc_ltc_terms_with_low_idf_carry_less_weight(self):
        # "pizza and the": pizza / and have idf = ln2, while the has idf = ln4
        # (it occurs only in d1). d1 contains all three terms, and the rarest of them
        # carries the largest weight, so d1 ranks first.
        results = tfidf_search(self.index, "pizza and the", variant=LNC_LTC)
        self.assertEqual([doc_id for doc_id, _ in results], ["d1", "d2", "d3"])
        self.assertAlmostEqual(results[0][1], 0.9078534846653029)

    # ---------- ltc.ltc ----------

    def test_ltc_ltc_single_term_query(self):
        # Switching the document side to (1+ln tf)·idf gives ‖d2‖_ltc = (1+ln2)·ln2,
        # which equals the numerator, so the cosine is still 1.
        # ‖d1‖_ltc = √([(1+ln2)·ln4]² + 3·ln2²) is much larger than under lnc, so the
        # same term scores lower on d1: 0.4129 → 0.2629.
        results = tfidf_search(self.index, "pizza", variant=LTC_LTC)
        self.assertEqual([doc_id for doc_id, _ in results], ["d2", "d1"])
        self.assertAlmostEqual(results[0][1], 1.0)
        self.assertAlmostEqual(results[1][1], 0.2629123075078803)

    def test_ltc_ltc_two_term_query(self):
        results = tfidf_search(self.index, "pizza pasta", variant=LTC_LTC)
        self.assertEqual([doc_id for doc_id, _ in results], ["d2", "d1", "d3"])
        self.assertAlmostEqual(results[1][1], 0.37181415099245)
        self.assertAlmostEqual(results[2][1], 0.28867513459481287)

    def test_variants_can_disagree_on_ranking(self):
        #   lnc.ltc  d1 0.7151 > d3 0.6667 > d2 0.5774 — d1, matching all three query
        #            terms, ranks first
        #   ltc.ltc  d2 0.5774 > d3 0.4714 > d1 0.4554 — d2, with a single term, tops it
        # Mechanism: d1 also holds the, which is missing from the query and rare
        # (idf = ln4). Under lnc it adds 1+ln2 = 1.69 to ‖d1‖; under ltc that becomes
        # 1.69·ln4 = 2.35. It stretches only the denominator and never the numerator, so
        # d1's cosine collapses; d2 holds pizza alone, so its direction is immune to idf
        # and it wins by default.
        lnc = tfidf_search(self.index, "pizza and pasta", variant=LNC_LTC)
        ltc = tfidf_search(self.index, "pizza and pasta", variant=LTC_LTC)
        self.assertEqual([doc_id for doc_id, _ in lnc], ["d1", "d3", "d2"])
        self.assertEqual([doc_id for doc_id, _ in ltc], ["d2", "d3", "d1"])
        self.assertAlmostEqual(lnc[0][1], 0.7150920231517413)
        self.assertAlmostEqual(ltc[0][1], 0.5773502691896257)

    def test_variants_can_also_agree(self):
        # On another query the two variants agree completely — the difference is not a
        # matter of "which weighting is more correct", but of how idf is spread across
        # matched and unmatched terms. d2 has a single term, so after normalization its
        # direction is always that term's basis vector and both weightings give 1/√2.
        lnc = tfidf_search(self.index, "pizza pasta", variant=LNC_LTC)
        ltc = tfidf_search(self.index, "pizza pasta", variant=LTC_LTC)
        self.assertEqual([doc_id for doc_id, _ in lnc], ["d2", "d1", "d3"])
        self.assertEqual([doc_id for doc_id, _ in ltc], ["d2", "d1", "d3"])
        self.assertAlmostEqual(lnc[0][1], ltc[0][1])

    # ---------- edge cases ----------

    def test_repeating_a_term_in_the_query_changes_nothing(self):
        # Repeating a term only scales the query vector itself; ‖q‖ grows by the same
        # factor and normalization cancels it out — see the query_weights test above.
        self.assertEqual(tfidf_search(self.index, "pizza pizza", variant=LNC_LTC),
                         tfidf_search(self.index, "pizza", variant=LNC_LTC))

    def test_unknown_query_terms_return_nothing(self):
        self.assertEqual(tfidf_search(self.index, "sushi"), [])
        self.assertEqual(tfidf_search(self.index, ""), [])

    def test_empty_document_never_shows_up(self):
        # d4 has ‖d‖ = 0, so the cosine denominator vanishes; it must be skipped rather
        # than dividing by zero.
        results = tfidf_search(self.index, "the salad", variant=LNC_LTC)
        self.assertNotIn("d4", [doc_id for doc_id, _ in results])

    def test_top_k_truncates(self):
        results = tfidf_search(self.index, "pizza and the", top_k=2, variant=LNC_LTC)
        self.assertEqual([doc_id for doc_id, _ in results], ["d1", "d2"])

    def test_unknown_variant_fails_loudly(self):
        with self.assertRaises(ValueError):
            tfidf_search(self.index, "pizza", variant="ltc.lnc")


if __name__ == "__main__":
    unittest.main()
