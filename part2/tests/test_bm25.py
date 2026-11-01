"""Hand-checked verification of the three BM25 variants (course / Okapi / Lucene).

Corpus A holds five documents and is used to check IDF and scores term by term.

    d1 "the pizza and the pasta"   the=2 pizza=1 and=1 pasta=1   dl=5
    d2 "pizza pizza"               pizza=2                       dl=2
    d3 "pasta and salad"           pasta=1 and=1 salad=1         dl=3
    d4 ""                          empty review                  dl=0
    d5 "sushi is great"            sushi=1 is=1 great=1          dl=3

    N = 5, avgdl = (5+2+3+0+3)/5 = 2.6
    df: pizza / and / pasta = 2; the / salad / sushi / is / great = 1

Corpus B holds four documents and exists purely to demonstrate Okapi clamping
(the occurs in 3 of the 4 documents, so df > N/2).

    d1 "the pizza"  d2 "the pasta"  d3 "the salad"  d4 "sushi"
    N = 4, avgdl = 7/4 = 1.75, df(the) = 3

The default parameters are k1 = 1.2 and b = 0.75. Every test below first spells out
the arithmetic by hand, then compares it against the code output value by value.
"""

import math
import unittest

from part2.bm25 import (
    COURSE,
    IDF_BY_VARIANT,
    LUCENE,
    OKAPI,
    bm25_search,
    course_idf,
    lucene_idf,
    okapi_idf,
    okapi_idf_unclamped,
    term_weight,
)
from part2.inverted_index import build_inverted_index

TOY_A = [
    {"review_id": "d1", "text": "the pizza and the pasta"},
    {"review_id": "d2", "text": "pizza pizza"},
    {"review_id": "d3", "text": "pasta and salad"},
    {"review_id": "d4", "text": ""},
    {"review_id": "d5", "text": "sushi is great"},
]

TOY_B = [
    {"review_id": "d1", "text": "the pizza"},
    {"review_id": "d2", "text": "the pasta"},
    {"review_id": "d3", "text": "the salad"},
    {"review_id": "d4", "text": "sushi"},
]

K1, B = 1.2, 0.75
AVGDL_A = 2.6


class IdfTests(unittest.TestCase):
    def test_three_idf_variants_on_the_toy_corpus(self):
        # With N = 5, terms with df = 2 (pizza / and / pasta) give:
        #   course ln(5/2) = 0.9163; okapi ln(3.5/2.5) = 0.3365; lucene ln(1 + 3.5/2.5) = 0.8755
        self.assertAlmostEqual(course_idf(2, 5), 0.9162907318741551)
        self.assertAlmostEqual(okapi_idf(2, 5), 0.3364722366212129)
        self.assertAlmostEqual(lucene_idf(2, 5), 0.8754687373538999)
        # Terms with df = 1 (the / salad / sushi / is / great):
        #   course ln(5/1) = 1.6094; okapi ln(4.5/1.5) = 1.0986; lucene ln(1 + 4.5/1.5) = 1.3863
        self.assertAlmostEqual(course_idf(1, 5), 1.6094379124341003)
        self.assertAlmostEqual(okapi_idf(1, 5), 1.0986122886681098)
        self.assertAlmostEqual(lucene_idf(1, 5), 1.3862943611198906)

    def test_terms_missing_from_the_index_get_zero_idf(self):
        self.assertEqual(course_idf(0, 5), 0.0)
        # Okapi's df = 0 branch is unreachable in practice (a query term absent from
        # the index is skipped entirely), so all we pin down here is that df = 0
        # does not raise.
        self.assertGreater(okapi_idf(0, 5), 0.0)

    def test_okapi_idf_goes_negative_before_clamping(self):
        # Corpus B: df(the) = 3 with N = 4, so df > N/2 and the raw IDF is negative:
        # ln((4-3+0.5)/(3+0.5)) = ln(1.5/3.5) = -0.8473, clamped to 0.
        raw = math.log((4 - 3 + 0.5) / (3 + 0.5))
        self.assertAlmostEqual(raw, -0.8472978603872037)
        self.assertLess(raw, 0.0)
        self.assertEqual(okapi_idf(3, 4), 0.0)
        # The other two variants stay positive on the same (df, N) pair — that is
        # exactly where the three of them substantively disagree.
        self.assertAlmostEqual(course_idf(3, 4), math.log(4 / 3))
        self.assertAlmostEqual(lucene_idf(3, 4), math.log(1 + 1.5 / 3.5))

    def test_course_and_lucene_idf_are_nearly_identical_for_rare_terms(self):
        # The Lucene form is ln((N+1)/(df+0.5)); when df << N it nearly coincides with
        # the course form ln(N/df). With N = 5 and df = 1 the term still counts as
        # "frequent" here and the gap is 0.22 — they only really converge at smaller df.
        self.assertAlmostEqual(course_idf(1, 5) - lucene_idf(1, 5), 0.22314355131420976)


class TermWeightTests(unittest.TestCase):
    def test_saturation_factor_matches_hand_computation(self):
        # d2 (dl = 2, tf = 2): 2.2 × 2 / (2 + 1.2 × (0.25 + 0.75 × 2/2.6)) = 4.4/2.9923
        self.assertAlmostEqual(term_weight(2, 2, AVGDL_A), 1.4704370179948587)

    def test_average_length_documents_are_neutral(self):
        # When dl = avgdl the bracket term equals 1, so length normalization does
        # nothing: weight = 2.2 × tf / (tf + 1.2).
        self.assertAlmostEqual(term_weight(1, AVGDL_A, AVGDL_A), 2.2 / 2.2)
        self.assertAlmostEqual(term_weight(5, AVGDL_A, AVGDL_A), 2.2 * 5 / 6.2)

    def test_k1_zero_collapses_to_binary(self):
        # k1 = 0 gives weight = (0+1)·tf / (tf + 0) = 1 regardless of how many times
        # the term occurs, collapsing to a simple "does it appear" signal.
        self.assertEqual(term_weight(1, 5, AVGDL_A, k1=0), 1.0)
        self.assertEqual(term_weight(20, 5, AVGDL_A, k1=0), 1.0)


class ToyRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.index = build_inverted_index(TOY_A)

    def test_course_variant_scores(self):
        # "pizza pasta": both terms have df = 2, so they share idf = ln 2.5.
        # d2 = idf × 2.2×2/(2 + 1.2×(0.25+0.75×2/2.6)) = idf × 1.4704
        # d1 = idf × 2 × 2.2/(1 + 1.2×(0.25+0.75×5/2.6)) = idf × 1.4516
        # d3 = idf × 2.2/(1 + 1.2×(0.25+0.75×3/2.6)) = idf × 0.9408
        results = bm25_search(self.index, "pizza pasta", variant=COURSE)
        self.assertEqual([doc for doc, _ in results], ["d2", "d1", "d3"])
        self.assertAlmostEqual(results[0][1], 1.3473478113933592)
        self.assertAlmostEqual(results[1][1], 1.3302494889137482)
        self.assertAlmostEqual(results[2][1], 0.8620366753816067)

    def test_okapi_variant_scores_are_on_a_different_scale(self):
        # Same documents, but swapping idf 0.9163 for 0.3365 rescales every score by
        # the same factor and leaves the ranking untouched — because both terms in
        # this query share the same df, so idf is a common factor.
        results = bm25_search(self.index, "pizza pasta", variant=OKAPI)
        self.assertEqual([doc for doc, _ in results], ["d2", "d1", "d3"])
        self.assertAlmostEqual(results[0][1], 0.4947612322553568)
        self.assertAlmostEqual(results[1][1], 0.48848253641455275)
        self.assertAlmostEqual(results[2][1], 0.3165495384002201)

    def test_lucene_variant_scores(self):
        results = bm25_search(self.index, "pizza pasta", variant=LUCENE)
        self.assertEqual([doc for doc, _ in results], ["d2", "d1", "d3"])
        self.assertAlmostEqual(results[0][1], 1.2873216395023928)
        self.assertAlmostEqual(results[1][1], 1.2709850704731744)
        self.assertAlmostEqual(results[2][1], 0.8236317726421559)

    def test_three_way_query_with_a_rare_term(self):
        # Adding salad, which occurs only in d3 (df = 1), pushes d3 to the top.
        # All three variants agree on the ordering.
        for variant in (COURSE, OKAPI, LUCENE):
            results = bm25_search(self.index, "pizza pasta salad", variant=variant)
            self.assertEqual([doc for doc, _ in results], ["d3", "d2", "d1"], variant)
        self.assertAlmostEqual(
            bm25_search(self.index, "pizza pasta salad", variant=COURSE)[0][1],
            2.376178921947899)

    def test_repeating_a_query_term_changes_nothing(self):
        # Queries use de-duplicated terms, matching how Lucene handles a TermQuery.
        self.assertEqual(bm25_search(self.index, "pizza pizza"),
                         bm25_search(self.index, "pizza"))

    def test_unknown_terms_and_empty_query(self):
        self.assertEqual(bm25_search(self.index, "gluten free"), [])
        self.assertEqual(bm25_search(self.index, ""), [])

    def test_empty_document_never_shows_up(self):
        self.assertNotIn("d4", [doc for doc, _ in bm25_search(self.index, "the sushi")])

    def test_top_k_truncates(self):
        self.assertEqual([doc for doc, _ in bm25_search(self.index, "pizza pasta", top_k=2)],
                         ["d2", "d1"])

    def test_unknown_variant_fails_loudly(self):
        with self.assertRaises(ValueError):
            bm25_search(self.index, "pizza", variant="bm25+")


class ParameterTests(unittest.TestCase):
    """Degenerate settings of k1 / b — the assignment asks us to "handle these two
    parameters correctly", so here is verifiable behaviour for each."""

    def setUp(self):
        self.index = build_inverted_index(TOY_A)

    def test_k1_zero_sum_of_idf_only(self):
        # k1 = 0 means each matched term contributes exactly its idf, independent of tf.
        # d1 matches two terms = 2 × 0.3365; d2 and d3 match one term each = 0.3365.
        results = bm25_search(self.index, "pizza pasta", k1=0, variant=OKAPI)
        self.assertEqual([doc for doc, _ in results], ["d1", "d2", "d3"])
        self.assertAlmostEqual(results[0][1], 2 * 0.3364722366212129)
        self.assertAlmostEqual(results[1][1], 0.3364722366212129)

    def test_b_zero_makes_the_longer_document_win(self):
        # b = 0 disables length normalization entirely: d1 (dl = 5), which matches two
        # terms, wins. Back at the default b = 0.75, the shorter d2 (dl = 2), matching
        # only one term, overtakes it.
        no_length_norm = bm25_search(self.index, "pizza pasta", b=0, variant=OKAPI)
        default = bm25_search(self.index, "pizza pasta", variant=OKAPI)
        self.assertEqual([doc for doc, _ in no_length_norm], ["d1", "d2", "d3"])
        self.assertEqual([doc for doc, _ in default], ["d2", "d1", "d3"])
        self.assertAlmostEqual(no_length_norm[0][1], 0.6729444732424258)

    def test_b_one_agrees_with_the_default_here(self):
        b_one = bm25_search(self.index, "pizza pasta", b=1, variant=OKAPI)
        self.assertEqual([doc for doc, _ in b_one], ["d2", "d1", "d3"])
        self.assertAlmostEqual(b_one[0][1], 0.5064792614403522)


class ClampingTests(unittest.TestCase):
    """Okapi must clamp once df > N/2 — the only qualitative difference between the
    three variants."""

    def setUp(self):
        self.index = build_inverted_index(TOY_B)

    def test_clamping_changes_the_result_set_not_just_the_scores(self):
        # Query "the pizza":
        #   course / lucene return 3 docs (d2, d3 each pick up a little score from the)
        #   okapi        returns only 1 — the's idf is clamped to 0 and the term is
        #                skipped whole, so d2 / d3, which contain nothing but the, end
        #                up with no score at all and drop out of the result set.
        course = bm25_search(self.index, "the pizza", variant=COURSE)
        lucene = bm25_search(self.index, "the pizza", variant=LUCENE)
        okapi = bm25_search(self.index, "the pizza", variant=OKAPI)
        self.assertEqual([doc for doc, _ in course], ["d1", "d2", "d3"])
        self.assertEqual([doc for doc, _ in lucene], ["d1", "d2", "d3"])
        self.assertEqual([doc for doc, _ in okapi], ["d1"])
        self.assertAlmostEqual(course[0][1], 1.5815482869327453)
        self.assertAlmostEqual(lucene[0][1], 1.474477013697908)
        self.assertAlmostEqual(okapi[0][1], 0.8005145429425117)

    def test_unclamped_okapi_would_reduce_the_score_of_a_matching_document(self):
        # Without clamping, d1 would score 0.8005 + (-0.8473) × 0.9448 = 0.0001 —
        # essentially zero. A document containing the query term would earn almost
        # nothing for it, which is precisely why clamping is necessary.
        idf_the = math.log((4 - 3 + 0.5) / (3 + 0.5))
        weight = term_weight(1, 2, 1.75)
        self.assertAlmostEqual(0.8005145429425117 + idf_the * weight, 0.0, places=6)


class UnclampedIdfTests(unittest.TestCase):
    """The unclamped Okapi IDF exists only for the ablation study; structurally it
    must stay unreachable from the default path.

    Half of these tests check the arithmetic, the other half guard the fact that it
    cannot be triggered by accident.
    """

    def setUp(self):
        self.index = build_inverted_index(TOY_B)

    def test_unclamped_is_the_raw_logarithm(self):
        # Corpus B: df(the) = 3, N = 4 → ln(1.5/3.5) = -0.8473, versus 0 clamped.
        self.assertAlmostEqual(okapi_idf_unclamped(3, 4), -0.8472978603872037)
        self.assertAlmostEqual(okapi_idf(3, 4), 0.0)
        # On the same (df, N) pair the two functions differ by exactly the clamped part.
        self.assertAlmostEqual(okapi_idf(3, 4) - okapi_idf_unclamped(3, 4),
                               0.8472978603872037)

    def test_clamping_is_a_no_op_when_df_is_small(self):
        # For df <= N / 2 the raw value is non-negative, so clamping changes
        # nothing value by value — clamping only affects frequent terms and never
        # quietly reweights the great majority of terms.
        for df in (1, 2):
            self.assertEqual(okapi_idf_unclamped(df, 5), okapi_idf(df, 5))
        # Edge case: at df = N / 2 the raw value is exactly 0; at df = (N + 1) / 2 it is
        # already negative.
        self.assertEqual(okapi_idf_unclamped(2, 4), 0.0)
        self.assertEqual(okapi_idf_unclamped(3, 5), -0.3364722366212129)

    def test_unclamped_is_not_registered_as_a_variant(self):
        # This is the core guarantee that it is never reached by default: it appears
        # on no key and no value of the variant table.
        self.assertNotIn(okapi_idf_unclamped, IDF_BY_VARIANT.values())
        self.assertEqual(sorted(IDF_BY_VARIANT), [COURSE, LUCENE, OKAPI])

    def test_no_variant_name_can_reach_the_unclamped_idf(self):
        # Guessing a name gets you nowhere, and passing the function itself as a name
        # gets you nowhere either — the variant channel is closed to it completely.
        for bad in ("okapi_raw", "okapi_unclamped", "raw"):
            with self.assertRaises(ValueError):
                bm25_search(self.index, "the pizza", variant=bad)
        with self.assertRaises(ValueError):
            bm25_search(self.index, "the pizza", variant=okapi_idf_unclamped)

    def test_only_the_explicit_idf_fn_channel_reaches_it(self):
        # Using it requires spelling the function out at the call site, so the reader
        # can see at a glance that this is an experimental configuration.
        results = bm25_search(self.index, "the pizza", idf_fn=okapi_idf_unclamped)
        # In corpus B df(the) = 3 and df(pizza) = 1, so the two IDF values are exact
        # opposites: d1, which matches both terms, cancels out to precisely 0, while
        # d2/d3, matching only the, receive negative scores.
        self.assertEqual([doc for doc, _ in results], ["d1", "d2", "d3"])
        self.assertAlmostEqual(results[0][1], 0.0)
        self.assertAlmostEqual(results[1][1], -0.8005145429425117)
        self.assertAlmostEqual(results[2][1], -0.8005145429425117)

    def test_idf_fn_takes_precedence_over_variant(self):
        # Passing it explicitly takes full precedence, whatever variant says — this
        # rules out silent mistakes like "I thought I was using course, but another
        # parameter quietly swapped it out".
        explicit = bm25_search(self.index, "the pizza", variant=COURSE,
                               idf_fn=okapi_idf_unclamped)
        self.assertAlmostEqual(explicit[0][1], 0.0)

    def test_the_clamped_and_unclamped_paths_diverge_on_result_sets(self):
        # What the ablation is really about: clamping does not merely rescale scores,
        # it changes which documents appear at all.
        #   clamped   d1 (0.8005): the's idf is 0, the term is skipped, d2/d3 drop out
        #   unclamped d1 (0.0), d2, d3 (-0.8005 each): the frequent term turns negative
        clamped = bm25_search(self.index, "the pizza", variant=OKAPI)
        unclamped = bm25_search(self.index, "the pizza", idf_fn=okapi_idf_unclamped)
        self.assertEqual([doc for doc, _ in clamped], ["d1"])
        self.assertEqual([doc for doc, _ in unclamped], ["d1", "d2", "d3"])
        # The top score drops from 0.8005 to 0: a document containing the query term
        # earns nothing for it, which is exactly the pathology.
        self.assertAlmostEqual(clamped[0][1], 0.8005145429425117)
        self.assertAlmostEqual(unclamped[0][1], 0.0)

    def test_the_default_path_is_untouched_by_any_of_this(self):
        # On the deliverable path (no idf_fn passed) the three variants return results
        # identical value by value to what they produced before this hook existed.
        expected = {
            COURSE: (["d1", "d2", "d3"], 1.5815482869327453),
            LUCENE: (["d1", "d2", "d3"], 1.474477013697908),
            OKAPI: (["d1"], 0.8005145429425117),
        }
        for variant, (docs, top_score) in expected.items():
            results = bm25_search(self.index, "the pizza", variant=variant)
            self.assertEqual([doc for doc, _ in results], docs, variant)
            self.assertAlmostEqual(results[0][1], top_score, msg=variant)
        # With no IDF-related argument at all, the clamped Okapi variant runs — the
        # unclamped one can never be what you get by default.
        self.assertEqual(bm25_search(self.index, "the pizza"),
                         bm25_search(self.index, "the pizza", variant=OKAPI))


if __name__ == "__main__":
    unittest.main()
