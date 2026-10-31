"""Hand-checked verification of the rank agreement metrics (overlap@k, Kendall tau-b).

A metric implemented wrongly produces wrong conclusions, so every property here is
reduced to a hand-computable example: identical order must give 1, reversed order must
give -1, a single swap must give 4/6, and an all-ties input must come back undefined.
"""

import unittest

from part2.rank_metrics import (
    compare_pair,
    kendall_tau_b,
    overlap_at_k,
    rank_map,
    self_check,
)


def ranked(pairs):
    """Shorthand for building [(doc_id, score)] lists: the score only fixes the
    descending order, so the values are arbitrary."""
    return list(pairs)


class OverlapTests(unittest.TestCase):
    def test_identical_lists_overlap_fully(self):
        left = ranked([("a", 3), ("b", 2), ("c", 1)])
        right = ranked([("a", 9), ("b", 8), ("c", 7)])
        self.assertEqual(overlap_at_k(left, right, 3), 3)

    def test_disjoint_lists_overlap_zero(self):
        left = ranked([("a", 3), ("b", 2)])
        right = ranked([("x", 3), ("y", 2)])
        self.assertEqual(overlap_at_k(left, right, 2), 0)

    def test_partial_overlap_and_truncation(self):
        left = ranked([("a", 5), ("b", 4), ("c", 3), ("d", 2)])
        right = ranked([("a", 5), ("x", 4), ("c", 3), ("b", 2)])
        self.assertEqual(overlap_at_k(left, right, 2), 1)   # top 2 only: {a,b} ∩ {a,x} = {a}
        self.assertEqual(overlap_at_k(left, right, 3), 2)   # top 3: {a,b,c} ∩ {a,x,c} = {a,c}
        self.assertEqual(overlap_at_k(left, right, 4), 3)   # top 4: everything except x

    def test_overlap_never_exceeds_available_results(self):
        left = ranked([("a", 1)])
        right = ranked([("a", 1)])
        self.assertEqual(overlap_at_k(left, right, 10), 1)


class KendallTauTests(unittest.TestCase):
    def test_self_check_constants(self):
        result = self_check()
        self.assertAlmostEqual(result["identical"], 1.0)
        self.assertAlmostEqual(result["reversed"], -1.0)
        self.assertAlmostEqual(result["hand_check"], 4 / 6)

    def test_identical_ordering_gives_one(self):
        order = {d: i for i, d in enumerate("abcdefgh", 1)}
        self.assertAlmostEqual(kendall_tau_b(order, dict(order))["tau"], 1.0)

    def test_reversed_ordering_gives_minus_one(self):
        docs = "abcdefgh"
        order = {d: i for i, d in enumerate(docs, 1)}
        reverse = {d: len(docs) - i + 1 for i, d in enumerate(docs, 1)}
        self.assertAlmostEqual(kendall_tau_b(order, reverse)["tau"], -1.0)

    def test_single_swap_matches_hand_computation(self):
        # x = a b c d, y = a c b d: exactly one pair is swapped
        # 5 concordant pairs, 1 discordant, n0 = 6 → tau = (5 - 1) / 6
        x = {"a": 1, "b": 2, "c": 3, "d": 4}
        y = {"a": 1, "b": 3, "c": 2, "d": 4}
        result = kendall_tau_b(x, y)
        self.assertEqual(result["nc"], 5)
        self.assertEqual(result["nd"], 1)
        self.assertEqual(result["n0"], 6)
        self.assertAlmostEqual(result["tau"], 4 / 6)

    def test_p_value_only_reported_without_ties(self):
        docs = "abcdefgh"
        order = {d: i for i, d in enumerate(docs, 1)}
        no_ties = kendall_tau_b(order, dict(order))
        self.assertIsNotNone(no_ties["z"])
        self.assertIsNotNone(no_ties["p_two_sided"])

        x = {"a": 1, "b": 1, "c": 3}      # a and b tie on the left
        y = {"a": 1, "b": 2, "c": 3}
        with_ties = kendall_tau_b(x, y)
        self.assertEqual(with_ties["n1"], 1)
        self.assertIsNone(with_ties["z"])
        self.assertIsNone(with_ties["p_two_sided"])

    def test_all_ties_on_one_side_is_undefined(self):
        # If one side is entirely tied it carries no ordering information at all, so the
        # denominator is 0 and the result is None.
        x = {"a": 1, "b": 1, "c": 1}
        y = {"a": 1, "b": 2, "c": 3}
        self.assertIsNone(kendall_tau_b(x, y))

    def test_too_few_documents_is_undefined(self):
        self.assertIsNone(kendall_tau_b({"a": 1}, {"a": 1}))


class ComparePairTests(unittest.TestCase):
    def test_pool_defines_union_and_intersection(self):
        # Top 3 from each side: left {a,b,c}, right {a,x,c} → union 4, intersection 2
        left = ranked([("a", 3), ("b", 2), ("c", 1), ("d", 0)])
        right = ranked([("a", 3), ("x", 2), ("c", 1), ("b", 0)])
        result = compare_pair(left, right, ks=(1, 2, 3), pool=3)
        self.assertEqual(result["n_union"], 4)
        self.assertEqual(result["n_intersection"], 2)
        self.assertEqual(result["overlap"], {"@1": 1, "@2": 1, "@3": 2})
        self.assertTrue(result["top1_same"])

    def test_empty_result_is_reported_not_crashed(self):
        result = compare_pair([], [], ks=(10,), pool=50)
        self.assertIsNone(result["tau_b_union"])
        self.assertIsNone(result["tau_b_intersection"])
        self.assertFalse(result["top1_same"])
        self.assertEqual(result["overlap"]["@10"], 0)

    def test_different_top1_is_flagged(self):
        left = ranked([("a", 3)])
        right = ranked([("b", 3)])
        self.assertFalse(compare_pair(left, right, ks=(1,), pool=1)["top1_same"])

    def test_documents_missing_on_one_side_are_ranked_last(self):
        # The left-hand side retrieves d; the right-hand side does not retrieve it at
        # all (this happens in BM25 when a term whose idf clamps to 0 is skipped whole).
        left = ranked([("a", 3), ("b", 2), ("d", 1)])
        right = ranked([("b", 3), ("a", 2)])
        result = compare_pair(left, right, ks=(3,), pool=3)
        self.assertEqual(result["n_union"], 3)
        self.assertEqual(result["n_missing_b"], 1)   # d absent on the right
        self.assertEqual(result["n_missing_a"], 0)
        self.assertEqual(result["n_intersection"], 2)
        # d is recorded as "after last place" on the right, so the relative order of a
        # and b stays comparable and tau is still not None.
        self.assertIsNotNone(result["tau_b_union"])


class RankMapTests(unittest.TestCase):
    def test_rank_starts_at_one_and_follows_input_order(self):
        self.assertEqual(rank_map(ranked([("a", 5), ("b", 4)])), {"a": 1, "b": 2})


if __name__ == "__main__":
    unittest.main()
