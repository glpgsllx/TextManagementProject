"""排名一致性指标的手算核对（overlap@k 与 Kendall tau-b）。

指标实现错了就会得出错结论，所以这里把每一条性质都还原成能手算的例子：
同序必须得 1、逆序必须得 −1、交换一对必须得 4/6、全并列必须算不出来。
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
    """[(doc_id, score)] 的简写：分数只用来保持降序，数值随手给。"""
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
        self.assertEqual(overlap_at_k(left, right, 2), 1)   # 只比前 2 条：{a,b} ∩ {a,x} = {a}
        self.assertEqual(overlap_at_k(left, right, 3), 2)   # 前 3 条：{a,b,c} ∩ {a,x,c} = {a,c}
        self.assertEqual(overlap_at_k(left, right, 4), 3)   # 前 4 条：少了 x

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
        # x = a b c d，y = a c b d：只交换一对
        # 一致对 5、不一致对 1、n0 = 6 → τ = (5 − 1) / 6
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

        x = {"a": 1, "b": 1, "c": 3}      # 左侧 a、b 并列
        y = {"a": 1, "b": 2, "c": 3}
        with_ties = kendall_tau_b(x, y)
        self.assertEqual(with_ties["n1"], 1)
        self.assertIsNone(with_ties["z"])
        self.assertIsNone(with_ties["p_two_sided"])

    def test_all_ties_on_one_side_is_undefined(self):
        # 一侧全部并列时，这一侧提供不了任何方向信息，分母为 0 → None
        x = {"a": 1, "b": 1, "c": 1}
        y = {"a": 1, "b": 2, "c": 3}
        self.assertIsNone(kendall_tau_b(x, y))

    def test_too_few_documents_is_undefined(self):
        self.assertIsNone(kendall_tau_b({"a": 1}, {"a": 1}))


class ComparePairTests(unittest.TestCase):
    def test_pool_defines_union_and_intersection(self):
        # 两侧各取前 3 条：左 {a,b,c}、右 {a,x,c} → 并集 4、交集 2
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
        # 左侧召回了 d，右侧压根没召回（BM25 把 idf 钳成 0 的词项整个跳过时就会这样）
        left = ranked([("a", 3), ("b", 2), ("d", 1)])
        right = ranked([("b", 3), ("a", 2)])
        result = compare_pair(left, right, ks=(3,), pool=3)
        self.assertEqual(result["n_union"], 3)
        self.assertEqual(result["n_missing_b"], 1)   # d 在右侧缺席
        self.assertEqual(result["n_missing_a"], 0)
        self.assertEqual(result["n_intersection"], 2)
        # d 在右侧记为"最末名之后"，于是 a、b 的相对顺序仍然可比、τ 不为 None
        self.assertIsNotNone(result["tau_b_union"])


class RankMapTests(unittest.TestCase):
    def test_rank_starts_at_one_and_follows_input_order(self):
        self.assertEqual(rank_map(ranked([("a", 5), ("b", 4)])), {"a": 1, "b": 2})


if __name__ == "__main__":
    unittest.main()
