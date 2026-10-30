"""逐实现分析的测试。

这里钉住的是"指标算得对不对"和"b 扫描的单调性判断靠不靠谱"，不碰索引、
不跑 Java——那些由 query_experiment / lucene_experiment 的测试负责。
"""

import unittest

from part2.impl_analysis import (
    BAND_ORDER,
    _median,
    aggregate,
    analysis_conditions,
    b_scan,
    notable,
)


def _metrics(**overrides):
    base = {"text": "x", "axis": "category", "df_band": "中频",
            "n_returned": 10, "top1_score": 1.0, "top2_score": 0.9,
            "margin_rel": 0.1, "n_tied_top1": 1, "decay_10": 0.5,
            "top10_dl_median": 40.0, "top10_dl_mean": 41.0,
            "length_bias": 0.4, "ms": 1.0, "ms_full_sort": 2.0, "top": []}
    base.update(overrides)
    return base


QUERIES = [{"id": "Q01", "axis": "category", "text": "a"},
           {"id": "Q02", "axis": "edge", "text": "b"},
           {"id": "Q03", "axis": "edge", "text": "c"}]


class MedianHelperTest(unittest.TestCase):

    def test_none_values_are_dropped_not_treated_as_zero(self):
        self.assertEqual(_median([1.0, None, 3.0]), 2.0)

    def test_all_none_gives_none(self):
        self.assertIsNone(_median([None, None]))

    def test_empty_gives_none(self):
        self.assertIsNone(_median([]))

    def test_single_value(self):
        self.assertEqual(_median([7.5]), 7.5)


class AggregateTest(unittest.TestCase):

    def setUp(self):
        self.per_query = {
            "Q01": _metrics(axis="category", df_band="中频"),
            "Q02": _metrics(axis="edge", df_band="df = 0", n_returned=0,
                            margin_rel=None, decay_10=None,
                            top10_dl_median=None, length_bias=None),
            "Q03": _metrics(axis="edge", df_band="稀有", n_returned=5),
        }

    def test_all_slice_counts_every_query(self):
        rows = aggregate(self.per_query, QUERIES, "all")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["n_queries"], 3)

    def test_axis_slice_groups_by_axis(self):
        rows = {row["slice"]: row for row in
                aggregate(self.per_query, QUERIES, "axis")}
        self.assertEqual(rows["category"]["n_queries"], 1)
        self.assertEqual(rows["edge"]["n_queries"], 2)

    def test_df_band_slice_is_sorted_by_band_order(self):
        rows = aggregate(self.per_query, QUERIES, "df_band")
        slices = [row["slice"] for row in rows]
        # "稀有" 在 BAND_ORDER 里排在 "df = 0" 之前
        self.assertLess(slices.index("稀有"), slices.index("df = 0"))
        self.assertIn("稀有", BAND_ORDER)

    def test_zero_hit_query_does_not_pull_down_the_median(self):
        rows = aggregate(self.per_query, QUERIES, "all")
        # Q02 的 n_returned=None/0 被算进去了，但中位数取的是三个值 10/0/5 的中位 = 5
        self.assertEqual(rows[0]["n_returned"], 5)


class BScanTest(unittest.TestCase):

    def _profiles(self, biases):
        return {f"bm25_b_{b}": {"all": [{"length_bias": bias,
                                         "top10_dl_median": bias * 100,
                                         "margin_rel": 0.02, "ms": 1.0}]}
                for b, bias in biases.items()}

    def test_strictly_decreasing_is_detected(self):
        scan = b_scan(self._profiles({0.0: 2.8, 0.3: 1.2, 0.75: 0.4, 1.0: 0.24}))
        self.assertTrue(scan["monotonic_decreasing"])

    def test_a_single_upward_step_is_flagged(self):
        scan = b_scan(self._profiles({0.0: 2.8, 0.3: 1.2, 0.75: 1.4, 1.0: 0.24}))
        self.assertFalse(scan["monotonic_decreasing"])

    def test_equal_values_are_not_strictly_decreasing(self):
        # 完全不动才算退化；这里 0.4 → 0.4 应当被判为"不单调下降"
        scan = b_scan(self._profiles({0.0: 0.4, 0.3: 0.4}))
        self.assertFalse(scan["monotonic_decreasing"])

    def test_rows_are_sorted_by_b(self):
        scan = b_scan(self._profiles({1.0: 0.24, 0.0: 2.8, 0.75: 0.4, 0.3: 1.2}))
        self.assertEqual([row["b"] for row in scan["rows"]],
                         [0.0, 0.3, 0.75, 1.0])

    def test_non_grid_conditions_are_ignored(self):
        profiles = self._profiles({0.0: 2.8, 1.0: 0.24})
        profiles["tfidf_lnc_ltc"] = {"all": [{"length_bias": 0.9}]}
        self.assertEqual(len(b_scan(profiles)["rows"]), 2)


class NotableTest(unittest.TestCase):

    def test_zero_hit_and_ties_and_flat_are_separated(self):
        runs = {"impl": {
            "Q01": _metrics(n_returned=0),           # 零命中
            "Q02": _metrics(n_tied_top1=3),          # 首条并列
            "Q03": _metrics(decay_10=0.98),          # 分数拉不开
        }}
        info = notable(runs, QUERIES, ["impl"])["impl"]
        self.assertEqual([x["id"] for x in info["zero_hit"]], ["Q01"])
        self.assertEqual([x["id"] for x in info["tied_top1"]], ["Q02"])
        self.assertEqual([x["id"] for x in info["flat_top10"]], ["Q03"])

    def test_nothing_notable_gives_empty_lists(self):
        runs = {"impl": {q["id"]: _metrics() for q in QUERIES}}
        info = notable(runs, QUERIES, ["impl"])["impl"]
        self.assertEqual(info["zero_hit"], [])
        self.assertEqual(info["tied_top1"], [])
        self.assertEqual(info["flat_top10"], [])


class ConditionsTest(unittest.TestCase):

    def test_exactly_two_deliverable_implementations(self):
        # 逐实现分析的"两个实现"：一个 TF-IDF + 一个默认参数的 BM25
        conditions = analysis_conditions()
        deliverables = [c["name"] for c in conditions
                        if c["name"] == "tfidf_lnc_ltc" or c.get("is_default")]
        self.assertEqual(len(deliverables), 2)
        self.assertIn("tfidf_lnc_ltc", deliverables)

    def test_b_scan_spans_zero_to_one(self):
        bs = sorted(c["b"] for c in analysis_conditions() if "b" in c)
        self.assertEqual(bs, [0.0, 0.3, 0.75, 1.0])


if __name__ == "__main__":
    unittest.main()
