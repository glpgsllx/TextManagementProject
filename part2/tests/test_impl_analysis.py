"""Tests for the per-implementation analysis.

What is pinned down here is whether the metrics are computed correctly and whether the
monotonicity verdict on the b sweep can be trusted. It touches neither the index nor
Java — those are covered by the query_experiment / lucene_experiment tests.
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
    base = {"text": "x", "axis": "category", "df_band": "medium",
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
            "Q01": _metrics(axis="category", df_band="medium"),
            "Q02": _metrics(axis="edge", df_band="df = 0", n_returned=0,
                            margin_rel=None, decay_10=None,
                            top10_dl_median=None, length_bias=None),
            "Q03": _metrics(axis="edge", df_band="rare", n_returned=5),
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
        # "rare" precedes "df = 0" in BAND_ORDER
        self.assertLess(slices.index("rare"), slices.index("df = 0"))
        self.assertIn("rare", BAND_ORDER)

    def test_zero_hit_query_does_not_pull_down_the_median(self):
        rows = aggregate(self.per_query, QUERIES, "all")
        # Q02's n_returned of 0 is included, but the median of 10/0/5 is still 5.
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
        # Only a completely flat series counts as degenerate; here 0.4 → 0.4 must be
        # judged "not strictly decreasing".
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
            "Q01": _metrics(n_returned=0),           # zero hits
            "Q02": _metrics(n_tied_top1=3),          # tie for first place
            "Q03": _metrics(decay_10=0.98),          # scores barely separate
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
        # The "two implementations" of the per-implementation analysis: one TF-IDF plus
        # one BM25 with default parameters.
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
