"""查询集的结构核对：条数、去重、分档边界、画像函数。

查询集是实验的输入，输入写错了后面全白跑，所以这些性质必须钉住：
50 条（IIR §8.1 的经验下限）、id 与文本都不重复、每个轴都有足够条数、
df 分档的边界值一个都不许串档。
"""

import unittest

from part2.inverted_index import build_inverted_index
from part2.queries import DF_BANDS, QUERIES, df_band, profile_query, query_axes

TOY = [
    {"review_id": "d1", "text": "the pizza and the pasta"},
    {"review_id": "d2", "text": "pizza pizza"},
    {"review_id": "d3", "text": "pasta and salad"},
    {"review_id": "d4", "text": ""},
    {"review_id": "d5", "text": "sushi is great"},
]
# N = 5；df：pizza / and / pasta = 2，the / salad / sushi / is / great = 1


class QuerySetStructureTests(unittest.TestCase):
    def test_exactly_fifty_queries(self):
        self.assertEqual(len(QUERIES), 50)

    def test_ids_are_unique_and_sequential(self):
        ids = [query["id"] for query in QUERIES]
        self.assertEqual(len(set(ids)), 50)
        self.assertEqual(ids, [f"Q{i:02d}" for i in range(1, 51)])

    def test_texts_are_unique_and_non_empty(self):
        texts = [query["text"] for query in QUERIES]
        self.assertEqual(len({text.lower() for text in texts}), 50)
        for text in texts:
            self.assertTrue(text.strip())

    def test_every_query_declares_axis_and_purpose(self):
        for query in QUERIES:
            self.assertIn(query["axis"],
                          {"category", "aspect", "attribute", "df_spectrum", "edge"})
            self.assertTrue(query["purpose"].strip())

    def test_axes_are_balanced_enough_to_support_attribution(self):
        counts = query_axes()
        self.assertEqual(counts, {"category": 12, "aspect": 12, "attribute": 10,
                                  "df_spectrum": 10, "edge": 6})
        # 每个轴都至少 5 条：单看任何一个轴都满足作业原文的最低要求
        self.assertTrue(all(count >= 5 for count in counts.values()))

    def test_far_more_than_the_five_required(self):
        self.assertGreaterEqual(len(QUERIES), 5)


class DfBandTests(unittest.TestCase):
    def test_boundary_values_land_in_the_right_band(self):
        self.assertEqual(df_band(0), "df = 0 (零命中)")
        self.assertEqual(df_band(1), "0 < df < 300 (稀有)")
        self.assertEqual(df_band(299), "0 < df < 300 (稀有)")
        self.assertEqual(df_band(300), "300 <= df < 1000 (低频)")
        self.assertEqual(df_band(999), "300 <= df < 1000 (低频)")
        self.assertEqual(df_band(1000), "1000 <= df < 3000 (中频)")
        self.assertEqual(df_band(2999), "1000 <= df < 3000 (中频)")
        self.assertEqual(df_band(3000), "3000 <= df < 10000 (高频)")
        self.assertEqual(df_band(9999), "3000 <= df < 10000 (高频)")
        self.assertEqual(df_band(10000), "df >= 10000 (超高频)")
        self.assertEqual(df_band(37401), "df >= 10000 (超高频)")

    def test_bands_cover_every_non_negative_df_exactly_once(self):
        seen = {}
        for df in list(range(0, 12000, 7)) + [20000, 37401]:
            seen.setdefault(df_band(df), 0)
            seen[df_band(df)] += 1
        self.assertEqual(set(seen), {name for name, _low, _high in DF_BANDS})


class ProfileQueryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = build_inverted_index(TOY, "custom")

    def test_profile_reports_tokenization_and_df(self):
        profile = profile_query(self.index, "pizza")
        self.assertEqual(profile["terms"], ["pizza"])
        self.assertEqual(profile["df_max"], 2)
        self.assertEqual(profile["df_band"], "0 < df < 300 (稀有)")
        self.assertEqual(profile["n_candidates"], 2)   # d1、d2 提到 pizza

    def test_repeated_term_is_counted_once_for_candidates(self):
        profile = profile_query(self.index, "pizza pizza")
        self.assertEqual(profile["n_terms"], 2)
        self.assertEqual(profile["n_unique_terms"], 1)
        self.assertEqual(profile["n_candidates"], 2)

    def test_unknown_term_gives_zero_hit_profile(self):
        profile = profile_query(self.index, "zzyzx")
        self.assertEqual(profile["df_max"], 0)
        self.assertEqual(profile["df_band"], "df = 0 (零命中)")
        self.assertEqual(profile["n_candidates"], 0)

    def test_contraction_is_expanded_the_same_way_as_documents(self):
        profile = profile_query(self.index, "don't")
        self.assertEqual(profile["terms"], ["do", "not"])

    def test_every_query_in_the_set_can_be_profiled(self):
        for query in QUERIES:
            profile = profile_query(self.index, query["text"])
            self.assertTrue(profile["terms"])
            self.assertIn(profile["df_band"],
                          {name for name, _low, _high in DF_BANDS})


if __name__ == "__main__":
    unittest.main()
