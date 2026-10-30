"""TF-IDF 余弦的手算核对（lnc.ltc 与 ltc.ltc）。

语料是 test_inverted_index.py 里那 4 篇短文，N = 4：

    d1 "the pizza and the pasta"   the=2 pizza=1 and=1 pasta=1
    d2 "pizza pizza"               pizza=2
    d3 "pasta and salad"           pasta=1 and=1 salad=1
    d4 ""                          空评论

    df：the=1 salad=1 | pizza=2 and=2 pasta=2
    idf：df=1 → ln 4 = 1.3863；df=2 → ln 2 = 0.6931

下面每个测试都先把算式写出来，再和代码输出逐位比对，这是报告里 defend 用的证据。
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
        # 查询侧两个变体一样：w = (1 + ln tf_q) · ln(N/df)。
        # "the pizza"：两词都是查询里出现 1 次，tf 项就是 1，权重等于各自的 idf。
        self.assertAlmostEqual(query_weights(self.index, "the pizza".split()),
                               {"the": LOG4, "pizza": LOG2})
        # 查询里重复同一个词，tf = 2 → 权重乘 (1 + ln 2)，不是乘 2。
        self.assertAlmostEqual(query_weights(self.index, "pizza pizza".split()),
                               {"pizza": (1 + math.log(2)) * LOG2})
        # 词不在索引里：df = 0 → idf 记 0，这个词不参与打分。
        self.assertEqual(query_weights(self.index, ["sushi"]), {"sushi": 0.0})

    # ---------- lnc.ltc ----------

    def test_lnc_ltc_single_term_query(self):
        # 单查询词时 idf 在分子分母里约掉，余弦只由文档侧的 tf 决定：
        # d2 只含 pizza，w = 1 + ln 2，‖d2‖_lnc = 1 + ln 2 → 余弦 = 1。
        # d1 里 pizza 出现 1 次 → w = 1，‖d1‖_lnc = √((1+ln2)² + 3) → 余弦 = 1/‖d1‖。
        results = tfidf_search(self.index, "pizza", variant=LNC_LTC)
        self.assertEqual([doc_id for doc_id, _ in results], ["d2", "d1"])
        self.assertAlmostEqual(results[0][1], 1.0)
        self.assertAlmostEqual(results[1][1], 1 / math.sqrt((1 + LOG2) ** 2 + 3))
        self.assertAlmostEqual(results[1][1], 0.4128585720620119)

    def test_lnc_ltc_two_term_query(self):
        # q = (ln2, ln2)，‖q‖ = √2·ln2 → 归一化后每个查询词的权重都是 1/√2。
        # d2 只含 pizza：1/√2 × 1 = 0.7071
        # d1 含 pizza 与 pasta，各 1 次：1/√2 × (1+1)/‖d1‖_lnc = 1/√2 × 2/2.4221
        # d3 只含 pasta：1/√2 × 1/√3 = 0.4082
        results = tfidf_search(self.index, "pizza pasta", variant=LNC_LTC)
        self.assertEqual([doc_id for doc_id, _ in results], ["d2", "d1", "d3"])
        self.assertAlmostEqual(results[0][1], 1 / math.sqrt(2))
        self.assertAlmostEqual(results[2][1], 1 / math.sqrt(6))
        self.assertAlmostEqual(results[1][1], 0.583870191952087)

    def test_lnc_ltc_terms_with_low_idf_carry_less_weight(self):
        # "pizza and the"：pizza / and 的 idf = ln2，the 的 idf = ln4（the 只出现在 d1）。
        # d1 同时含三个词，其中低频词 the 权重最大，d1 排第一。
        results = tfidf_search(self.index, "pizza and the", variant=LNC_LTC)
        self.assertEqual([doc_id for doc_id, _ in results], ["d1", "d2", "d3"])
        self.assertAlmostEqual(results[0][1], 0.9078534846653029)

    # ---------- ltc.ltc ----------

    def test_ltc_ltc_single_term_query(self):
        # 文档侧权重换成 (1+ln tf)·idf，‖d2‖_ltc = (1+ln2)·ln2，分子也是它 → 余弦还是 1。
        # d1 的 ‖d1‖_ltc = √([(1+ln2)·ln4]² + 3·ln2²)，比 lnc 那份大得多，
        # 同一个词在 d1 上的余弦就比 lnc 低：0.4129 → 0.2629。
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
        # 这条查询把两个变体的差别摆在最明显的地方：
        #   lnc.ltc  d1 0.7151 > d3 0.6667 > d2 0.5774 —— 命中三个查询词的 d1 排第一
        #   ltc.ltc  d2 0.5774 > d3 0.4714 > d1 0.4554 —— 只含一个词的 d2 反而登顶
        # 机制：d1 里有个不在查询中的稀有词 the（idf = ln4）。lnc 下它对 ‖d1‖ 的
        # 贡献是 1+ln2 = 1.69，ltc 下变成 1.69·ln4 = 2.35 —— 只拉伸分母、不贡献分子，
        # 于是 d1 的余弦塌下去；d2 只有 pizza 一个词，向量方向不受 idf 影响，坐收渔利。
        lnc = tfidf_search(self.index, "pizza and pasta", variant=LNC_LTC)
        ltc = tfidf_search(self.index, "pizza and pasta", variant=LTC_LTC)
        self.assertEqual([doc_id for doc_id, _ in lnc], ["d1", "d3", "d2"])
        self.assertEqual([doc_id for doc_id, _ in ltc], ["d2", "d3", "d1"])
        self.assertAlmostEqual(lnc[0][1], 0.7150920231517413)
        self.assertAlmostEqual(ltc[0][1], 0.5773502691896257)

    def test_variants_can_also_agree(self):
        # 换个查询两者名次又完全一致 —— 说明差别不是"哪套更对"，而是 idf 怎么分配到
        # 命中词和未命中词头上。d2 只有一个词，归一化后方向恒等于该词的基向量，
        # 两套权重的余弦都是 1/√2。
        lnc = tfidf_search(self.index, "pizza pasta", variant=LNC_LTC)
        ltc = tfidf_search(self.index, "pizza pasta", variant=LTC_LTC)
        self.assertEqual([doc_id for doc_id, _ in lnc], ["d2", "d1", "d3"])
        self.assertEqual([doc_id for doc_id, _ in ltc], ["d2", "d1", "d3"])
        self.assertAlmostEqual(lnc[0][1], ltc[0][1])

    # ---------- 边界 ----------

    def test_repeating_a_term_in_the_query_changes_nothing(self):
        # 重复词只放大查询向量本身，‖q‖ 同步放大，归一化后抵消 —— 见 query_weights 那条。
        self.assertEqual(tfidf_search(self.index, "pizza pizza", variant=LNC_LTC),
                         tfidf_search(self.index, "pizza", variant=LNC_LTC))

    def test_unknown_query_terms_return_nothing(self):
        self.assertEqual(tfidf_search(self.index, "sushi"), [])
        self.assertEqual(tfidf_search(self.index, ""), [])

    def test_empty_document_never_shows_up(self):
        # d4 的 ‖d‖ = 0，余弦分母为 0，必须跳过而不是除零。
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
