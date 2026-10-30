"""BM25 三个变体的手算核对（课件版 / Okapi 版 / Lucene 版）。

语料 A：5 篇，用来逐个比对 idf 与分数。

    d1 "the pizza and the pasta"   the=2 pizza=1 and=1 pasta=1   dl=5
    d2 "pizza pizza"               pizza=2                       dl=2
    d3 "pasta and salad"           pasta=1 and=1 salad=1         dl=3
    d4 ""                          空评论                        dl=0
    d5 "sushi is great"            sushi=1 is=1 great=1          dl=3

    N = 5，avgdl = (5+2+3+0+3)/5 = 2.6
    df：pizza / and / pasta = 2；the / salad / sushi / is / great = 1

语料 B：4 篇，专门演示 Okapi 的钳制（the 出现在 3/4 篇，df > N/2）。

    d1 "the pizza"  d2 "the pasta"  d3 "the salad"  d4 "sushi"
    N = 4，avgdl = 7/4 = 1.75，df(the) = 3

默认参数 k1 = 1.2、b = 0.75。下面每个测试都先把算式写出来，再和代码输出逐位比对。
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
        # N = 5。df = 2 的词（pizza / and / pasta）：
        #   course ln(5/2) = 0.9163；okapi ln(3.5/2.5) = 0.3365；lucene ln(1 + 3.5/2.5) = 0.8755
        self.assertAlmostEqual(course_idf(2, 5), 0.9162907318741551)
        self.assertAlmostEqual(okapi_idf(2, 5), 0.3364722366212129)
        self.assertAlmostEqual(lucene_idf(2, 5), 0.8754687373538999)
        # df = 1 的词（the / salad / sushi / is / great）：
        #   course ln(5/1) = 1.6094；okapi ln(4.5/1.5) = 1.0986；lucene ln(1 + 4.5/1.5) = 1.3863
        self.assertAlmostEqual(course_idf(1, 5), 1.6094379124341003)
        self.assertAlmostEqual(okapi_idf(1, 5), 1.0986122886681098)
        self.assertAlmostEqual(lucene_idf(1, 5), 1.3862943611198906)

    def test_terms_missing_from_the_index_get_zero_idf(self):
        self.assertEqual(course_idf(0, 5), 0.0)
        # Okapi 的 df = 0 分支走不到（查询词不在索引里时整个词项被跳过），
        # 这里只确认不会因为 df = 0 抛异常。
        self.assertGreater(okapi_idf(0, 5), 0.0)

    def test_okapi_idf_goes_negative_before_clamping(self):
        # 语料 B：df(the) = 3，N = 4 > 2 × 3 … 即 df > N/2，原始 idf 为负：
        # ln((4-3+0.5)/(3+0.5)) = ln(1.5/3.5) = -0.8473，钳制后为 0。
        raw = math.log((4 - 3 + 0.5) / (3 + 0.5))
        self.assertAlmostEqual(raw, -0.8472978603872037)
        self.assertLess(raw, 0.0)
        self.assertEqual(okapi_idf(3, 4), 0.0)
        # 同一对 (df, N) 下另外两个变体仍然为正，这就是三者的实质分歧。
        self.assertAlmostEqual(course_idf(3, 4), math.log(4 / 3))
        self.assertAlmostEqual(lucene_idf(3, 4), math.log(1 + 1.5 / 3.5))

    def test_course_and_lucene_idf_are_nearly_identical_for_rare_terms(self):
        # Lucene 版 = ln((N+1)/(df+0.5))，df ≪ N 时和课件版 ln(N/df) 几乎重合；
        # 这里 N = 5、df = 1 已经算是"高频"，差距仍有 0.22，df 更小才会真正贴近。
        self.assertAlmostEqual(course_idf(1, 5) - lucene_idf(1, 5), 0.22314355131420976)


class TermWeightTests(unittest.TestCase):
    def test_saturation_factor_matches_hand_computation(self):
        # d2（dl = 2，tf = 2）：2.2 × 2 / (2 + 1.2 × (0.25 + 0.75 × 2/2.6)) = 4.4/2.9923
        self.assertAlmostEqual(term_weight(2, 2, AVGDL_A), 1.4704370179948587)

    def test_average_length_documents_are_neutral(self):
        # dl = avgdl 时括号里等于 1，长度归一不起作用：weight = 2.2 × tf / (tf + 1.2)
        self.assertAlmostEqual(term_weight(1, AVGDL_A, AVGDL_A), 2.2 / 2.2)
        self.assertAlmostEqual(term_weight(5, AVGDL_A, AVGDL_A), 2.2 * 5 / 6.2)

    def test_k1_zero_collapses_to_binary(self):
        # k1 = 0：weight = (0+1)·tf / (tf + 0) = 1，不管出现几次，退化成"有没有出现"。
        self.assertEqual(term_weight(1, 5, AVGDL_A, k1=0), 1.0)
        self.assertEqual(term_weight(20, 5, AVGDL_A, k1=0), 1.0)


class ToyRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.index = build_inverted_index(TOY_A)

    def test_course_variant_scores(self):
        # "pizza pasta"：两个词 df 都是 2，共用 idf = ln 2.5。
        # d2 = idf × 2.2×2/(2 + 1.2×(0.25+0.75×2/2.6)) = idf × 1.4704
        # d1 = idf × 2 × 2.2/(1 + 1.2×(0.25+0.75×5/2.6)) = idf × 1.4516
        # d3 = idf × 2.2/(1 + 1.2×(0.25+0.75×3/2.6)) = idf × 0.9408
        results = bm25_search(self.index, "pizza pasta", variant=COURSE)
        self.assertEqual([doc for doc, _ in results], ["d2", "d1", "d3"])
        self.assertAlmostEqual(results[0][1], 1.3473478113933592)
        self.assertAlmostEqual(results[1][1], 1.3302494889137482)
        self.assertAlmostEqual(results[2][1], 0.8620366753816067)

    def test_okapi_variant_scores_are_on_a_different_scale(self):
        # 同一批文档，idf 从 0.9163 换成 0.3365，分数等比缩小，名次不变：
        # 因为这条查询里两个词的 df 相同，idf 是个公因子。
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
        # 加上只出现在 d3 的 salad（df = 1），d3 反超。三个变体名次一致。
        for variant in (COURSE, OKAPI, LUCENE):
            results = bm25_search(self.index, "pizza pasta salad", variant=variant)
            self.assertEqual([doc for doc, _ in results], ["d3", "d2", "d1"], variant)
        self.assertAlmostEqual(
            bm25_search(self.index, "pizza pasta salad", variant=COURSE)[0][1],
            2.376178921947899)

    def test_repeating_a_query_term_changes_nothing(self):
        # 查询只用去重后的词项，和 Lucene 对 TermQuery 的处理一致。
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
    """k1 / b 的退化配置——作业要求"正确处理这两个参数"，这里给出可核验的行为。"""

    def setUp(self):
        self.index = build_inverted_index(TOY_A)

    def test_k1_zero_sum_of_idf_only(self):
        # k1 = 0 → 每个命中词的贡献就是它的 idf，与 tf 无关。
        # d1 命中两个词 = 2 × 0.3365；d2、d3 各命中一个词 = 0.3365。
        results = bm25_search(self.index, "pizza pasta", k1=0, variant=OKAPI)
        self.assertEqual([doc for doc, _ in results], ["d1", "d2", "d3"])
        self.assertAlmostEqual(results[0][1], 2 * 0.3364722366212129)
        self.assertAlmostEqual(results[1][1], 0.3364722366212129)

    def test_b_zero_makes_the_longer_document_win(self):
        # b = 0 完全不做长度归一：匹配了两个词的 d1（dl = 5）压过去；
        # 换回默认 b = 0.75，只匹配一个词但更短的 d2（dl = 2）反超。
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
    """df > N/2 时 Okapi 必须钳制，这是三个变体唯一的"定性"差异。"""

    def setUp(self):
        self.index = build_inverted_index(TOY_B)

    def test_clamping_changes_the_result_set_not_just_the_scores(self):
        # 查询 "the pizza"：
        #   course / lucene 返回 3 篇（d2、d3 靠 the 拿了点分）
        #   okapi    只返回 1 篇 —— the 的 idf 被钳成 0，整个词项被跳过，
        #            只含 the 的 d2 / d3 一分得不到，直接从结果里消失。
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
        # 不钳制的话 d1 的分数是 0.8005 + (-0.8473) × 0.9448 = 0.0001，
        # 几乎归零 —— 文档里出现了查询词反而几乎得不到分，这就是为什么必须钳。
        idf_the = math.log((4 - 3 + 0.5) / (3 + 0.5))
        weight = term_weight(1, 2, 1.75)
        self.assertAlmostEqual(0.8005145429425117 + idf_the * weight, 0.0, places=6)


class UnclampedIdfTests(unittest.TestCase):
    """不钳制的 Okapi idf：只做消融实验用，结构上必须够不到默认路径。

    这组测试一半是在验算术，一半是在守住"它不能被默认触发"这件事本身。
    """

    def setUp(self):
        self.index = build_inverted_index(TOY_B)

    def test_unclamped_is_the_raw_logarithm(self):
        # 语料 B：df(the) = 3，N = 4 → ln(1.5/3.5) = -0.8473，钳制版是 0。
        self.assertAlmostEqual(okapi_idf_unclamped(3, 4), -0.8472978603872037)
        self.assertAlmostEqual(okapi_idf(3, 4), 0.0)
        # 同一对 (df, N) 下两个函数相差的就是被钳掉的那部分。
        self.assertAlmostEqual(okapi_idf(3, 4) - okapi_idf_unclamped(3, 4),
                               0.8472978603872037)

    def test_clamping_is_a_no_op_when_df_is_small(self):
        # df ≤ (N + 1) / 2 时原始值为非负，钳不钳逐位相同——说明钳制只影响高频词，
        # 不会悄悄改动绝大多数词项的权重。
        for df in (1, 2):
            self.assertEqual(okapi_idf_unclamped(df, 5), okapi_idf(df, 5))
        # 边界：df = (N + 1) / 2 时原始值恰好为 0。
        self.assertEqual(okapi_idf_unclamped(2, 4), 0.0)
        self.assertEqual(okapi_idf_unclamped(3, 5), -0.3364722366212129)

    def test_unclamped_is_not_registered_as_a_variant(self):
        # 这句是"不会被默认触发"的核心保证：它不在变体表里的任何一个键或值上。
        self.assertNotIn(okapi_idf_unclamped, IDF_BY_VARIANT.values())
        self.assertEqual(sorted(IDF_BY_VARIANT), [COURSE, LUCENE, OKAPI])

    def test_no_variant_name_can_reach_the_unclamped_idf(self):
        # 猜名字拿不到，把函数本身当名字传也拿不到——变体这一路对它是彻底关闭的。
        for bad in ("okapi_raw", "okapi_unclamped", "raw"):
            with self.assertRaises(ValueError):
                bm25_search(self.index, "the pizza", variant=bad)
        with self.assertRaises(ValueError):
            bm25_search(self.index, "the pizza", variant=okapi_idf_unclamped)

    def test_only_the_explicit_idf_fn_channel_reaches_it(self):
        # 想用就必须把函数名写出来，调用处一眼能看出这是实验口径。
        results = bm25_search(self.index, "the pizza", idf_fn=okapi_idf_unclamped)
        # 语料 B 里 df(the) = 3、df(pizza) = 1，两个 idf 恰好互为相反数，
        # 命中两个词的 d1 精确抵消成 0；只命中 the 的 d2/d3 拿到负分。
        self.assertEqual([doc for doc, _ in results], ["d1", "d2", "d3"])
        self.assertAlmostEqual(results[0][1], 0.0)
        self.assertAlmostEqual(results[1][1], -0.8005145429425117)
        self.assertAlmostEqual(results[2][1], -0.8005145429425117)

    def test_idf_fn_takes_precedence_over_variant(self):
        # 显式传入就完全接管，variant 说什么都不算——避免"以为用的是 course，
        # 其实被别的参数悄悄换掉"这类静默错误。
        explicit = bm25_search(self.index, "the pizza", variant=COURSE,
                               idf_fn=okapi_idf_unclamped)
        self.assertAlmostEqual(explicit[0][1], 0.0)

    def test_the_clamped_and_unclamped_paths_diverge_on_result_sets(self):
        # 消融实验真正要看的东西：钳制不只是改分数，它改变了谁出现在结果里。
        #   钳制   d1（0.8005）—— the 的 idf 为 0，整项跳过，d2/d3 落榜
        #   不钳制 d1（0.0）、d2、d3（各 -0.8005）—— 高频词变成负证据
        clamped = bm25_search(self.index, "the pizza", variant=OKAPI)
        unclamped = bm25_search(self.index, "the pizza", idf_fn=okapi_idf_unclamped)
        self.assertEqual([doc for doc, _ in clamped], ["d1"])
        self.assertEqual([doc for doc, _ in unclamped], ["d1", "d2", "d3"])
        # 首条分数从 0.8005 掉到 0：文档里出现了查询词却拿不到分，这就是病态所在。
        self.assertAlmostEqual(clamped[0][1], 0.8005145429425117)
        self.assertAlmostEqual(unclamped[0][1], 0.0)

    def test_the_default_path_is_untouched_by_any_of_this(self):
        # 交付路径（不传 idf_fn）三个变体的结果，和不加这个接口之前逐位一致。
        expected = {
            COURSE: (["d1", "d2", "d3"], 1.5815482869327453),
            LUCENE: (["d1", "d2", "d3"], 1.474477013697908),
            OKAPI: (["d1"], 0.8005145429425117),
        }
        for variant, (docs, top_score) in expected.items():
            results = bm25_search(self.index, "the pizza", variant=variant)
            self.assertEqual([doc for doc, _ in results], docs, variant)
            self.assertAlmostEqual(results[0][1], top_score, msg=variant)
        # 不传任何 idf 相关参数时走的是 okapi（钳制版），不可能是不钳制的那份。
        self.assertEqual(bm25_search(self.index, "the pizza"),
                         bm25_search(self.index, "the pizza", variant=OKAPI))


if __name__ == "__main__":
    unittest.main()
