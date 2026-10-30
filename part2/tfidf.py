"""TF-IDF + 余弦检索（作业 3.3.2），两个变体：lnc.ltc 和 ltc.ltc。

按 SMART 三元组记号写成 ddd.qqq：左边三元组描述**文档向量**怎么加权，右边三元组
描述**查询向量**怎么加权（Manning 教材就是这个顺序）。每个位置一个字母：

    第 1 位（词频怎么算）  n = tf 原始计数   l = 1 + ln(tf)
    第 2 位（要不要 idf）  n = 不乘           t = ln(N / df)
    第 3 位（怎么归一化）  n = 不归一         c = 余弦归一（向量除以自己的长度 ‖·‖）

于是两个变体的区别只在文档侧要不要乘 idf：

    lnc.ltc   文档权重 = 1 + ln(tf)，查询权重 = (1 + ln tf) · ln(N / df)
              idf 全程只出现一次。这是课上讲的那一套，也是 Manning 教材推荐的默认
              组合。附带好处：文档向量与全集的 df 无关，任何文档的 ‖d‖ 都是定死的，
              不会因为新文档进来、某个词的 df 变了而要重算（增量索引友好）。

    ltc.ltc   文档权重 = (1 + ln tf) · ln(N / df)，查询权重同上
              这就是"每篇文档做成一个 TF-IDF 向量，再对查询向量求余弦"这句话的
              直译，也是大多数教科书正文里给的那版 TF-IDF 余弦。代价是同一个 idf
              在文档和查询里各乘一次，分子上等价于 idf²，稀有词被二次放大。

两者文档向量不同，所以余弦分母 ‖d‖ 也不同，两份都在建索引时算好，存在索引对象的
doc_norms_lnc / doc_norms_ltc 里。查询期不做全库扫描：对每个查询词走一次它的
posting 列表，把 (查询权重 / ‖q‖) · (文档权重 / ‖d‖) 累加到该文档的分数上（即
term-at-a-time），只有至少命中一个查询词的文档才会出现在候选里。

边界情况：idf = ln(N / df) 在 df = N 时为 0（"and""the" 这类每篇都有的词自动权重
0），df = 0（词不在索引里）时按 0 处理，所以查询里的生僻词不会给任何文档加分，
也不会遇到 ln(N / 0)。
"""

import math
from collections import Counter

from text_processing import get_tokenizer

from .inverted_index import (
    inverse_document_frequency,
    log_tf_weight,
    tfidf_weight,
)

LNC_LTC = "lnc.ltc"
LTC_LTC = "ltc.ltc"


def query_weights(index, query_terms):
    """查询侧 ltc 权重：每个查询词 w = (1 + ln tf_q) · ln(N / df)。

    两个变体的查询侧完全一样，区别只在文档侧。词不在索引里时 df = 0，idf 为 0，
    权重就是 0 —— 检索时把这些词扔掉即可，它们对任何文档的贡献都是 0。
    """
    n_docs = len(index.doc_lengths)
    return {
        term: tfidf_weight(tf, inverse_document_frequency(index.df(term), n_docs))
        for term, tf in Counter(query_terms).items()
    }


def tfidf_search(index, query_text, top_k=10, variant=LNC_LTC):
    """检索一条自然语言查询，返回 [(doc_id, score)]，分数是余弦值，降序排列。

    query_text 原句进来，内部用建索引时那个分词器切词，保证查询和文档走同一条
    预处理流水线。variant 选 "lnc.ltc"（默认）或 "ltc.ltc"。
    """
    if variant == LNC_LTC:
        doc_norms, doc_weight = index.doc_norms_lnc, log_tf_weight
    elif variant == LTC_LTC:
        doc_norms, doc_weight = index.doc_norms_ltc, tfidf_weight
    else:
        raise ValueError(f"不支持的权重组合：{variant}，可用 {LNC_LTC} 或 {LTC_LTC}")

    terms = get_tokenizer(index.tokenizer).tokenize(query_text)
    weights = {term: w for term, w in query_weights(index, terms).items() if w}
    if not weights:
        return []
    query_norm = math.sqrt(sum(weight * weight for weight in weights.values()))

    n_docs = len(index.doc_lengths)
    scores = {}
    for term, query_weight in weights.items():
        idf = inverse_document_frequency(index.df(term), n_docs)
        for doc_id, tf in index.postings.get(term, {}).items():
            norm = doc_norms[doc_id]
            if norm == 0.0:
                continue  # 空文档，余弦分母为 0，没有相似度可言
            scores[doc_id] = (
                scores.get(doc_id, 0.0) + query_weight * doc_weight(tf, idf) / norm
            )

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    # 除掉 ‖q‖ 才是真正的余弦；这一步对所有文档是同一个因子，不影响名次，影响读数值。
    return [(doc_id, score / query_norm) for doc_id, score in ranked[:top_k]]
