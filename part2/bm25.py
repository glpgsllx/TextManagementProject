"""Okapi BM25 检索（作业 3.3.2），三个变体：课件版、Okapi 版、Lucene 版。

三个变体共用同一个"词频饱和 + 长度归一"因子，只有 idf 的算法不同：

    score(Q, d) = Σ_{t∈Q}  idf(t)  ·  (k1 + 1) · tf  /  (tf + k1 · (1 − b + b · dl / avgdl))

逐符号：
    tf    词项在文档 d 里的出现次数
    dl    文档 d 分词后的长度，avgdl 是全语料平均长度（索引期一次算全）
    k1    词频饱和速度：tf 越大该项越接近上限 (k1 + 1)，出现 20 次不该有 20 倍证据
    b     长度归一强度：b = 0 完全不看长度，b = 1 完全按长度惩罚，默认 0.75
    idf   逆文档频率，三个变体的唯一区别，见下

idf 的三个版本：

    course  ln(N / df)
            课件版（Lecture 9.1 第 18–19 页），也是 TF-IDF 里那一个。df = N 时为 0，
            词不在索引里（df = 0）时按 0 处理、整项跳过。

    okapi   max(0, ln((N − df + 0.5) / (df + 0.5)))
            Robertson & Zaragoza 2009 的标准式（无相关反馈信息时 R = r = 0 的退化形式）。
            注意它**会取负值**：df > (N + 1) / 2 时分子小于分母。负值意味着"文档里出现了
            这个查询词反而扣分"，违背常识，所以必须钳到 0。本语料实测有 12 个词项中招，
            最高频的 and（df = 33,401 / N = 37,401）原始 idf = −2.12。

    okapi_raw（实验用，不进交付）  ln((N − df + 0.5) / (df + 0.5))，即 Okapi 原式去掉钳制。
            只用于做消融实验，量化"钳制到底改变了什么"。见下方"关于不钳制的 idf"。

    lucene  ln(1 + (N − df + 0.5) / (df + 0.5))
            Lucene 8.x / 9.x（也是 Elasticsearch 7、8）BM25Similarity 的实际实现，
            官方文档原文："Implemented as log(1 + (docCount - docFreq + 0.5)/(docFreq + 0.5))"。
            加 1 是 Lucene 故意的：既规避了负 idf，又不必像 Okapi 那样硬钳。它和课件版
            在 df ≪ N 时几乎重合（ln((N+1)/(df+0.5)) ≈ ln(N/df)），差别只在 df 接近 N 时
            才显现——课件版趋于 0，Lucene 版趋于 ln2 = 0.693。

关于不钳制的 idf（只给实验用）
--------------------------------
`okapi_idf_unclamped` 是 Okapi 原式去掉 max(0, ·)，会返回负值。它**故意不登记进**
`IDF_BY_VARIANT`，所以 `bm25_search(index, q, variant=...)` 无论传什么都选不到它，
循环遍历变体的脚本也不会误撞上。实验要用它，必须显式写：

    bm25_search(index, q, idf_fn=okapi_idf_unclamped)

`idf_fn` 一旦传入就完全接管 idf，`variant` 不再参与。交付的默认路径（不传 idf_fn）
只会走 `IDF_BY_VARIANT` 里的三个变体。

尺度提醒：Lucene 当前主干（10.x）把对所有词项都相同的常数 (k1 + 1) 去掉了。
那个常数不影响排名，只改分数尺度，所以这里保留课件 / Okapi 教科书共有的 (k1 + 1)，
让三个变体的差别干净地收敛在 idf 一处。

查询只用去重的词项（和 Lucene 对 TermQuery 的处理一致）：课件里给长查询准备的
(k3 + 1)·tf_q / (k3 + tf_q) 查询词频饱和项没有实现，因为本作业的查询 2–4 个词、
每个词 tf_q = 1，该项恒等于 1。真要做长查询（比如把整条评论当查询）再加。

打分走倒排表 term-at-a-time：对每个查询词扫一遍 posting 列表，只有至少命中一个查询词
的文档才会出现在候选里，不用遍历全库。idf 算出 0 的词项直接跳过——它对所有文档的贡献
都是 0，跳过它还能少扫一条 postings。
"""

import math

from text_processing import get_tokenizer

COURSE = "course"
OKAPI = "okapi"
LUCENE = "lucene"

DEFAULT_K1 = 1.2
DEFAULT_B = 0.75


def course_idf(df, n_docs):
    """课件版 idf = ln(N / df)。df = 0（词不在索引里）返回 0。"""
    return math.log(n_docs / df) if df else 0.0


def okapi_idf(df, n_docs):
    """Okapi 版 idf = max(0, ln((N − df + 0.5) / (df + 0.5)))，负的钳到 0。"""
    return max(0.0, math.log((n_docs - df + 0.5) / (df + 0.5)))


def lucene_idf(df, n_docs):
    """Lucene 版 idf = ln(1 + (N − df + 0.5) / (df + 0.5))，天然非负。"""
    return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))


def okapi_idf_unclamped(df, n_docs):
    """实验用：Okapi 原式去掉钳制，df > (N + 1) / 2 时返回负值。

    只服务于消融实验——拿到负 idf，才能量化"钳制这一步到底改变了结果多少"。
    它**没有**登记进 IDF_BY_VARIANT，所以 bm25_search 的 variant 参数选不到它；
    要用必须通过 idf_fn= 显式传入。别在交付路径里用它：负 idf 让"文档里出现了
    查询词"反过来扣分，是教科书里明令要避免的病态行为。
    """
    return math.log((n_docs - df + 0.5) / (df + 0.5))


# 交付路径认可的变体。okapi_idf_unclamped 刻意不在这里，见模块顶部说明。
IDF_BY_VARIANT = {COURSE: course_idf, OKAPI: okapi_idf, LUCENE: lucene_idf}


def term_weight(tf, dl, avgdl, k1=DEFAULT_K1, b=DEFAULT_B):
    """词频饱和 + 长度归一因子：(k1 + 1) · tf / (tf + k1 · (1 − b + b · dl / avgdl))。

    tf → ∞ 时趋于 (k1 + 1)：重复出现的边际证据递减，堆砌关键词堆不出分。
    dl = avgdl 时括号里等于 1，长度归一不起作用；dl 越大分母越大，分数越低。
    """
    return tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))


def bm25_search(index, query_text, k1=DEFAULT_K1, b=DEFAULT_B,
                variant=OKAPI, top_k=10, idf_fn=None):
    """检索一条自然语言查询，返回 [(doc_id, score)]，分数降序、同分按 doc_id 升序。

    query_text 原句进来，内部用建索引时那个分词器切词。variant 选 course / okapi
    （默认）/ lucene，k1、b 是可调参数（作业原文要求 "correctly handle the
    parameters k1 and b"）。

    idf_fn 是实验用的后门：传了它就用它算 idf，variant 不再参与。只有这里能拿到
    未登记进 IDF_BY_VARIANT 的变体（比如不钳制的 okapi_idf_unclamped）。默认
    None，走变体表；交付路径不要传。
    """
    if idf_fn is not None:
        idf = idf_fn
    else:
        if variant not in IDF_BY_VARIANT:
            raise ValueError(f"不支持的 idf 变体：{variant}，可用 {sorted(IDF_BY_VARIANT)}")
        idf = IDF_BY_VARIANT[variant]

    terms = set(get_tokenizer(index.tokenizer).tokenize(query_text))
    n_docs = len(index.doc_lengths)
    scores = {}
    for term in terms:
        plist = index.postings.get(term)
        if not plist:
            continue  # 词不在索引里：df = 0，对所有文档的贡献都是 0
        weight = idf(len(plist), n_docs)
        if weight == 0.0:
            continue  # df = N（或 Okapi 被钳到 0）的词项区分不了任何文档
        for doc_id, tf in plist.items():
            scores[doc_id] = (scores.get(doc_id, 0.0)
                              + weight * term_weight(tf, index.doc_lengths[doc_id],
                                                     index.avg_doc_length, k1, b))

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return ranked[:top_k]
