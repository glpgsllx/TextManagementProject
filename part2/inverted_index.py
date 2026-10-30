"""倒排索引（作业 3.3.1）。

一篇文档 = 一条 Yelp 评论，doc_id 用 review_id。词项来自公共分词器
`text_processing` 的 custom 方法（停用词保留、不做词干提取），与 part1 口径一致。

索引就是三个字典加一个数，没有第二层封装：

    index.postings[term][doc_id] = tf     某个词项在某篇评论里出现了几次
    index.doc_lengths[doc_id] = dl        某篇评论分词后有多少个词项
    index.doc_norms_lnc[doc_id] = ‖d‖     文档侧 lnc 权重的向量长度（余弦的分母）
    index.doc_norms_ltc[doc_id] = ‖d‖     文档侧 ltc 权重的向量长度
    index.avg_doc_length = avgdl          全语料平均文档长度，索引期一次算全

存两份 ‖d‖ 是因为 TF-IDF 有两种经典权重组合，它们的文档向量不同，长度也就不同：

    lnc.ltc  文档侧权重 = 1 + ln tf（不含 idf）
    ltc.ltc  文档侧权重 = (1 + ln tf) · ln(N / df)

余弦的分母必须和分子用同一套权重，拿错一份分就错了。详见 part2/tfidf.py。

df（含某词项的文档数）不单独存：它是 postings 的长度，`index.df(term)` 直接算。
不存词项位置：本作业没有短语查询，用不上。

构建分两遍。第一遍逐篇分词，累加 postings 与 dl；第二遍算 ‖d‖——含 idf 的那一份
依赖全量 df，只能等第一遍结束后算。有了 ‖d‖，查询期算余弦只需遍历查询词，不用再扫全表。

全量构建约 8 秒，所以建好一次就 pickle 落盘（`save_index`），之后用 `load_index`
读回来，几秒钟的事。落盘时一起记下语料的文件名、字节数、条数和分词器，读回时
对不上就会报错提醒重建，避免拿着旧索引跑新语料。
"""

import math
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from text_processing import get_tokenizer

DEFAULT_TOKENIZER = "custom"


def inverse_document_frequency(df, n_docs):
    """idf = ln(N / df)。N 是文档总数，df 是含该词项的文档数。

    df = 0（查询词不在索引里）时返回 0，这个词就不会给任何文档加分。
    df = N（每篇文档都有）时 idf = 0，这个词区分不了任何文档，权重自然为 0。
    """
    return math.log(n_docs / df) if df else 0.0


def log_tf_weight(tf, idf=None):
    """文档侧 lnc 权重 = 1 + ln tf，不乘 idf。

    idf 参数不参与计算，留着只为了和 tfidf_weight 同签名，打分循环里两者可以
    互换地写成 weight(tf, idf)。idf 交给查询侧乘一次就够了，两个位置都乘等于
    用成了 idf²，会把稀有词的权重放得比设计的更重。
    """
    return 1.0 + math.log(tf)


def tfidf_weight(tf, idf):
    """TF-IDF 词项权重 w = (1 + ln tf) · idf。

    用 1 + ln tf 而不是原始 tf：出现 20 次的词不该比出现 1 次的词重 20 倍，
    子线性缩放可以压住长文档里重复词的主导作用。索引算 ‖d‖ 和查询期算分数
    共用这一个定义，改动只需改这里。
    """
    return (1.0 + math.log(tf)) * idf


@dataclass
class InvertedIndex:
    """一个倒排索引。字段就是索引的全部内容，没有隐藏状态。"""

    postings: dict
    doc_lengths: dict
    doc_norms_lnc: dict
    doc_norms_ltc: dict
    avg_doc_length: float
    total_tokens: int
    tokenizer: str

    def df(self, term):
        """含 term 的文档数；term 不在索引里时为 0。"""
        return len(self.postings.get(term, ()))


def compute_doc_norms(postings, doc_ids, weight, n_docs):
    """第二遍：逐条 posting 累加 ‖d‖² = Σ_t w(t, d)²，最后开方。

    weight 决定算的是哪一套权重的长度：lnc 传 log_tf_weight，ltc 传
    tfidf_weight。doc_ids 用来先把每篇文档的累加器置 0，这样空文档（dl = 0）
    也会出现在结果里，取到 ‖d‖ = 0，打分函数看到 0 就能跳过它，不会除零。
    """
    squared = {doc_id: 0.0 for doc_id in doc_ids}
    for plist in postings.values():
        idf = inverse_document_frequency(len(plist), n_docs)
        for doc_id, tf in plist.items():
            value = weight(tf, idf)
            squared[doc_id] += value * value
    return {doc_id: math.sqrt(value) for doc_id, value in squared.items()}


def build_inverted_index(reviews, tokenizer_name=DEFAULT_TOKENIZER):
    """给一批评论建倒排索引。

    reviews 是可迭代对象，每项至少带 review_id 与 text 两个字段（part1 的
    documents 就是这个形状）。返回值可直接交给打分函数：BM25 用 postings /
    doc_lengths / avg_doc_length，TF-IDF 余弦再加两份 doc_norms。
    """
    tokenizer = get_tokenizer(tokenizer_name)
    postings = {}
    doc_lengths = {}
    total_tokens = 0
    for review in reviews:
        doc_id = review["review_id"]
        terms = tokenizer.tokenize(review["text"])
        doc_lengths[doc_id] = len(terms)
        total_tokens += len(terms)
        for term, tf in Counter(terms).items():
            postings.setdefault(term, {})[doc_id] = tf
    if not doc_lengths:
        raise ValueError("没有读到任何文档，无法建索引")
    n_docs = len(doc_lengths)
    return InvertedIndex(
        postings=postings,
        doc_lengths=doc_lengths,
        doc_norms_lnc=compute_doc_norms(postings, doc_lengths, log_tf_weight, n_docs),
        doc_norms_ltc=compute_doc_norms(postings, doc_lengths, tfidf_weight, n_docs),
        avg_doc_length=total_tokens / n_docs,
        total_tokens=total_tokens,
        tokenizer=tokenizer_name,
    )


def save_index(index, path, source=None):
    """把索引 pickle 到磁盘，返回文件字节数。

    source 是索引的"出身"：语料文件名、字节数、条数、分词器。存下来是为了
    读回时能核对，见 load_index。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": dict(source or {}),
        "postings": index.postings,
        "doc_lengths": index.doc_lengths,
        "doc_norms_lnc": index.doc_norms_lnc,
        "doc_norms_ltc": index.doc_norms_ltc,
        "avg_doc_length": index.avg_doc_length,
        "total_tokens": index.total_tokens,
        "tokenizer": index.tokenizer,
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=4)
    return path.stat().st_size


def load_index(path, source=None):
    """从磁盘读回索引。

    传了 source 就先逐项核对出身；有一项对不上说明语料或分词器变了，直接报错，
    让调用方去重建，不要拿旧索引继续跑。
    """
    path = Path(path)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    expected = dict(source or {})
    saved = payload.get("source", {})
    mismatch = {key: (saved.get(key), value)
                for key, value in expected.items() if saved.get(key) != value}
    if mismatch:
        raise ValueError(f"索引与当前语料不一致，请用 --rebuild 重建：{mismatch}")
    return InvertedIndex(
        postings=payload["postings"],
        doc_lengths=payload["doc_lengths"],
        doc_norms_lnc=payload["doc_norms_lnc"],
        doc_norms_ltc=payload["doc_norms_ltc"],
        avg_doc_length=payload["avg_doc_length"],
        total_tokens=payload["total_tokens"],
        tokenizer=payload["tokenizer"],
    )
