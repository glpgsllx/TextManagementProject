# 报告片段（作业 4.1：code snippets + 公式解释 + 第三方库引用）

这份文件是**给写报告的人用的素材**，不是交付物本身。作业 4.1 的原话：

> "DO NOT include all of the source code and complete result sets in your report.
> However, you must include important **code snippets** for the **main functions**
> of your tasks. You should **cite all third-party libraries** used."

所以下面每节给三样东西：**贴进报告的短代码片段**（每段 15 行上下，只留主函数骨架，
注释已按报告口径重写）、**公式逐符号解释**、**可直接引用的数字**。数字全部来自
`part2/results/` 下已生成的产物，不手写。

---

## §3.3.1 倒排索引

### 索引里装了什么（报告用这张表，配一句说明）

| 字段 | 形状 | 谁要用 |
|---|---|---|
| `postings[term][doc_id] = tf` | 词项 → {文档: 词频} | 两个打分函数共用；`df = len(postings[term])`，不用另存 |
| `doc_lengths[doc_id] = dl` | 文档 → 长度（token 数） | BM25 长度归一 |
| `avg_doc_length` | 一个浮点数 | BM25 长度归一，**索引期一次算全** |
| `doc_norms_lnc` / `doc_norms_ltc` | 文档 → ‖d‖ | TF-IDF 余弦的分母，两种权重各一份 |

不存词位置（positions）：本作业没有短语查询，存了也用不上。

### 代码片段 1：索引构建（含 avgdl 的全局计算）

```python
def build_inverted_index(reviews, tokenizer_name="custom"):
    """给一批评论建倒排索引。reviews 逐条流式读入，只取 review_id 与 text。"""
    tokenizer = get_tokenizer(tokenizer_name)      # 与 Part 1 同一套分词器
    postings, doc_lengths, total_tokens = {}, {}, 0
    for review in reviews:
        terms = tokenizer.tokenize(review["text"])  # 文档侧分词
        doc_lengths[review["review_id"]] = len(terms)   # dl
        total_tokens += len(terms)
        for term, tf in Counter(terms).items():
            postings.setdefault(term, {})[review["review_id"]] = tf
    n_docs = len(doc_lengths)
    return InvertedIndex(
        postings=postings,
        doc_lengths=doc_lengths,
        # 余弦分母 ‖d‖ 要等全量 df 出来才能算 → 天然两遍式
        doc_norms_lnc=compute_doc_norms(postings, doc_lengths, log_tf_weight, n_docs),
        doc_norms_ltc=compute_doc_norms(postings, doc_lengths, tfidf_weight, n_docs),
        avg_doc_length=total_tokens / n_docs,       # ← 3.3.2 要求的"索引期全局算"
        tokenizer=tokenizer_name,
    )
```

**为什么 avgdl 必须放在这里**：BM25 的长度归一要用全库的文档平均长度，查询期再算就
得遍历全库或者缓存一个外部文件。放在索引构建的最后一行，是"索引期一次算全"最直接的
写法，也是作业原文的措辞。

### 可直接引用的数字（`part2/results/index_statistics.json`）

| 量 | 值 |
|---|---:|
| 文档数 N | 37,401 |
| 词表大小（不同词项） | 38,120 |
| token 总数 | 3,783,589 |
| **avgdl** | **101.1628** |
| postings 总数（= Σ df） | 2,491,298 |
| 只出现 1 次的词项 | 15,862 |
| dl 最小 / 中位 / 最大 | 4 / 69 / 1,005 |
| 建索引耗时 | 10.9 s（含分词） |
| 与 Part 1 交叉验证 | 篇数 / token 数 / 词表三者**逐位一致** |

df 最高的 5 个词：`and` 33,401、`the` 33,073、`a` 27,959、`to` 27,138、`i` 26,856。
`and` 的 df 已经超过 (N+1)/2 = 18,701——这个数字后面 BM25 的钳制那一节要用。

---

## §3.3.2 实现一：TF-IDF + 余弦

### 公式

```
                    Σ_{t∈Q∩d}  w_q(t) · w_d(t, d)
score(Q, d) = ─────────────────────────────────────
                          ‖q‖ · ‖d‖
```

| 符号 | 是什么 | 从哪来 |
|---|---|---|
| `w_d(t, d)` | 文档侧词项权重，`lnc` = `1 + ln tf`，`ltc` = `(1 + ln tf) · ln(N/df)` | 索引里的 tf 与 df |
| `w_q(t)` | 查询侧权重（ltc 固定）：`(1 + ln tf_q) · ln(N/df)` | 查询分词后现算 |
| `‖d‖` | 文档向量长度 = √Σ_t w_d(t,d)² | **建索引时算好存两份** |
| `‖q‖` | 查询向量长度 = √Σ_t w_q(t)² | 查询期算；对所有文档是同一个因子，**不影响名次**，只影响读数 |
| `df` / `N` | 含该词项的文档数 / 文档总数 | 倒排表长度 / 文档数 |

两个变体的区别**只在文档侧要不要乘 idf**：`lnc.ltc`（默认，Manning 教材推荐）idf 全程
只乘一次；`ltc.ltc`（教科书正文那版）文档和查询各乘一次，分子上等价于 idf²，稀有词
被二次放大。

### 代码片段 2：打分主循环（term-at-a-time）

```python
def tfidf_search(index, query_text, top_k=10, variant="lnc.ltc"):
    """返回 [(doc_id, cosine)]，分数降序、同分按 doc_id 升序。"""
    doc_norms, doc_weight = (index.doc_norms_lnc, log_tf_weight) \
        if variant == "lnc.ltc" else (index.doc_norms_ltc, tfidf_weight)
    terms = get_tokenizer(index.tokenizer).tokenize(query_text)  # 查询侧同一条流水线
    weights = {t: w for t, w in query_weights(index, terms).items() if w}  # 去掉权重 0 的词
    if not weights:
        return []
    query_norm = math.sqrt(sum(w * w for w in weights.values()))
    scores = {}
    for term, query_weight in weights.items():
        for doc_id, tf in index.postings.get(term, {}).items():   # 只扫命中的 posting
            norm = doc_norms[doc_id]
            if norm == 0.0:
                continue                       # 空文档，余弦分母为 0，无相似度可言
            scores[doc_id] = scores.get(doc_id, 0.0) \
                + query_weight * doc_weight(tf, idf) / norm
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [(d, s / query_norm) for d, s in ranked[:top_k]]      # 除 ‖q‖ 才是真余弦
```

**两个值得在报告里点名的实现决定**：
1. **term-at-a-time**：对每个查询词扫一遍它的 posting 列表，只有至少命中一个查询词的
   文档才进候选。37,401 篇里查 `espresso`（df=76）只摸 76 条 posting，不做全库扫描。
2. **权重为 0 的词项直接丢掉**：`df = N` 的词 idf = 0，对所有文档贡献都是 0，留着只会
   多扫一条 posting。

---

## §3.3.2 实现二：Okapi BM25

### 公式

```
                                    (k1 + 1) · tf
score(Q, d) = Σ_{t∈Q}  idf(t) · ─────────────────────────────────
                                 tf + k1 · (1 − b + b · dl / avgdl)
```

| 符号 | 是什么 | 取值 / 来源 |
|---|---|---|
| `tf` | 词项 t 在文档 d 里的出现次数 | 倒排表里的 tf |
| `dl` | 文档 d 的长度（token 数） | 索引的 `doc_lengths` |
| `avgdl` | 全库平均文档长度 = 3,783,589 / 37,401 = **101.1628** | **索引期一次算全** |
| `k1` | 词频饱和速度。tf → ∞ 时该项趋于 `k1 + 1`：出现 20 次不该有 20 倍证据 | 默认 **1.2** |
| `b` | 长度归一强度。`b = 0` 完全不看长度，`b = 1` 完全按长度惩罚 | 默认 **0.75** |
| `idf(t)` | 逆文档频率，三个变体见下 | 见下 |

三个 idf 变体（差异**只在 idf**，其余部分三个变体共用同一行代码）：

| 变体 | 公式 | 出处 / 特点 |
|---|---|---|
| `course` | `ln(N / df)` | 课件版；`df = N` 时为 0 |
| `okapi`（**交付默认**） | `max(0, ln((N − df + 0.5)/(df + 0.5)))` | Robertson & Zaragoza；**会取负值，必须钳** |
| `lucene` | `ln(1 + (N − df + 0.5)/(df + 0.5))` | Lucene 实际实现；加 1 天然非负 |

### 代码片段 3：BM25 打分主循环

```python
def term_weight(tf, dl, avgdl, k1=1.2, b=0.75):
    """词频饱和 + 长度归一因子。tf→∞ 时趋于 (k1+1)；dl 越大分母越大、分数越低。"""
    return tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))


def bm25_search(index, query_text, k1=1.2, b=0.75, variant="okapi", top_k=10):
    """返回 [(doc_id, score)]，降序、同分按 doc_id 升序。k1 / b 是可调参数。"""
    idf = {"course": course_idf, "okapi": okapi_idf, "lucene": lucene_idf}[variant]
    terms = set(get_tokenizer(index.tokenizer).tokenize(query_text))  # 查询按词项集合
    n_docs, scores = len(index.doc_lengths), {}
    for term in terms:
        plist = index.postings.get(term)
        if not plist:
            continue                       # 词不在索引里，对所有文档贡献 0
        weight = idf(len(plist), n_docs)   # df = len(plist)，不用另存
        if weight == 0.0:
            continue                       # 钳到 0 的词区分不了任何文档
        for doc_id, tf in plist.items():
            scores[doc_id] = scores.get(doc_id, 0.0) + weight * term_weight(
                tf, index.doc_lengths[doc_id], index.avg_doc_length, k1, b)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
```

**k1 与 b 的处理**（作业原文点名要求 "correctly handle the parameters k1 and b"）：
两个都是 `bm25_search` 的显式参数，不是常量；命令行 `search_cli.py --k1 --b` 可直接调，
参数敏感性实验跑了完整的 4×4 网格（k1 ∈ {0, 0.9, 1.2, 2.0} × b ∈ {0, 0.3, 0.75, 1.0}）。

---

## 设计辩护（defend 用，作业 2.5 要求"能 defend 每一行提交的代码"）

| 可能被问 | 答案 |
|---|---|
| 为什么用 Python dict 而不是 Lucene？ | 3.3.1 明写二选一；dict 让 df / dl / avgdl / ‖d‖ 四个量都显式可见，便于核对。Lucene 只用来做参照对比（3.3.2 的 "may also"）。 |
| 为什么保留 `(k1 + 1)` 常数？ | 教科书与课件都有它。Lucene 当前主干（10.x）把它去掉了，但它对所有词项相同，**不改排名、只改分数尺度**。保留是为了让三个 idf 变体的差异干净地收敛在一处。**后果**：我们的 BM25 绝对分数是 Lucene 的 2.2 倍，两者分数不可直接比，只能比名次。 |
| 为什么 Okapi idf 要钳到 0？ | `df > (N+1)/2` 时原式为负，意味着"文档里出现了查询词反而扣分"，是教科书明令要避免的病态行为。本语料有词项中招（最高频的 `and`，df = 33,401，原始 idf = −2.12）。**代价要如实写**：Q45 `the and of` 三个词全被钳到 0 → 整条查询返回 0 条结果。 |
| 为什么默认 `lnc.ltc` 而不是 `ltc.ltc`？ | 前者 idf 全程只乘一次，文档向量与全量 df 无关，任何文档的 ‖d‖ 定死，增量索引友好。两者差异实测不小（overlap@10 = 5.14、τ = 0.248），是个真选择，不是无所谓的默认值。 |
| 查询为什么要去重？ | 教科书 BM25 写的是 Σ_{t∈Q}，Q 是词项的**集合**。`pizza pizza` = `pizza`。Lucene 的 BooleanQuery **不去重**，同一词项加两次 SHOULD 分数正好翻倍（实测 2.6385446 → 5.2770890）——对照实验里已在 Java 侧显式去重，避免污染对比。 |
| 同分怎么排？ | 按 `doc_id` 升序，是确定性的 tiebreak，保证两次跑出同样的结果。检索结果可复现比"随机挑一个"重要。 |
| 为什么没有做 P/R/MAP/NDCG？ | 没有相关性标注（qrels）。作业也没要求。所有指标都是**排序一致性**（overlap@k、Kendall τ-b），**不是检索质量**——报告里必须写明这一条。 |

---

## Analysis 章节：每个结论对应哪个产物文件

| 报告要写的点 | 数字在哪 |
|---|---|
| 两个实现在 50 条查询上各自的行为 | `part2/results/impl_analysis.md` |
| 逐条查询的一句定性判读（50 条各一句） | `part2/results/qualitative_reading.md` |
| 钳制前后对比（`fish and chips`） | `part2/results/qualitative_reading.md` 同名小节 |
| overlap@k 与 Kendall τ-b 总表 | `part2/results/query_experiment_summary.md` |
| k1 × b 参数敏感性网格 | `part2/results/query_experiment_summary.md` E7 网格表 |
| Lucene 10.5.2 三方对照（可选加分项） | `part2/results/lucene_comparison.md` |

**三条最硬的结论**（都能追到产物）：
1. 分歧的主导变量是查询词的 **df**，不是查询主题：τ 按 df 分档从稀有词 0.73 掉到超高频 0.23。
2. BM25 真正的旋钮是 **b** 不是 k1：b = 0 时 τ 转负（−0.18 … −0.20）；k1 从 0.9 到 2.0 几乎不影响（overlap@10 8.98–9.20）。
3. b 扫描可证伪且通过：top-10 长度中位随 b 单调下降（b=0 → 285.25，0.75 → 41.25，1.0 → 24.5），证明长度归一真的在起作用；同时说明 b=0.75 在本语料上偏激进（top-10 长度只有均值的 0.41 倍）。

---

## 第三方库引用（4.1 要求 "cite all third-party libraries"）

报告 References 用这一段（ACM sigconf 格式）：

```bibtex
@inproceedings{bird2009nltk,
  author    = {Bird, Steven and Klein, Ewan and Loper, Edward},
  title     = {Natural Language Processing with Python},
  subtitle  = {Analyzing Text with the Natural Language Toolkit},
  booktitle = {Proceedings of the 3rd Workshop on Issues in the Teaching of
               Computational Linguistics},
  pages     = {60--63}, year = {2009}
}
@article{harris2020numpy,
  author  = {Harris, Charles R. and Millman, K. Jarrod and van der Walt,
             St{\'e}fan J. and others},
  title   = {Array programming with {NumPy}},
  journal = {Nature}, volume = {585}, number = {7825},
  pages   = {357--362}, year = {2020}
}
@article{hunter2007matplotlib,
  author  = {Hunter, John D.},
  title   = {Matplotlib: A 2D graphics environment},
  journal = {Computing in Science \& Engineering},
  volume  = {9}, number = {3}, pages = {90--95}, year = {2007}
}
@misc{lucene2025,
  author       = {{The Apache Software Foundation}},
  title        = {Apache Lucene 10.5.2},
  howpublished = {\url{https://lucene.apache.org/core/}}, year = {2025},
  note         = {Used only as a reference implementation for cross-checking
                  ranking; all scoring functions in this work are written
                  from scratch.}
}
@misc{regex2025,
  author = {Barnett, Matthew}, title = {regex --- alternative regular
           expression module},
  howpublished = {\url{https://pypi.org/project/regex/}}, year = {2025}
}
```

纯文本版（若报告不用 BibTeX）：

1. Natural Language Toolkit (NLTK) 3.9.4 — Bird, Klein & Loper, O'Reilly / https://www.nltk.org/ — 仅用于 Part 1 分词器对比的参照实现（WhitespaceTokenizer、WordPunctTokenizer）。
2. regex 2026.5.9 — Matthew Barnett — https://pypi.org/project/regex/ — 自定义分词器依赖（Unicode 属性转义）。
3. NumPy — Harris et al., *Nature* 585 (2020) — Part 1 拟合绘图。
4. Matplotlib — Hunter, *Computing in Science & Engineering* 9(3) (2007) — Part 1 图表。
5. Apache Lucene 10.5.2 — The Apache Software Foundation — https://lucene.apache.org/core/ — 仅作排名参照，打分函数全部自写。
6. Yelp Open Dataset — https://www.yelp.com/dataset — 数据集（不随源码提交）。

**注意版本口径**：`requirements.txt` 钉的是 nltk 3.10.4 / regex 2026.9.29，但报告里的
数字是用 nltk 3.9.4 / regex 2026.5.9 跑出来的。两者都能跑，差别只影响 Part 1 的
分词器对比表，对 Part 2 无任何影响。提交前要么对齐版本，要么按实际版本改 requirements。

---

## 篇幅与禁项提醒

- 报告正文**不超过 10 页**（不含封面、参考文献、附录）。
- **不要**放全量代码、**不要**放完整结果集。上面三段代码片段每段控制在 15 行内即可。
- 已有产物里的长表（50 条逐条明细）适合放**附录**，正文只引聚合值和分档趋势。
- 局限必须写：无相关性标注 → 所有指标是排序一致性而非质量；50 条是 IIR 的经验下限
  而非统计充分，不拿单条查询下结论。
