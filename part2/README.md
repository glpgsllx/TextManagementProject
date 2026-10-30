# Part 2：倒排索引与两个打分函数

这部分做作业 3.3.1 的倒排索引，以及 3.3.2 里的 TF-IDF + 余弦和 Okapi BM25 两个打分函数（两个共用同一份索引）。数据直接读仓库里的 `data/meridian_id/`，和 Part 1 用的是同一份。

口径与 Part 1 保持一致：文档是一条评论，`doc_id` 用 `review_id`；词项用公共分词器的 `custom` 方法，停用词保留、不做词干提取；`dl` 是分词后的词项数。评论和查询必须用同一个分词器，换方法要重新建索引。

## 怎么运行

先按[总 README](../README.md) 建好 Python 3.11 环境，然后在仓库根目录执行：

```bash
python -m pip install -r part2/requirements.txt
python -m part2.run
```

运行会打印一份索引统计，同时写到 `part2/results/index_statistics.json`。只处理前 N 条评论、或者换分词器时：

```bash
python -m part2.run --limit 2000 --tokenizer wordpunct
```

索引存在 `part2/.cache/index.pkl`。语料或分词器换过之后，直接跑会报
"索引与当前语料不一致，请用 --rebuild 重建"，加 `--rebuild` 重新构建并覆盖：

```bash
python -m part2.run --rebuild
```

## 索引里有什么

`inverted_index.py` 里只有一个 `InvertedIndex`，字段就是索引的全部内容：

| 字段 | 含义 | 谁用 |
| --- | --- | --- |
| `postings[term][doc_id]` | 词项在某篇评论里出现的次数 tf | 两个打分函数都要 |
| `doc_lengths[doc_id]` | 该评论分词后的词项数 dl | BM25 的长度归一化 |
| `doc_norms_lnc[doc_id]` | 文档侧 lnc 权重的向量长度 ‖d‖ | TF-IDF 余弦（lnc.ltc）的分母 |
| `doc_norms_ltc[doc_id]` | 文档侧 ltc 权重的向量长度 ‖d‖ | TF-IDF 余弦（ltc.ltc）的分母 |
| `avg_doc_length` | 全语料平均文档长度 avgdl，索引期一次算全 | BM25 |
| `total_tokens` | 全语料总词项数，用来和 Part 1 交叉验证 | 报告 |
| `df(term)` | 含该词项的文档数，不在索引里就是 0 | 两个打分函数的 idf |

不存词项位置（本作业没有短语查询），不做压缩。索引在内存里建好，顺手 pickle 落盘到
`part2/.cache/index.pkl`（20 MB），之后再跑就直接读回来：构建 10.9 秒，读回 0.3 秒。
落盘时一起记下语料的文件名、字节数、条数和分词器，读回时只要有一项对不上就报错提醒
重建，不会拿着旧索引跑新语料。`.cache/` 已加进 `.gitignore`，不进仓库。

构建分两遍：第一遍逐篇分词，累加 postings 和 dl；第二遍算 ‖d‖，其中含 idf 的那份依赖全量 df，只能等第一遍结束。词项权重只在 `inverted_index.py` 里定义一次（`log_tf_weight`、`tfidf_weight`、`inverse_document_frequency` 三个函数），第二遍算范数和后面查询期算分数共用它们，改口径只改这一处。

存两份 ‖d‖ 是因为 TF-IDF 有两种经典权重组合，文档向量不同、长度就不同，余弦的分母必须和分子配同一套。

## 全量数字

2026-10-29 在 Meridian, ID 全量语料上实测（custom 分词器，Python 3.13 / rag-dev 环境）：

| 指标 | 数值 | 交叉验证 |
| --- | --- | --- |
| 评论数 N | 37,401 | 与 `manifest.json` 一致 |
| 词表大小 | 38,120 | 与 Part 1 `tokenizer_comparison.csv` 的 custom 行一致 |
| 总词项数 | 3,783,589 | 同上 |
| avgdl | 101.1628 | — |
| posting 条目数 Σdf | 2,491,298 | — |
| df = 1 的词项 | 15,862（占词表 41.6%） | — |
| dl | min 4 / median 69 / max 1,005 | — |
| df 最高的词项 | and 33,401、the 33,073、a 27,959、to 27,138、i 26,856 | — |
| 构建耗时 | 10.9 秒（含读文件、分词、两遍构建、落盘） | — |
| 读回耗时 | 0.3 秒 | — |
| 索引文件 | 20 MB（`part2/.cache/index.pkl`） | — |

三个数（37,401 / 3,783,589 / 38,120）对上了 Part 1，说明这份索引和 Part 1 数的是同一批词。`python -m part2.run` 每次都会自动做这个对齐，结果放在输出的 `cross_check_part1` 里。

查询侧的开销只和词项的 df 有关，因为走倒排表遍历的是 postings 而不是全库：df 最大的 `and`（33,401 条）扫一遍约 25 ms，中频词如 `pizza`（1,941）约 0.6 ms、`sushi`（967）约 0.3 ms。

## TF-IDF 检索：两个变体

`tfidf.py` 按 SMART 三元组记号 `ddd.qqq` 实现两种权重组合，**左边三个字母描述文档向量、右边三个字母描述查询向量**（Manning《Introduction to Information Retrieval》就是这个顺序）：

| 位置 | 字母 | 含义 |
| --- | --- | --- |
| 第 1 位（词频） | `l` | `1 + ln(tf)`，出现 20 次不该比 1 次重 20 倍 |
| 第 2 位（idf） | `n` / `t` | 不乘 / 乘 `ln(N / df)` |
| 第 3 位（归一化） | `c` | 向量除以自己的长度 ‖·‖ |

| 变体 | 文档侧权重 | 查询侧权重 | 说明 |
| --- | --- | --- | --- |
| `lnc.ltc`（默认） | `1 + ln tf` | `(1 + ln tf_q) · ln(N/df)` | 课上讲的那一套，也是 Manning 教材推荐的默认组合；idf 全程只出现一次 |
| `ltc.ltc` | `(1 + ln tf) · ln(N/df)` | 同上 | "每篇文档做成一个 TF-IDF 向量，再对查询向量求余弦"的直译，也是教科书正文里最常见的那版 |

两者差别只在文档侧要不要乘 idf，但影响不小：`ltc.ltc` 分子上等价于 idf²，稀有词被二次放大；而且文档里那些**没被查询命中**的稀有词会把 ‖d‖ 撑大却不贡献分子，反而拖低该文档。副作用是 `lnc.ltc` 的文档向量与全集的 df 无关，任何 ‖d‖ 都是定死的，增量加文档时不用重算。

```python
from part2.tfidf import tfidf_search, LNC_LTC, LTC_LTC

results = tfidf_search(index, "gluten free pancakes", variant=LNC_LTC)
for doc_id, score in results:
    print(score, doc_id)
```

`tfidf_search` 收自然语言原文，内部用建索引时那个分词器切词（查询和文档必须同口径），返回 `[(doc_id, 分数)]`，分数是真的余弦值、降序，默认取 top 10。

2026-10-29 在全量语料上实测：

| 查询 | lnc.ltc 首条余弦 | ltc.ltc 首条余弦 | 两者 top-10 重合 |
| --- | --- | --- | --- |
| `gluten free pancakes` | 0.4155 | 0.5864 | 7/10 |
| `terrible service` | 0.3627 | 0.4651 | 4/10 |
| `best pizza in town` | 0.4744 | 0.6135 | 7/10 |
| `pizza and pasta` | 0.4278 | 0.4562 | 6/10 |
| `Thai food` | 0.4846 | 0.6294 | 4/10 |

两个变体的分数量纲不同（跨模型不能比绝对值，只能比名次），名次也有实质分歧。顺带一提，`best pizza in town` 的首条命中评论压根没提 pizza，靠 `best` + `town` 两个词赢过了更长的真实 pizza 评论——向量空间模型求的是向量夹角短文档的天然优势，这条留给后面的定性分析。

## BM25：三个 idf 变体

`bm25.py` 实现三个变体，它们**共用同一个词频饱和 + 长度归一因子，只有 idf 不同**：

    score(Q, d) = Σ_{t∈Q}  idf(t) · (k1 + 1) · tf / (tf + k1 · (1 − b + b · dl / avgdl))

| 变体 | idf 公式 | 出处 | 特点 |
| --- | --- | --- | --- |
| `course` 课件版 | `ln(N / df)` | 课件 Lecture 9.1 第 18–19 页原文 | 和 TF-IDF 用的是同一个 idf；df = N 时为 0 |
| `okapi` Okapi 版 | `max(0, ln((N − df + 0.5)/(df + 0.5)))` | Robertson & Zaragoza 2009，无相关反馈时 R = r = 0 的退化式 | **原始值会取负**，df > (N+1)/2 时必须钳到 0 |
| `lucene` Lucene 版 | `ln(1 + (N − df + 0.5)/(df + 0.5))` | Lucene 8.x / 9.x `BM25Similarity.idf()` 官方文档原文 | 加 1 规避负 idf，不需要硬钳 |

默认 `k1 = 1.2`、`b = 0.75`（Lucene 与 Elasticsearch 的默认值），都是函数参数，作业原文要求 "correctly handle the parameters k1 and b"。

```python
from part2.bm25 import bm25_search, COURSE, OKAPI, LUCENE

bm25_search(index, "gluten free pancakes", variant=OKAPI)          # 默认 okapi
bm25_search(index, "terrible service", k1=2.0, b=0.3)              # 调参
bm25_search(index, "pizza", variant=LUCENE, top_k=5)
```

**为什么不装 PyLucene**：Lucene 版本是按官方源码手工复现的，没有装 PyLucene。Windows 上装它需要 JDK + JCC 编译，成本远超收益；而 `BM25Similarity` 的公式是公开的，`idf()` 和 `avgFieldLength()` 的写法在官方文档里逐字给出，照抄即可。源码见 `lucene/core/src/java/org/apache/lucene/search/similarities/BM25Similarity.java`。对照时有个坑：**Lucene 10 主干去掉了对所有词项都相同的常数 `(k1 + 1)`**（8.x/9.x 里是 `boostVal = boost * (k1 + 1)`）。那个常数不影响排名，只改分数尺度，所以这里保留教科书共有的 `(k1 + 1)`，让三个变体的差别干净地收敛在 idf 一处。

### 实测（2026-10-30，全量 37,401 条）

| 词项 | df | 课件版 `ln(N/df)` | Okapi 钳后 | Lucene | Okapi 原始值 |
| --- | --- | --- | --- | --- | --- |
| and | 33,401 | 0.1131 | 0 | 0.1131 | **−2.1222** |
| the | 33,073 | 0.1230 | 0 | 0.1230 | **−2.0335** |
| a | 27,959 | 0.2910 | 0 | 0.2910 | **−1.0855** |
| food | 13,139 | 1.0461 | 0.6133 | 1.0461 | — |
| service | 11,347 | 1.1927 | 0.8312 | 1.1927 | — |
| pizza | 1,941 | 2.9585 | 2.9050 | 2.9583 | — |

两条结论：

1. **全语料有 12 个词项的 Okapi 原始 idf 为负**（df > (N+1)/2），最高频的 `and` 是 −2.12。不钳制的话，"文档里出现了查询词"会反过来扣分。
2. **课件版和 Lucene 版几乎重合**（`ln((N+1)/(df+0.5))` ≈ `ln(N/df)`，差在第 4 位小数）；真正和它们拉开距离的是 **Okapi 版**——它的 idf 在 df 接近 N 时掉得快得多（food：0.61 vs 1.05）。6 条查询上课件版与 Lucene 版的 top-10 重合是 10/10，Okapi 版与它们的重合是 9–10/10，`Thai food` 这条连首条都不一样。

单条查询耗时 6–111 ms，取决于查询词的 df 之和。三条独立验证：

- **两条计算路径对账**：把 37,401 篇重新分词、按 document-at-a-time 重算，与倒排表 term-at-a-time 的结果在 6 条查询 × 3 个变体（18 组）上名次全部一致，分数最大偏差 7.1×10⁻¹⁵。
- **参数退化行为**：`k1 = 0` 退化成 idf 求和（与 tf 无关）；`b = 0` 与 `b = 0.75` 在小语料上会把前两名交换——长度归一确实在起作用，不是摆设。
- **钳制的定性影响**：小语料里 `the` 出现在 3/4 篇时，Okapi 版只返回 1 篇结果，课件版和 Lucene 版返回 3 篇。钳制改变的不只是分数，还有**结果集本身**。

### 实验用接口：不钳制的 Okapi idf

做消融实验要拿到"没钳制会怎样"的对照，所以额外给了 `okapi_idf_unclamped(df, n_docs)`，即 `ln((N − df + 0.5)/(df + 0.5))` 去掉 `max(0, ·)`。

**它刻意没有登记进变体表 `IDF_BY_VARIANT`**，所以 `bm25_search(index, q, variant=...)` 无论传什么都选不到它，遍历变体的循环也不会误撞。要用必须显式走后门：

```python
from part2.bm25 import bm25_search, okapi_idf_unclamped, OKAPI

bm25_search(index, "pizza and pasta", variant=OKAPI)                  # 交付口径，钳制
bm25_search(index, "pizza and pasta", idf_fn=okapi_idf_unclamped)     # 实验口径，不钳制
```

`idf_fn` 一传入就完全接管 idf，`variant` 不再参与。测试里有 9 条断言专门守住这件事：猜名字（`"okapi_raw"` 等）拿不到、把函数本身当变体名传也拿不到、只有 `idf_fn=` 这一条路能拿到、不传 `idf_fn` 时三个变体的结果与加这个接口之前逐位一致。

**别在交付路径里用它**：负 idf 会让"文档里出现了查询词"反过来扣分，是教科书明令要避免的病态行为。它的唯一用途是量化钳制的影响，下面是全量语料上的对照结果：

| 查询 | 钳制 vs 不钳制的 top-10 重合 | 首条是否相同 | 说明 |
| --- | --- | --- | --- |
| `gluten free pancakes` | 10/10 | 是 | 查询词都是中低频，钳制是空操作 |
| `terrible service` | 10/10 | 是 | 同上 |
| `Thai food` | 10/10 | 是 | 同上 |
| `best pizza in town` | 10/10 | 是 | 首条分数掉了 0.225（`in` 的负 idf） |
| `fish and chips` | 7/10 | **否** | `and` 的负 idf 重新洗牌 |
| `pizza and pasta` | 6/10 | **否** | 同上 |
| `the food` | 1/10 | **否** | `the` 原始 idf = −2.03，几乎完全重排 |

关键读数：**钳制只影响 12 个词项（占词表 0.031%），但一旦查询里带上了它们，排名就会显著改变**。所以这 12 个词既是"钳制是必要的"的证据，也是"钳制的影响面其实很窄"的证据——两条都值得写进报告。

## 打分函数怎么用

```python
from pathlib import Path

from part2.inverted_index import build_inverted_index, load_index
from part2.run import read_reviews

index = build_inverted_index(read_reviews(Path("data/meridian_id/review.json")))
# 或直接读已落盘的索引：index = load_index(Path("part2/.cache/index.pkl"))

print(index.df("pizza"))          # 1941
print(index.avg_doc_length)       # 101.1628
for doc_id, tf in index.postings["sushi"].items():
    print(doc_id, tf, index.doc_lengths[doc_id], index.doc_norms_lnc[doc_id])
```

查询词不在索引里时 `index.df(term)` 返回 0、`index.postings[term]` 取不到，打分函数按"该词不贡献分数"处理即可。空评论（dl = 0）的 ‖d‖ 也是 0，余弦分母为 0 时跳过这篇，不会除零。

## 测试和依赖

```bash
python -m unittest discover -s part2/tests -v
```

测试用一份 4 篇的手算小语料：

- `test_inverted_index.py` 逐位核对 postings、df、dl、avgdl，以及两份 ‖d‖，另外覆盖 df = 0、df = N、空评论、空语料四种边界。
- `test_tfidf.py` 核对查询权重和两套余弦分数（含"单查询词时余弦恰为 1""重复查询词不影响结果""两个变体可以排出不同名次"这些能还原出算式的性质），边界覆盖索引外的词、空查询、空文档、top-k 截断、非法变体名。
- `test_bm25.py` 核对三个 idf 变体在 df = 1 / 2 / 3（含 df > N/2）下的取值、词频饱和因子的手算值、三个变体的完整分数，以及 `k1 = 0` 退化成 idf 求和、`b = 0` 与 `b = 0.75` 交换名次这两条参数性质。钳制的定性影响单独用一份 4 篇小语料钉住；不钳制的实验接口另有一组 9 条断言，一半验算术、一半守"它不能被默认触发"。

另外做过一次独立复算：把 37,401 篇重新分词、按 document-at-a-time 从头算一遍，与倒排表 term-at-a-time 的结果在 TF-IDF 两个变体和 BM25 三个变体上名次全部一致，分数最大偏差 7.1×10⁻¹⁵（浮点累加顺序差异）。两条不同的计算路径对得上，这是除手算之外最硬的验证。

分词用到 [regex](https://github.com/mrabarnett/mrab-regex) 和 [NLTK](https://www.nltk.org/api/nltk.tokenize.regexp.html)，由公共包 `text_processing` 引入，版本写在 `requirements.txt` 中。索引本身只用标准库（`math`、`collections`、`dataclasses`），没有用任何现成的检索或排序库。

## 查询集与两项分析（作业 3.3.2）

作业原文是 "Run a set of at least five diverse queries on **both implementations**"。
注意它要求的是**两个实现各自都要跑**，"各自跑"和"拉到一起比"是两件事。所以这里拆成
两项分析，主次分明：

| 分析 | 脚本 | 回答什么 | 作业定位 |
| --- | --- | --- | --- |
| **逐实现分析** | `impl_analysis.py` | 每个实现的行为对不对（长度倾向、首条分差、零命中…） | **字面要求** |
| 分歧机制分析 | `query_experiment.py` | 两个实现为什么分歧、分歧由什么变量控制 | 附带，作业未要求 |

前者不需要相关性标注就能算，且能直接落到"实现是否符合公式预期"。后者测的是排序
一致性，**没有 qrels 就判不了谁更好**——这是它的天花板，下面每一节都按这个口径写。

### 查询集：50 条，不是 5 条

作业原文的下限是 5 条，我们按 50 条建。依据是 Manning/Raghavan/Schütze《Introduction to
Information Retrieval》§8.1 的原文：

> "The test document collection and suite of information needs have to be of a reasonable
> size: you need to average performance over fairly large test sets, as results are highly
> variable over different documents and information needs. **As a rule of thumb, 50
> information needs has usually been found to be a sufficient minimum.**"

这条经验法则针对的是"要在多条查询上平均，因为单条查询的波动极大"。我们这份作业没有
相关性标注，但波动这一点同样成立：`pizza` 上两个实现的 top-10 重合 2/10，`espresso`
上是 10/10，单看任何一条都会得出错误结论。

50 条按五个轴铺开（`queries.py` 里每条都写了为什么选它）：

| 轴 | 条数 | 选这些查询是为了看什么 |
| --- | ---: | --- |
| `category` | 12 | 业态/主题检索（pizza、sushi、brake repair…），Yelp 上真实存在的店类 |
| `aspect` | 12 | 方面 + 情感（terrible service、rude staff 及其极性反转对照） |
| `attribute` | 10 | 属性约束（gluten free pancakes、dog friendly patio…），最接近真实信息需求 |
| `df_spectrum` | 10 | 单词项按 df 从 13,139 铺到 1，用来检验"查询词稀有度"是不是分歧的解释变量 |
| `edge` | 6 | 边界输入：全停用词、负 idf 词、重复词、缩写展开、df = 0 零命中 |

df 分档在跑的时候按索引实测现算（`profile_query`），不写死在表里，换语料后自动跟着变。

### 怎么跑

```bash
python -m part2.impl_analysis                        # 逐实现分析：2 个交付实现 + b 扫描，约 3s
python -m part2.query_experiment                      # 分歧机制：50 条 × 6 个基础条件 + 16 组参数网格，约 1 分钟
python -m part2.query_experiment --skip-grid          # 只跑基础条件
python -m part2.query_experiment --axes edge          # 只跑某个轴
python -m part2.search_cli "gluten free pancakes" --all   # 单条查询现场看两个实现的前 10 条
```

产出在 `part2/results/`：`impl_analysis.md` / `.json`（逐实现画像）、
`query_experiment_summary.md`（分歧机制，给人看的表格）、
`query_experiment_summary.json`（逐条 + 聚合）、`query_experiment_runs.json`（原始 top-50）、
`query_experiment_overlap.csv`（长表，画图用）。

---

### 逐实现分析：每个实现各自的行为（作业要求的那一项）

`impl_analysis.py` 给两个交付实现（`tfidf_lnc_ltc`、`bm25_okapi`）各画一张像。指标全部
**不需要相关性标注**，但都直接指向"这个实现的行为对不对"：

| 指标 | 定义 | 读到什么 |
| --- | --- | --- |
| 召回条数 | 该查询返回了多少篇 | 0 条 = 查询被整个跳过（BM25 钳制会这样） |
| 首条相对分差 | (s₁ − s₂) / \|s₁\| | 接近 0 = 第一名不是赢出来的，是从并列堆里取的 |
| 首条并列数 | 与首条同分的文档数 | >1 = 换 tiebreak 就换第一名 |
| s10/s1 | 第 10 名 ÷ 第 1 名的分数 | 接近 1 = 前十名分数拉不开 |
| 长度倾向 | top-10 长度中位数 ÷ 全库平均长度 | 1.0 不偏、>1 偏长、<1 偏短 |

**50 条查询的读数（2026-10-30）：**

| 实现 | 召回条数 | 首条相对分差 | s10/s1 | top-10 长度中位 | 长度倾向 | 耗时 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TF-IDF 余弦 `lnc.ltc` | 2,293 | 0.0764 | 0.7496 | 24.0 | **0.2372** | 1.73 |
| BM25 `okapi` (1.2, 0.75) | 2,079 | 0.0241 | 0.8877 | 41.25 | **0.4078** | 1.90 |

全库平均长度 101.16。两条都是**显著偏短**，TF-IDF 尤甚（top-10 中位数只有均值的 0.24
倍）。这不是 bug，是归一化的必然结果：余弦把文档向量除到单位长度，短文档更容易与查询
向量对齐；BM25 的 `b·dl/avgdl` 也是同样的方向。但**偏短是不是更好，测不出来**——那需要
标注。

**b 扫描是这条结论的可证伪检验**（k1 固定 1.2）：如果长度归一真的在起作用，top-10 的
长度中位数必须随 b 单调下降。实测：

| b | 0.0 | 0.3 | 0.75 | 1.0 |
| --- | ---: | ---: | ---: | ---: |
| top-10 长度中位 | 285.25 | 118.5 | 41.25 | 24.5 |
| 长度倾向 | 2.8197 | 1.1713 | 0.4078 | 0.2422 |

**单调下降，与预期一致**。b = 0（完全不做长度归一）时 top-10 中位长度是全库均值的
**2.82 倍**——长文档因为词多、命中查询词的概率高而被顶上来；b 越大压得越狠，b = 1.0 时
只剩 0.24 倍。这条曲线同时说明 b = 0.75 这个默认值放在本语料上是**偏激进**的，它把
top-10 压到了均值的 0.41 倍。

**两个实现各自暴露的问题（都写进 `impl_analysis.md` 的"异常"一节）：**

- BM25 有 **2 条零命中**：`Q45 the and of`（三个词 idf 全被钳成 0 → 整条查询被跳过）、
  `Q50 zzyzx`（df = 0）。TF-IDF 只有后者一条。
- TF-IDF 有 **2 条首条并列**（`Q08 bbq chicken`、`Q26 vegan options`，各 2 篇同分）——
  这两条的第一名实际由 tiebreak（doc_id）决定。BM25 没有。
- BM25 有 **10 条查询 s10/s1 > 0.95**（前十名分数几乎拉不开，名次基本是并列噪声），
  TF-IDF **0 条**。BM25 的 top-10 分数集中度明显更高。

这个偏短不是抽象数字，抽查 `Q01 pizza` 的 top-5 一眼就能看见：TF-IDF 的 5 篇长度是
17 / 12 / 22 / 15 / 16 词，全是 "This was GOOD pizza." 这类一句话评价；BM25 的是
49 / 17 / 34 / 53 / 57 词，是真正描述店和口味的评价。两者都"关于 pizza"，但抓到的
是完全不同的一类文档——这不是谁错，是长度归一化的直接后果。

代表查询（五个轴各一条 + df = 0 那条）的 top-5 原文片段在 `impl_analysis.md` 里，靠人眼
判断相关性——这是全套实验里唯一一处能直接看出"检索出来的东西对不对"的地方。

### W5 对比分析：五项核心项都落在哪里

工作项清单里 W5 列了五项（"—"那项明确不做）。逐项对照：

| # | 项 | 状态 | 落在哪 |
| ---: | --- | --- | --- |
| 1 | overlap@k + Kendall τ（每 query） | 完成 | `query_experiment.py` E1–E7，`query_experiment_summary.md` |
| 2 | 钳制前后对比（`fish and chips`） | 完成 | `qualitative_reading.md` 的"钳制前后对比" |
| 3 | 参数敏感性（k1 × b） | 完成 | E7 的 4 × 4 全网格（不是压缩网格），`query_experiment_summary.md` |
| 4 | **定性判读 top 结果（每 query 一句）** | 完成 | `qualitative_reading.py` + `qualitative_reading.md`，50 条各一句 |
| 5 | Lucene 三方对比 | 完成 | `lucene_experiment.py` E8–E11 |
| — | precision / recall / MAP | **不做** | 无相关性标注，算不了；已在"局限"写明原因 |

**第 2 项的结果本身就是一条结论**：`fish and chips` 在钳制下召回 1,358 条、前三条全都在讲
fish and chips；不钳制下召回 33,462 条、前三条里只有一条勉强相关（另两条是海鲜店和烧烤）。
原因是 "and" 的 df = 33,401 > (N+1)/2，不钳制时它的 idf 为负，把大量不含 fish/chips 的
文档顶了上来。这不是"钳制让分数更准"，而是**钳制把负 idf 词剔除出了打分**。

**第 4 项的归纳（50 条逐条判读后浮现的四个现象）**：

- **A** TF-IDF 的首条几乎总是极短（4–20 词）、逐字复述查询词的评价——余弦归一化的必然后果；
- **B** BM25 的首条更长、信息更多，但在高 df 查询上更容易漂到不相关的长评；
- **C** 语料本身存在重复/近重复评价（Q08、Q19、Q26、Q33、Q31 都撞到了），会同时污染
  两个实现的 top-k 多样性——这是**数据问题，不是模型问题**；
- **D** 多词查询里 TF-IDF 可能漏掉核心词：Q19 `best pizza in town` 的首条是
  "One of the best burgers in town"，命中 best/in/town 却漏掉 pizza。

两实现 **top-1 相同 25/50 条**（与 E1 的首条相同率 0.50 对得上）、
**top-3 集合完全相同仅 7/50 条**——集合层面的一致比首条层面的要低得多。

---

### 以下都是附带分析（作业未要求）

作业只要"两个实现各自跑一遍"。下面这些是我们**额外**做的，价值在于解释分歧机制，
**但都不能用来判断谁更好**——没有 qrels，判断不了。

#### 度量：只有两个，因为没有标注

没有 qrels，P@10 / MAP / NDCG 一律不算。能算的只有"两个排序有多像"，两层各一个指标
（实现在 `rank_metrics.py`，每一条性质都有手算测试钉着）：

| 指标 | 看什么 | 读法 |
| --- | --- | --- |
| `overlap@k` | 集合 | 两侧前 k 条里有几条是同一篇，"7/10" = 前 10 条里 7 条共同 |
| Kendall τ-b | 顺序 | +1 完全一致、0 无关、−1 完全相反；并列从分母扣除 |

τ 同时报**并集**（两侧的 top-50 并集，含成员差异）和**交集**（两侧都进了 top-50 的文档，
纯看次序）。一侧压根没召回的文档在该侧记为"最末名之后"（截尾），数量在
`n_missing_a` / `n_missing_b` 里报出来——看到它们不为 0，就知道 τ 里混着成员差异。

#### 实验组：每组的差异都收敛在一处

| 组 | 条件 A | 条件 B | 唯一的差异 | 50 条平均 overlap@5 / @10 / @20 / @50 | τ 并集 | 首条相同率 |
| --- | --- | --- | --- | --- | ---: | ---: |
| E1 | `tfidf_lnc_ltc` | `bm25_okapi` | **整套打分公式**（分歧机制分析，作业未要求） | 2.38 / 5.50 / 12.22 / 34.22 | 0.397 | 0.50 |
| E2 | `tfidf_lnc_ltc` | `tfidf_ltc_ltc` | 文档侧要不要乘 idf | 2.38 / 5.14 / 11.70 / 31.86 | 0.248 | 0.40 |
| E3 | `bm25_okapi` | `bm25_course` | 只换 idf（Okapi 钳制式 vs 课件式） | 4.66 / 9.24 / 18.34 / 46.04 | 0.953 | 0.92 |
| E4 | `bm25_okapi` | `bm25_lucene` | 只换 idf（Okapi 钳制式 vs Lucene 式） | 4.66 / 9.24 / 18.34 / 46.02 | 0.953 | 0.92 |
| E5 | `bm25_course` | `bm25_lucene` | 只换 idf（课件式 vs Lucene 式） | 4.82 / 9.62 / 19.22 / 48.00 | 1.000 | 0.98 |
| E6 | `bm25_okapi` | `bm25_okapi_unclamped` | 只去掉 `max(0, ·)` 钳制 | 4.62 / 9.16 / 18.30 / 45.66 | 0.961 | 0.92 |
| E7 | `bm25_okapi` (1.2, 0.75) | `bm25_k1_*_b_*` | 只改 `k1` / `b`（4 × 4 网格） | 见下 | 见下 | 见下 |

E7 的 k1 × b 网格（每行是与默认 k1 = 1.2 / b = 0.75 比，overlap@10 与 τ 并集）：

| k1 \ b | 0.0 | 0.3 | 0.75 | 1.0 |
| --- | --- | --- | --- | --- |
| 0.0 | 1.04 / −0.293 | 1.04 / −0.293 | 1.04 / −0.293 | 1.04 / −0.293 |
| 0.9 | 1.94 / −0.178 | 4.30 / 0.239 | 9.20 / 0.949 | 5.86 / 0.480 |
| 1.2 | 1.92 / −0.185 | 4.22 / 0.238 | 默认 | 5.86 / 0.457 |
| 2.0 | 1.70 / −0.200 | 4.06 / 0.224 | 8.98 / 0.913 | 5.56 / 0.400 |

#### 读数（全量 37,401 条，2026-10-30）

1. **分歧的主导因素是查询词的 df，不是查询的主题。** E1 按 df 分档的 τ 并集：稀有 0.73、
   低频 0.55、中频 0.34、高频 0.39、超高频 0.23（overlap@10 从稀有的 6.5 掉到超高频的
   4.25）。高频档略高于中频档，不完全单调，但两端差了两倍多。单词查询 `espresso`
   （df = 76）两侧 top-10 全同；`food`（df = 13,139）只有 1/10，τ = −0.13。原因不是谁对
   谁错，而是高频词候选太多、大量文档得分接近，排名天然不稳定。
2. **"换模型"不是最大的变量。** E2（TF-IDF 内部换权重组合）的 τ = 0.248，比 E1
   （跨模型）的 0.397 还低——`lnc.ltc` 与 `ltc.ltc` 的分歧，不小于 TF-IDF 与 BM25 的分歧。
   报告里不能只说"TF-IDF 和 BM25 不同"，权重组合的选择同样量级。
3. **BM25 的三个 idf 变体几乎等价**（E5：overlap@10 = 9.62/10，τ = 0.9997）。课件式与
   Lucene 式在真实语料上几乎重合，Okapi 式与它们 τ ≈ 0.95——idf 的选法改变不了结果，
   真正改变结果的是词频饱和与长度归一（见下一条）。
4. **`b` 是 BM25 真正的旋钮，`k1` 几乎无所谓。** b = 0（不做长度归一）时 τ 转负
   （−0.18 … −0.20）、overlap@10 只有 1.7–1.9；b = 1.0 过度惩罚长文档，也掉到 5.6–5.9；
   k1 从 0.9 到 2.0 几乎不影响（8.98–9.20）。k1 = 0 是退化点：词频饱和因子恒为 1，
   所有命中同一批查询词的文档同分，排名完全由 tiebreak（doc_id）决定，所以 τ ≈ −0.29
   且 b 取什么都一样。
5. **钳制的影响面很窄、但很深。** E6 在四个非 edge 轴上 overlap@10 = 9.1–10、τ = 0.995–1.0
   （df_spectrum 的 9.1 是被 `xylophone` 只有 1 篇候选拉低的，与钳制无关）；6 条 edge
   查询掉到 overlap@10 = 4.5、τ = 0.559。`the and of` 这条最极端：BM25 把三个词的 idf
   全钳成 0 后整条查询被跳过、返回 0 条结果，而 TF-IDF 仍返回满屏（idf 只是很小，不是
   0）——"钳到 0"和"idf 趋于 0"是两件不同的事，前者是彻底不参与打分。

#### 局限（报告里要如实写）

- 没有相关性标注，所以以上**只是排序一致性，不是检索质量**。两个实现都排到同一批
  文档，不代表这批文档相关。
- 50 条是 IIR 的经验下限，不是"统计上足够"。单条查询的读数仍然抖（overlap@10 的
  标准差很大），所以报告里只引聚合值和分档趋势，不拿单条查询下结论。
- τ 的并集口径含成员差异，交集口径样本小且偏向"两边都认可的文档"，两个必须一起报。

## Lucene 标准库对照（作业 3.3.2 的最后一句）

3.3.2 在"两个实现"之后还有一句可选的话：

> "You may also run the same queries and compare your results with those returned
> by a standard library such as Lucene, e.g., the built-in ranking modules."

这一句由 `lucene_experiment.py` + `lucene/LuceneCompare.java` 承担，真跑 Apache
Lucene **10.5.2**（JDK 25）。它**不改动**上面 E1–E7 任何一个实验组，而是新增 E8–E11
作为第三方参照列。

### 两份索引，把"公式差异"和"分词差异"拆开

| 索引 | analyzer | 用途 |
| --- | --- | --- |
| `ours_tok` | WhitespaceAnalyzer，喂**我们自己切好的 token** | 两侧 token 逐条一致，剩下的差异纯粹是打分实现 |
| `std` | StandardAnalyzer（含英文停用词表） | Lucene 开箱即用的方式；与上一行的差就是分词器贡献 |

两侧 token 一致性是**实测**的，不是声明的：`ours_tok` 索引的 `sumTotalTermFreq =
3,783,589`、`avgdl = 101.16277639635304`，与我们的索引**逐位相同**。

每个索引上跑 `BM25Similarity(1.2, 0.75)` 与 `ClassicSimilarity`。

```bash
python -m part2.lucene_experiment                 # 导出语料 → 编译/跑 Java → 对比，约 20s
python -m part2.lucene_experiment --skip-run      # Java 跑过了，只重算对比
```
产出：`part2/results/lucene_comparison.md` / `.json`。

### 实验组与结果（50 条查询，overlap@5 / @10 / @20 / @50，τ 并集）

| 组 | 条件 A | 条件 B | 唯一的差异 | @5 | @10 | @20 | @50 | τ 并集 | 首条相同率 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E8 | `bm25_lucene`（手搓的 Lucene idf 复刻） | `lucene_bm25_ours_tok`（真 Lucene） | 同 token、同 k1/b，只看复刻忠不忠实 | 4.72 | 9.32 | 18.76 | 46.90 | 0.9448 | 0.88 |
| E9 | `bm25_okapi`（交付口径） | `lucene_bm25_ours_tok` | 差在 idf 选式与 `(k1+1)` 常数 | 4.62 | 9.02 | 18.04 | 45.32 | 0.9115 | 0.86 |
| E10 | `tfidf_lnc_ltc`（我们的 TF-IDF 余弦） | `lucene_classic_ours_tok` | 两套 TF-IDF 公式 | 3.80 | 7.80 | 16.00 | 40.64 | 0.6431 | 0.70 |
| E11 | `lucene_bm25_ours_tok` | `lucene_bm25_std` | 同为 Lucene BM25，只换分词器 | 4.70 | 9.18 | 18.72 | 46.92 | 0.9449 | 0.94 |

### 分数级校验：我们对 Lucene 公式的理解是逐项正确的

用我们索引里的 tf / df / avgdl 复算 Lucene 的 BM25 分数（481 个 (查询, 文档) 对）：

| 量 | 中位数 | p90 | 最大 |
| --- | ---: | ---: | ---: |
| 残差 A：用**真实 dl** 复算 | 6.7e-08 | 0.0082 | 0.0338 |
| 残差 B：用 **Lucene explain 里报的 dl** 复算 | 3.1e-08 | 8.0e-08 | 2.7e-07 |
| dl 被量化的相对损失 | 0.0 | 0.04 | 0.085 |

残差 B 只剩 **float32 精度**（Lucene 全程 float、我们全程 double），说明 idf 式
`ln(1+(N−df+0.5)/(df+0.5))`、tf 饱和 `tf/(tf+k1(1−b+b·dl/avgdl))`、无 `(k1+1)` 常数、
`N = docCount`、`avgdl = sumTotalTermFreq/numDocs` 这一整套，我们和 Lucene **逐项一致**。
残差 A 与 B 的差，全部来自下面这一条。

### 三处实打实的差异（都是写报告的料）

1. **Lucene 的文档长度是量化值。** `explain` 里那句 `dl, length of field
   (approximate)` 不是客套：norm 里存的不是真实长度。实测 top 结果上相对损失
   中位数 0、p90 4%、最大 8.5%（dl ≤ 40 无损，越长越粗）。它把 BM25 分数拉偏
   0.8%（p90）、最多 3.4%，**只影响分数，几乎不影响排序**。
2. **钳制 vs 加 1：唯一一处真分歧。** Q45 `the and of`，三个词的 df 分别是
   33,073 / 33,401 / 20,940，都大于 `(N+1)/2 = 18,701`。Okapi 原式 idf 全为负，
   我们的实现钳到 0 → 三个词项全跳过 → **返回 0 条结果**；Lucene 用 `ln(1+·)`
   天然非负，仍返回 ≥ 2,593 条。这是 E9 在 edge 轴上 τ 掉到 0.78、首条相同率
   掉到 0.33 的唯一原因。
3. **重复查询词。** 教科书 BM25 写的是 Σ_{t∈Q}，查询是词项的**集合**，
   `pizza pizza` = `pizza`，我们去重。Lucene 的 `BooleanQuery` 不去重——同一个
   `TermQuery` 加两次 SHOULD，分数正好翻倍（Q48 top-1 实测：去重 2.6385446、
   不去重 5.2770890，比值 2.0000）。对照用的 Lucene 结果已在 Java 侧显式去重，
   这条不会污染上面的 overlap / τ。

### 结论：BM25 可以让 Lucene 作数，TF-IDF 不行

- **BM25 方向差异很小**（E8：overlap@10 = 9.32/10、τ = 0.9448；E9：9.02/10、
  τ = 0.9115）。所以报告里讲"我们的 BM25 与标准库一致"这句话，现在有真 Lucene
  的实测数字撑着，不再依赖我们对 Lucene 公式的复刻。E4 / E5 里"Lucene 版 idf"
  的说法应当改引 E8。
- **TF-IDF 方向差异大**（E10：τ = 0.6431），但这**不是我们的实现有问题**：
  Lucene 的 `ClassicSimilarity` 是另一套 TF-IDF（`√tf · idf / √dl`，norm 还是
  3 位小数量化的字节），不是我们实现的余弦归一化 `lnc.ltc`。两个都是 TF-IDF，
  但不是一个公式，不能互相替换。
- **E1 不能被 Lucene 替换**。3.3.2 要求 TF-IDF 与 BM25 都 "from scratch（do not
  use built-in ranking modules）"，"跑一批 diverse query 在两个实现上"比的就是
  这两个手搓实现。Lucene 是加分项，是参照列，不是替身。

### 局限

- Lucene 侧只取前 200 条后按我们的 tiebreak 重排，因此"我方进了 top-50、
  Lucene 排在 200 名之后"的文档在 Lucene 侧记为 201 名（截尾），
  `n_missing_b` 会把它报出来。E9 的 Q46 有 9 篇属于这种情况。
- `totalHits` 在 `TOP_SCORES` 收集模式下是下界，不拿它当召回条数用。
