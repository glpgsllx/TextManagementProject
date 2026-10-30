# Lucene 标准库对照（作业 3.3.2 的 "compare with Lucene"）

- Lucene 10.5.2 / Java 25.0.3，向每个配置要前 200 条后按我们的 tiebreak 重排
- 我们的实现：37401 篇，分词器 custom，avgdl=101.1628

## 1. 两份 Lucene 索引的统计

| 索引 | analyzer | numDocs | sumTotalTermFreq | avgdl |
|---|---|---:|---:|---:|
| `ours_tok` | WhitespaceAnalyzer（不再分词，token 由我们的 custom 分词器给出） | 37401 | 3783589 | 101.1628 |
| `std` | StandardAnalyzer（Lucene 默认，含英文停用词表） | 37401 | 3737126 | 99.9205 |

## 2. 实验组与聚合结果

### E8：`bm25_lucene` vs `lucene_bm25_ours_tok`

差异只在：手搓的 Lucene idf 复刻 vs 真 Lucene BM25：同 token、同 k1/b，只看复刻忠不忠实

| 切片 | 查询数 | overlap@5 | overlap@10 | overlap@20 | overlap@50 | τ 并集 | τ 交集 | 首条相同率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 全部查询 | 50 | 4.72 | 9.32 | 18.76 | 46.9 | 0.9448 | 0.9482 | 0.88 |
| axis=aspect | 12 | 4.8333 | 9.8333 | 19.75 | 49.1667 | 0.9653 | 0.9685 | 0.8333 |
| axis=attribute | 10 | 5.0 | 9.6 | 19.4 | 48.8 | 0.9576 | 0.9588 | 1.0 |
| axis=category | 12 | 5.0 | 9.5833 | 19.75 | 48.8333 | 0.9433 | 0.9462 | 0.9167 |
| axis=df_spectrum | 10 | 4.4 | 8.8 | 17.4 | 44.2 | 0.9395 | 0.9425 | 0.9 |
| axis=edge | 6 | 4.0 | 8.1667 | 16.0 | 39.8333 | 0.8831 | 0.8936 | 0.6667 |

### E9：`bm25_okapi` vs `lucene_bm25_ours_tok`

差异只在：交付口径 BM25（Okapi 钳制 idf + (k1+1)）vs 真 Lucene BM25：同 token，差在 idf 与常数

| 切片 | 查询数 | overlap@5 | overlap@10 | overlap@20 | overlap@50 | τ 并集 | τ 交集 | 首条相同率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 全部查询 | 50 | 4.62 | 9.02 | 18.04 | 45.32 | 0.9115 | 0.9213 | 0.86 |
| axis=aspect | 12 | 4.8333 | 9.5833 | 19.3333 | 47.8333 | 0.8934 | 0.9083 | 0.8333 |
| axis=attribute | 10 | 5.0 | 9.6 | 19.2 | 48.5 | 0.9427 | 0.9455 | 1.0 |
| axis=category | 12 | 5.0 | 9.6667 | 19.6667 | 48.5 | 0.9263 | 0.9362 | 1.0 |
| axis=df_spectrum | 10 | 4.4 | 8.8 | 17.4 | 44.2 | 0.9395 | 0.9425 | 0.9 |
| axis=edge | 6 | 3.1667 | 6.0 | 11.3333 | 30.5 | 0.7805 | 0.8078 | 0.3333 |

### E10：`tfidf_lnc_ltc` vs `lucene_classic_ours_tok`

差异只在：我们的 TF-IDF 余弦 lnc.ltc vs Lucene 内置 ClassicSimilarity：同 token，两套 formula

| 切片 | 查询数 | overlap@5 | overlap@10 | overlap@20 | overlap@50 | τ 并集 | τ 交集 | 首条相同率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 全部查询 | 50 | 3.8 | 7.8 | 16.0 | 40.64 | 0.6431 | 0.6912 | 0.7 |
| axis=aspect | 12 | 4.1667 | 8.0833 | 16.9167 | 45.0833 | 0.6942 | 0.714 | 0.75 |
| axis=attribute | 10 | 4.0 | 9.1 | 17.4 | 45.0 | 0.7313 | 0.7506 | 0.6 |
| axis=category | 12 | 4.0833 | 8.0833 | 17.75 | 43.5 | 0.6866 | 0.7201 | 0.75 |
| axis=df_spectrum | 10 | 3.4 | 7.3 | 14.3 | 35.6 | 0.5912 | 0.658 | 0.7 |
| axis=edge | 6 | 2.8333 | 5.3333 | 11.1667 | 27.1667 | 0.3329 | 0.5081 | 0.6667 |

### E11：`lucene_bm25_ours_tok` vs `lucene_bm25_std`

差异只在：同为 Lucene BM25，只换分词器（我们的 custom vs Lucene StandardAnalyzer）

| 切片 | 查询数 | overlap@5 | overlap@10 | overlap@20 | overlap@50 | τ 并集 | τ 交集 | 首条相同率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 全部查询 | 50 | 4.7 | 9.18 | 18.72 | 46.92 | 0.9449 | 0.951 | 0.94 |
| axis=aspect | 12 | 4.8333 | 9.5833 | 19.5 | 48.6667 | 0.9545 | 0.9644 | 0.9167 |
| axis=attribute | 10 | 5.0 | 9.7 | 19.6 | 49.2 | 0.966 | 0.968 | 1.0 |
| axis=category | 12 | 5.0 | 9.5 | 19.75 | 49.1667 | 0.9631 | 0.9667 | 1.0 |
| axis=df_spectrum | 10 | 4.5 | 8.7 | 17.7 | 44.3 | 0.9335 | 0.9382 | 1.0 |
| axis=edge | 6 | 3.6667 | 7.6667 | 15.3333 | 39.5 | 0.8568 | 0.8703 | 0.6667 |

## 3. 分数级校验（BM25，ours_tok 索引，每条查询的前 10 篇）

用我们索引里的 tf / df / avgdl 复算 Lucene 的 BM25 分数，与 Lucene 实际返回的分数比。
两步归因：残差 A 用我们算的真实 dl，残差 B 用 Lucene explain 里报的 dl；
A − B 就是 Lucene 量化文档长度造成的损失。

| 量 | 比对数 | 中位数 | p90 | 最大 |
|---|---:|---:|---:|---:|
| 残差 A：用真实 dl 复算 | 481 | 6.74e-08 | 0.0082270688 | 0.033755284 |
| 残差 B：用 Lucene 实际 dl 复算 | 481 | 3.09e-08 | 8.02e-08 | 2.717e-07 |
| dl 本身被量化的相对损失 | 481 | 0.0 | 0.04 | 0.0849673203 |

样本（前 8 个）：

| 查询 | Lucene 分数 | 复算（真实 dl） | 复算（Lucene 的 dl） | 真实 dl | Lucene 的 dl |
|---|---:|---:|---:|---:|---:|
| Q01 | 2.638545 | 2.635060 | 2.638544 | 49 | 48.0 |
| Q01 | 2.571478 | 2.571478 | 2.571478 | 17 | 17.0 |
| Q01 | 2.571015 | 2.571015 | 2.571015 | 34 | 34.0 |
| Q01 | 2.566769 | 2.562812 | 2.566769 | 53 | 52.0 |
| Q01 | 2.551016 | 2.547107 | 2.551016 | 57 | 56.0 |
| Q01 | 2.539572 | 2.531286 | 2.539572 | 99 | 96.0 |
| Q01 | 2.538756 | 2.538756 | 2.538756 | 22 | 22.0 |
| Q01 | 2.531863 | 2.527052 | 2.531863 | 43 | 42.0 |

### 3.1 已知语义差异：重复查询词

教科书 BM25 写的是 Σ_{t∈Q}，查询被当成**词项的集合**，所以 `pizza pizza` 与 `pizza` 是同一个信息需求，我们的实现去重。
Lucene 的 BooleanQuery **不去重**：同一个 TermQuery 加两次 SHOULD，
分数正好翻倍。下表是实测（top-1 文档）：

| 查询 | token | 去重分数 | 不去重分数 | 不去重 / 去重 |
|---|---|---:|---:|---:|
| Q48 `pizza pizza` | ['pizza', 'pizza'] | 2.638544 | 5.277089 | 2.0000 |

对照用的 Lucene 结果已在 Java 侧显式去重（两边查询语义一致），
所以这一条不会污染上面的 overlap / τ 读数，但它本身是个要写进报告的
实现差异。

## 4. E9（交付 BM25 vs 真 Lucene BM25）逐条明细

| 查询 | 文本 | 轴 | overlap@10 | overlap@50 | τ 并集 | 首条相同 | BM25 未召回 | Lucene 未召回 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Q01 | pizza | category | 10 | 47 | 0.8737 | 是 | 0 | 0 |
| Q02 | sushi | category | 10 | 48 | 0.8673 | 是 | 0 | 0 |
| Q03 | thai food | category | 10 | 44 | 0.7545 | 是 | 0 | 0 |
| Q04 | taco tuesday | category | 10 | 49 | 0.9655 | 是 | 0 | 0 |
| Q05 | coffee shop | category | 9 | 48 | 0.9261 | 是 | 0 | 0 |
| Q06 | breakfast burrito | category | 9 | 50 | 0.9657 | 是 | 0 | 0 |
| Q07 | ice cream | category | 10 | 49 | 0.9529 | 是 | 0 | 0 |
| Q08 | bbq chicken | category | 9 | 49 | 0.9357 | 是 | 0 | 0 |
| Q09 | barber shop | category | 10 | 50 | 0.9837 | 是 | 0 | 0 |
| Q10 | dentist appointment | category | 10 | 49 | 0.9514 | 是 | 0 | 0 |
| Q11 | oil change | category | 9 | 50 | 0.9722 | 是 | 0 | 0 |
| Q12 | brake repair | category | 10 | 49 | 0.9671 | 是 | 0 | 0 |
| Q13 | terrible service | aspect | 10 | 48 | 0.8582 | 是 | 0 | 0 |
| Q14 | rude staff | aspect | 10 | 48 | 0.9563 | 是 | 0 | 0 |
| Q15 | friendly staff | aspect | 10 | 50 | 0.9804 | 否 | 0 | 0 |
| Q16 | slow service | aspect | 9 | 46 | 0.8519 | 是 | 0 | 0 |
| Q17 | long wait | aspect | 10 | 49 | 0.9467 | 是 | 0 | 0 |
| Q18 | overpriced | aspect | 9 | 50 | 0.9902 | 是 | 0 | 0 |
| Q19 | best pizza in town | aspect | 10 | 43 | 0.7694 | 否 | 0 | 0 |
| Q20 | worst experience ever | aspect | 10 | 48 | 0.9140 | 是 | 0 | 0 |
| Q21 | cold food | aspect | 9 | 47 | 0.7794 | 是 | 0 | 0 |
| Q22 | wrong order | aspect | 9 | 49 | 0.9404 | 是 | 0 | 0 |
| Q23 | dirty bathroom | aspect | 10 | 49 | 0.9404 | 是 | 0 | 0 |
| Q24 | great atmosphere | aspect | 9 | 47 | 0.7939 | 是 | 0 | 0 |
| Q25 | gluten free pancakes | attribute | 9 | 49 | 0.9592 | 是 | 0 | 0 |
| Q26 | vegan options | attribute | 10 | 49 | 0.9498 | 是 | 0 | 0 |
| Q27 | dog friendly patio | attribute | 10 | 49 | 0.9498 | 是 | 0 | 0 |
| Q28 | outdoor seating | attribute | 10 | 49 | 0.9749 | 是 | 0 | 0 |
| Q29 | drive thru | attribute | 9 | 48 | 0.9502 | 是 | 0 | 0 |
| Q30 | free wifi | attribute | 9 | 49 | 0.9702 | 是 | 0 | 0 |
| Q31 | happy hour | attribute | 10 | 49 | 0.9624 | 是 | 0 | 0 |
| Q32 | kid friendly | attribute | 10 | 46 | 0.9287 | 是 | 0 | 0 |
| Q33 | wheelchair accessible | attribute | 9 | 50 | 0.9527 | 是 | 0 | 0 |
| Q34 | takeout order | attribute | 10 | 47 | 0.8287 | 是 | 0 | 0 |
| Q35 | food | df_spectrum | 10 | 49 | 0.8714 | 是 | 0 | 0 |
| Q36 | service | df_spectrum | 10 | 50 | 0.9804 | 是 | 0 | 0 |
| Q37 | staff | df_spectrum | 10 | 49 | 0.9514 | 是 | 0 | 0 |
| Q38 | wait | df_spectrum | 9 | 49 | 0.9106 | 是 | 0 | 0 |
| Q39 | salad | df_spectrum | 9 | 47 | 0.9187 | 是 | 0 | 0 |
| Q40 | burger | df_spectrum | 9 | 49 | 0.9482 | 是 | 0 | 0 |
| Q41 | rude | df_spectrum | 10 | 50 | 0.9510 | 是 | 0 | 0 |
| Q42 | patio | df_spectrum | 10 | 48 | 0.9548 | 否 | 0 | 0 |
| Q43 | espresso | df_spectrum | 10 | 50 | 0.9690 | 是 | 0 | 0 |
| Q44 | xylophone | df_spectrum | 1 | 1 | n/a | 是 | 0 | 0 |
| Q45 | the and of | edge | 0 | 0 | n/a | 否 | 50 | 0 |
| Q46 | the food | edge | 7 | 41 | 0.4164 | 否 | 0 | 9 |
| Q47 | pizza and pasta | edge | 10 | 48 | 0.9321 | 否 | 0 | 0 |
| Q48 | pizza pizza | edge | 10 | 47 | 0.8737 | 是 | 0 | 0 |
| Q49 | don't waste your money | edge | 9 | 47 | 0.8999 | 是 | 0 | 0 |
| Q50 | zzyzx | edge | 0 | 0 | n/a | 否 | 0 | 0 |

