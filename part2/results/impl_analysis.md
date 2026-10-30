# 逐实现分析（作业 3.3.2 的字面要求）

语料 37,401 篇，平均长度 101.16277639635304，查询 50 条，2026-10-30 04:22:43。

这一节回答的是**每个实现各自的行为对不对**，不是两个实现谁更好——没有相关性标注，谁更好测不出来。

## TF-IDF 余弦 lnc.ltc

| 切片 | 查询数 | 召回条数 | 首条相对分差 | 首条并列数 | s10/s1 | top-10 长度中位 | 长度倾向 | 耗时 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 全部 | 50 | 2293.0 | 0.0764 | 1.0 | 0.7496 | 24.0 | 0.2372 | 1.82 |

| aspect | 12 | 8046.0 | 0.0947 | 1.0 | 0.7521 | 22.25 | 0.2199 | 6.07 |
| attribute | 10 | 1727.0 | 0.0613 | 1.0 | 0.6566 | 27.75 | 0.2743 | 1.315 |
| category | 12 | 1170.0 | 0.106 | 1.0 | 0.7101 | 27.25 | 0.2694 | 0.995 |
| df_spectrum | 10 | 1772.5 | 0.0852 | 1.0 | 0.77 | 22.25 | 0.2199 | 1.09 |
| edge | 6 | 28576.0 | 0.0488 | 1.0 | 0.7923 | 16.5 | 0.1631 | 30.345 |

| 0 < df < 300 (稀有) | 4 | 95.5 | 0.0383 | 1.0 | 0.7243 | 46.75 | 0.4621 | 0.075 |
| 300 <= df < 1000 (低频) | 10 | 890.0 | 0.0856 | 1.0 | 0.7161 | 24.25 | 0.2397 | 0.47 |
| 1000 <= df < 3000 (中频) | 12 | 1892.0 | 0.0604 | 1.0 | 0.7808 | 27.75 | 0.2743 | 1.135 |
| 3000 <= df < 10000 (高频) | 11 | 5668 | 0.0863 | 1 | 0.7051 | 24.5 | 0.2422 | 4.49 |
| df >= 10000 (超高频) | 12 | 13505.0 | 0.0563 | 1.0 | 0.7771 | 18.5 | 0.1829 | 11.085 |
| df = 0 (零命中) | 1 | 0 | — | 0 | — | — | — | 0.01 |

## BM25 k1=1.2, b=0.75（交付口径）

| 切片 | 查询数 | 召回条数 | 首条相对分差 | 首条并列数 | s10/s1 | top-10 长度中位 | 长度倾向 | 耗时 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 全部 | 50 | 2079.0 | 0.0241 | 1.0 | 0.8877 | 41.25 | 0.4078 | 2.13 |

| aspect | 12 | 7351.5 | 0.038 | 1.0 | 0.8857 | 37.5 | 0.3707 | 8.91 |
| attribute | 10 | 1727.0 | 0.0344 | 1.0 | 0.8144 | 37.75 | 0.3731 | 2.3 |
| category | 12 | 1170.0 | 0.0165 | 1.0 | 0.9181 | 44.5 | 0.4399 | 1.08 |
| df_spectrum | 10 | 1772.5 | 0.0105 | 1.0 | 0.9468 | 45.5 | 0.4498 | 1.32 |
| edge | 6 | 2053.0 | 0.0163 | 1.0 | 0.8899 | 44.25 | 0.4375 | 1.865 |

| 0 < df < 300 (稀有) | 4 | 95.5 | 0.0439 | 1.0 | 0.7408 | 63.0 | 0.6227 | 0.09 |
| 300 <= df < 1000 (低频) | 10 | 890.0 | 0.0303 | 1.0 | 0.8788 | 38.75 | 0.3831 | 0.58 |
| 1000 <= df < 3000 (中频) | 12 | 1892.0 | 0.0218 | 1.0 | 0.9317 | 44.25 | 0.4375 | 1.545 |
| 3000 <= df < 10000 (高频) | 11 | 5668 | 0.0404 | 1 | 0.887 | 33.5 | 0.3311 | 5.01 |
| df >= 10000 (超高频) | 12 | 12007.0 | 0.0085 | 1.0 | 0.9144 | 45.5 | 0.4498 | 12.445 |
| df = 0 (零命中) | 1 | 0 | — | 0 | — | — | — | 0.0 |

## b 扫描：长度归一是不是真的在压长文档

k1 固定 1.2，只动 b。可证伪的预期是：**b 越大，top-10 的文档长度中位数越小**。

| b | top-10 长度中位 | 长度倾向（÷ 全库均值） | 首条相对分差 | 耗时 ms |
|---:|---:|---:|---:|---:|
| 0.0 | 285.25 | 2.8197 | 0.0229 | 2.08 |
| 0.3 | 118.5 | 1.1713 | 0.0326 | 1.76 |
| 0.75 | 41.25 | 0.4078 | 0.0241 | 2.13 |
| 1.0 | 24.5 | 0.2422 | 0.0112 | 2.38 |

单调下降：**是**，与预期一致。

## 代表查询抽查（top-5，人工看相关性）

每条列出两个实现各自的前 5 篇，附原文片段。**这是唯一能直接看出「检索出来的东西到底对不对」的地方**，但它靠人眼，不靠指标。

### Q01

**TF-IDF 余弦 lnc.ltc**（召回 1,941 条）：

| # | 文档 | 分数 | 长度 | 原文片段 |
|---:|---|---:|---:|---|
| 1 | `GV2fdSGQIe3d…` | 0.478058 | 17 | This was GOOD pizza. A nice alternative to Idaho Pizza. Definitely my new go-to pizza place. |
| 2 | `HZl9SCSg0fus…` | 0.456882 | 12 | This place is great! Delicious pizza, great specialty pizza options. Highly recommend! |
| 3 | `Eq8MXQw6PiE0…` | 0.42598 | 22 | BEST PIZZA EVERRRRRRRRRR!!!! The cheese pizza, cheesy bread, and sun valley veggie pizza were amazing. High quality ingr |
| 4 | `CmH1qqVVBtEi…` | 0.413906 | 15 | Graet pizza, pasta, sandwiches and great service!  In my opinion, best pizza in Treasure Valley. |
| 5 | `mhjCLpxEz9zQ…` | 0.412267 | 16 | Pretty good low cost pizza. $6 for a medium. Tasted better than dominos or Pizza Hut. |

**BM25 k1=1.2, b=0.75（交付口径）**（召回 1,941 条）：

| # | 文档 | 分数 | 长度 | 原文片段 |
|---:|---|---:|---:|---|
| 1 | `Qvy-kB528kr7…` | 5.692673 | 49 | We ordered pizza from Varsity Pizza through Grub Hub.  We got the medium Combo pizza.  The crust was crispy and the topp |
| 2 | `GV2fdSGQIe3d…` | 5.555313 | 17 | This was GOOD pizza. A nice alternative to Idaho Pizza. Definitely my new go-to pizza place. |
| 3 | `KHt_JkAm-6Jr…` | 5.554314 | 34 | Great little pizza spot. NY style pizza with a lot of flop. A bit oily, but solid taste. Good crust. Usually I enjoy pep |
| 4 | `HxxD9L1QD2bC…` | 5.536593 | 53 | Blaze pizza is my all time favorite pizza place! I mean what's not to like about it? They have amazing customer service  |
| 5 | `zWsciATjJs1N…` | 5.502665 | 57 | This place rocks. It's like a subway but for pizza. You can select a signature pizza or for $7.55 you can build a pizza  |

### Q25

**TF-IDF 余弦 lnc.ltc**（召回 1,843 条）：

| # | 文档 | 分数 | 长度 | 原文片段 |
|---:|---|---:|---:|---|
| 1 | `O_fr1k1BiqRh…` | 0.415525 | 65 | Amazing staff, out of this world food!! I am gluten sensitive & this place will make anything gluten free & the pancakes |
| 2 | `djIltsOkyK42…` | 0.383776 | 19 | Mediocre gluten free pancakes but excellent allergy awareness and courteous staff. Great atmosphere and prompt seating d |
| 3 | `zir_LSXZW67d…` | 0.382482 | 8 | Great Pizza! They have gluten free options too! |
| 4 | `YELC2OtcbpWM…` | 0.375984 | 22 | Great healthy low carb and gluten free options. All of the donuts can be made gluten free. I highly recommend this spot |
| 5 | `slKzA1oZdFk9…` | 0.36414 | 42 | A great place to go for Gluten Free peeps!  They have everything gluten free marked (on a separate gluten free menu).  P |

**BM25 k1=1.2, b=0.75（交付口径）**（召回 1,843 条）：

| # | 文档 | 分数 | 长度 | 原文片段 |
|---:|---|---:|---:|---|
| 1 | `O_fr1k1BiqRh…` | 21.081032 | 65 | Amazing staff, out of this world food!! I am gluten sensitive & this place will make anything gluten free & the pancakes |
| 2 | `M36lgQfiQl-7…` | 19.333566 | 69 | Gluten-Free In Boise!!! Best kept secret in town- 90%of the menu is Gluten free!  I'm in town from California and didn't |
| 3 | `DptIUgXHB6RE…` | 18.949567 | 42 | Even though it was lunch time, I can't pass up gluten free pancakes! They were delicious. They also make gluten free Fre |
| 4 | `GAaRKSzQblJv…` | 18.949567 | 42 | Gluten free pancakes are a win!! Super good flavor, consistency, generous portion, and overall appeal! Been a big fan of |
| 5 | `djIltsOkyK42…` | 18.625495 | 19 | Mediocre gluten free pancakes but excellent allergy awareness and courteous staff. Great atmosphere and prompt seating d |

### Q35

**TF-IDF 余弦 lnc.ltc**（召回 13,139 条）：

| # | 文档 | 分数 | 长度 | 原文片段 |
|---:|---|---:|---:|---|
| 1 | `jdewChZ5pZgw…` | 0.5 | 4 | Great food, slow service.... |
| 2 | `9ts5HjiSLVEv…` | 0.494289 | 10 | Awesome food and fast service .  Homemade and creative food choices |
| 3 | `RhlvQeKz4-Od…` | 0.491506 | 11 | Awesome food. Best Mexican food around and the service is superb. |
| 4 | `-jizmznSLHFz…` | 0.441105 | 13 | Great bartenders good atmosphere and the food truck had great affordable Mexican food |
| 5 | `ykfiAenLWtHF…` | 0.439124 | 14 | Good food! Fun atmosphere, great service and authentic Mexican food. We will be back. |

**BM25 k1=1.2, b=0.75（交付口径）**（召回 13,139 条）：

| # | 文档 | 分数 | 长度 | 原文片段 |
|---:|---|---:|---:|---|
| 1 | `dUNCYlPxGbri…` | 1.176992 | 65 | Great drinks and decent service  but unfortunately the food was a poor excuse for Mexican food. My pacifico burrito was  |
| 2 | `lWnmQgwNFrgp…` | 1.172654 | 34 | You're going to pay restaurant prices and wait restaurant times for your fast food quality food. You're better off decid |
| 3 | `Dk0jN9GSKuGC…` | 1.170391 | 35 | Wonderful food and service.   My new favorite fir good Mexican food.  We had a party of 6 and food was fast hot and grea |
| 4 | `_10declOgQwv…` | 1.170391 | 35 | Excellent food! Superb & fast service. Best Chinese food place around. Nicely fried and freshly made you can taste the q |
| 5 | `ZGLW30TCOqvQ…` | 1.167113 | 54 | Ordered food at 6:29 and food arrived at 9. The food is not edible and the smoothies are water. This is by far the worst |

### Q43

**TF-IDF 余弦 lnc.ltc**（召回 76 条）：

| # | 文档 | 分数 | 长度 | 原文片段 |
|---:|---|---:|---:|---|
| 1 | `IaiZDzqmjFnW…` | 0.208514 | 23 | Never miss out when in the Boise area! Full espresso bar, fresh specialties & killer jams! Omelettes, eggs, benedicts, c |
| 2 | `QxyT5QrIhRsT…` | 0.200535 | 24 | My favorite place for coffee in meridian. They genuinely kind, make excellent espresso, and always give my dog love thro |
| 3 | `BRh2QLCghId4…` | 0.180382 | 29 | Good customer service. Decent coffee, messed up order twice but fixed it both times. Espresso sometimes tastes burnt but |
| 4 | `Y58qrE0FKZRQ…` | 0.180382 | 29 | Good customer service. Decent coffee, messed up order twice but fixed it both times. Espresso sometimes tastes burnt but |
| 5 | `qbP3I6qMqvzq…` | 0.178491 | 117 | The food is decent, nothing to write home about. The drinks, however, are awesome! Whatever your taste they have somethi |

**BM25 k1=1.2, b=0.75（交付口径）**（召回 76 条）：

| # | 文档 | 分数 | 长度 | 原文片段 |
|---:|---|---:|---:|---|
| 1 | `qbP3I6qMqvzq…` | 9.411634 | 117 | The food is decent, nothing to write home about. The drinks, however, are awesome! Whatever your taste they have somethi |
| 2 | `IaiZDzqmjFnW…` | 9.050992 | 23 | Never miss out when in the Boise area! Full espresso bar, fresh specialties & killer jams! Omelettes, eggs, benedicts, c |
| 3 | `XnBPSDtaqiPk…` | 9.016895 | 81 | My wife and I just love Lucky Perk.  They are very consistent with their drinks and always taste delicious.  I get the S |
| 4 | `QxyT5QrIhRsT…` | 8.99779 | 24 | My favorite place for coffee in meridian. They genuinely kind, make excellent espresso, and always give my dog love thro |
| 5 | `BRh2QLCghId4…` | 8.740892 | 29 | Good customer service. Decent coffee, messed up order twice but fixed it both times. Espresso sometimes tastes burnt but |

### Q45

**TF-IDF 余弦 lnc.ltc**（召回 36,558 条）：

| # | 文档 | 分数 | 长度 | 原文片段 |
|---:|---|---:|---:|---|
| 1 | `Bm4m5IAZgRW7…` | 0.562615 | 12 | good food nice ambiance and in the middle of lots of places |
| 2 | `zEB5xowb2Frg…` | 0.526698 | 11 | Delicious, lots of options including non-dairy, and tons of toppings! |
| 3 | `S-MjVKXgcnL7…` | 0.506642 | 15 | Probably one of my two favorites Grimaldi's and Tony's in the side of Egyptian Theater |
| 4 | `kywoACF2f0MO…` | 0.50381 | 16 | Great friendly service. No pressure. Good selection of clothing and of course great selection of Harleys. |
| 5 | `QXUbPtemxQgN…` | 0.479235 | 16 | Love the fry selection. Sauces are awesome too. Thousands of combinations of fries, salts and sauces. |

**BM25 k1=1.2, b=0.75（交付口径）**（召回 0 条）：

| # | 文档 | 分数 | 长度 | 原文片段 |
|---:|---|---:|---:|---|

### Q50

**TF-IDF 余弦 lnc.ltc**（召回 0 条）：

| # | 文档 | 分数 | 长度 | 原文片段 |
|---:|---|---:|---:|---|

**BM25 k1=1.2, b=0.75（交付口径）**（召回 0 条）：

| # | 文档 | 分数 | 长度 | 原文片段 |
|---:|---|---:|---:|---|

## 需要单独说明的异常

### TF-IDF 余弦 lnc.ltc

- **零命中**（返回 0 条）：`Q50` zzyzx
- **首条并列**（第一名不是唯一最高分）：`Q08`（2 篇同分）、`Q26`（2 篇同分）
- **分数拉不开**（s10/s1 > 0.95）：无

### BM25 k1=1.2, b=0.75（交付口径）

- **零命中**（返回 0 条）：`Q45` the and of、`Q50` zzyzx
- **首条并列**（第一名不是唯一最高分）：无
- **分数拉不开**（s10/s1 > 0.95）：`Q01`、`Q05`、`Q07`、`Q15`、`Q35`、`Q37`、`Q38`、`Q40`

