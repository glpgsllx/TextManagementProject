# Part 3：趋势分析

这部分用词典和共现统计分析情感、语义关联随时间的变化，不使用深度学习模型，对应作业 3.4.1 情感趋势和 3.4.2 PMI 漂移。

## 怎么运行

先按[总 README](../README.md) 建好 Python 3.11 环境，然后在仓库根目录执行：

```bash
python -m pip install -r part3/requirements.txt
python -m part3.run
```

结果写到 `part3/results/`。常用参数：

```bash
python -m part3.run --sectors "Italian" "Fast Food" --start-year 2014 --end-year 2021 --bootstrap 2000 --seed 42 \
  --terms service price --window 5 --top-n 20 --min-count 5 --min-businesses 3 --drift-resamples 50
```

`--sectors` 要写两个 Yelp 类别的完整名称，`--terms` 要写两个方面词。默认种子是 42，结果可以复现。全部跑完约 25 秒。

## 情感分析方法

- 评分：VADER 直接读取评论原文，不经过分词器，因为大小写和标点也是它的判断依据。compound ≥ 0.05 记为正面，≤ -0.05 记为负面，其余为中性。
- 行业：`categories` 按逗号拆分后精确匹配。同时属于两个比较行业的商户会被排除。这是按单个商户判断的，所以只排除了一家 Pizza Hut；另外两家只标了 Italian 的 Pizza Hut 仍然保留。
- 时间：默认使用 2014–2021 年。更早的年份评论太少，2022 年数据只到 1 月 19 日。季节按 12–2、3–5、6–8、9–11 月划分，并把各年合并比较。
- 不确定性：均值的 95% 置信区间用 percentile bootstrap，按评论重采样 2,000 次。同一家店的评论并不完全独立，所以区间可能偏窄。
- 重复评论：选中的 4,064 条里有 5 条文本重复，占比很小，没有删除。

## PMI 方法

- 范围：只用 2014–2021 年 `Restaurants` 类的 22,751 条评论，避免商户类型变化被当成语义变化。分词用公共的 `custom` 分词器。
- 公式：`PMI(t, w) = log2(C(t, w) × T / (C(t) × C(w)))`。`C(t, w)` 是 `w` 落在 `t` 前后 5 词内的次数，`T` 是当年总词数，每年分别计算。没有调用现成的 PMI 模块。
- 过滤：去掉停用词、含非字母字符的词和长度小于 3 的词，共现至少 5 次，并且来自至少 3 家商户。`service` 还去掉了 `customer`，因为“customer service”是固定搭配。
- 漂移：看相邻年份、以及每年与 2014 年的 Top-20 Jaccard 重合度。
- 噪声参照：同一年的评论重采样 50 次，算 Top-20 与原结果的重合度。不同年份的重合度只有明显低于它，才可能是真实变化。
- 分时段：把数据合并成 2014–2016、2017–2019、2020–2021 三段，共现门槛提高到 10，减少低频词的干扰。
- 主题：人工把词分成速度、员工态度、好评、差评四组，统计每 100 次 `service` 附近的出现次数。这个方法不受 PMI 偏向低频词的影响，但分组是主观选择，词表写在 `run.py` 里。
- 敏感性：把窗口换成 ±3 和 ±10，比较 Top-20 的重合度。

## 结果在哪里

| 文件 | 看什么 |
| --- | --- |
| `sentiment_trend` | 两个行业每年的 VADER 均值和星级，PNG 和 SVG 各一份 |
| `sentiment_seasonal` | 各季节的 VADER 均值 |
| `sentiment_by_year.csv` | 每年的评论数、商户数、均值、置信区间、正中负比例、星级，以及每家店权重相同的均值 |
| `sentiment_by_season.csv` | 同样的指标，按季节汇总 |
| `sentiment_same_business.csv` | 2015–2016 和 2020–2021 两段时间都至少有 5 条评论的商户，比较其前后变化 |
| `sentiment_cohorts.csv` | 2015 年前已有评论的老商户和之后才出现的新商户，分别的年度均值 |
| `vader_star_validation.csv` | 每个星级的 VADER 均值和正中负比例 |
| `vader_disagreements.csv` | 星级和 VADER 冲突最大的评论，用于人工分析误判 |
| `pmi_drift` | 两个方面词的 Top-20 重合度和噪声参照，PNG 和 SVG 各一份 |
| `pmi_themes` | `service` 附近四组主题词的频率，PNG 和 SVG 各一份 |
| `pmi_top_words_by_year.csv`、`pmi_top_words_by_period.csv` | 按年、按时段的 Top-20 词、PMI、共现次数和商户数 |
| `pmi_drift_by_year.csv`、`pmi_drift_by_period.csv` | 重合度、噪声参照、可入选的词数和新进入 Top-20 的词 |
| `pmi_tracked_words.csv` | 几个代表词每年的排名和 PMI；空白表示没有通过过滤 |
| `pmi_themes.csv` | 主题词频率和 95% 置信区间 |
| `pmi_window_sensitivity.csv` | ±3、±10 窗口与 ±5 窗口的 Top-20 重合度 |

新老商户按所选年份内的首条评论划分，不代表实际开业时间。

## 情感趋势发现

- Fast Food 明显下滑：VADER 均值从 2016 年的 0.614 降到 2021 年的 0.275，两年的置信区间不重叠；星级从 3.68 降到 2.73。
- 这个下滑不是新店造成的：18 家同店样本从 0.481 降到 0.229，其中 14 家下降。老商户在 2021 年只有 0.105，新商户是 0.452。
- Italian 的行业均值大体稳定，但细分结果不一致：8 家同店样本等权平均从 0.616 降到 0.447，其中 5 家下降；老商户按评论加权的均值却基本不变（2016 年 0.575，2021 年 0.585）。样本很小，不能下结论说同店情感下降。
- 季节差异不明显：两个行业各季节的置信区间都有重叠。
- 这些结果只说明时间上的变化，不能证明是疫情或其他单一原因造成的。

## PMI 漂移发现

- `service` 的核心关联很稳定：`slow`、`prompt`、`poor`、`terrible`、`horrible`、`excellent`、`exceptional`、`outstanding`、`notch`、`fast`、`friendly` 在三个时段都进入 Top-20。
- Top-20 的边缘词变化确实超过噪声：相邻年份重合度从 2016 年的 0.54 降到 2020、2021 年的 0.29，同年噪声参照约为 0.48–0.63；只有 2016 年没有低于参照。不过变化的多是低频词，例如 `impeccable` 一年的共现只有 5–7 次，所以这更像用词波动，不算含义转变。
- 主题频率显示好评用词在增加：好评词从 2014 年的每 100 次 39.2 升到 2021 年的 50.3，两端置信区间不重叠。去掉 `great` 后仍从 15.7 升到 22.1。差评词一直在 11–13 之间。这是频率变化，不代表服务真的变好，也可能是评论写法改变。
- 疫情相关词的证据很弱：`covid` 和 `delivery` 在 2020–2021 年才稳定通过过滤，但排名都在 40 以后，共现最多 13 次；`takeout`、`curbside`、`mask` 没有通过过滤。
- `price` 的结果只能作参考：每年可入选的词只有 15–36 个，Top-20 几乎包含全部候选词，2020–2021 段还混入了 `good`、`food`。核心词 `reasonable`、`fair`、`worth`、`portions` 比较稳定。
- 结果对窗口有一定依赖：`service` 的 ±3、±10 窗口与 ±5 的平均重合度是 0.68 和 0.62，`price` 只有 0.48 和 0.45。

PMI 偏向低频词，固定门槛下评论多的年份也更容易入选稀有词，所以逐年排名不能过度解读。噪声参照只反映抽样波动，不能证明变化的原因。

## VADER 的局限

全部 37,401 条评论中，VADER 与星级的 Pearson 相关系数是 0.682。1 星评论有 37.1% 被判为正面。4 星和 5 星的均值分别是 0.856 和 0.889，几乎分不开。

`vader_disagreements.csv` 里常见的误判类型：

- 讽刺："Turbo microwave - speed of light, superhero \"Flash\" fast."
- 长评论先夸后骂：词典分数按全文累加，投诉前的期待和铺垫也会被算成正面。
- 口语褒义词：`killer milk shakes`、`to die for`、`blew all other burritos out of the water` 被当成负面。
- 非英语评论：一条德语 5 星好评得到 -0.931，因为词典只覆盖英语。

## 测试和依赖

```bash
python -m unittest discover -s part3/tests -t . -v
```

分词复用仓库的 `text_processing`。情感分析用 [vaderSentiment](https://github.com/cjhutto/vaderSentiment)（Hutto & Gilbert, 2014），统计用 [NumPy](https://numpy.org/)，画图用 [Matplotlib](https://matplotlib.org/)。版本写在 `requirements.txt` 中，不需要额外下载词典。
