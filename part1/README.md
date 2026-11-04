# Part 1：数据统计和正则分词器

这部分做词频统计、Zipf 和 Heaps 拟合，并比较自写正则、Whitespace、WordPunct 三种分词器。数据直接读取仓库里的 `data/meridian_id/`。

## 怎么运行

先按[总 README](../README.md) 建好 Python 3.11 环境，然后在仓库根目录执行：

```bash
python -m pip install -r part1/requirements.txt
python -m part1.run
```

运行后，图表和结果表会写到 `part1/results/`。需要换路径或采样种子时，可以这样写：

```bash
python -m part1.run --input-dir data/meridian_id --output-dir part1/results --seed 42
```

默认种子是 42，评论按 `SHA256(seed + NUL + review_id)` 排序。Heaps 使用同一条归一化 token 流，从 1,000 个 token 到全量取 80 个对数间隔的检查点，统一在 log-log 空间拟合。

1k、10k、100k、1M token 和 10%、30%、50%、100% 评论的位置也会记录在同一张表里。`fit_checkpoint` 标明哪些点属于固定拟合网格；额外标注点不会改变拟合权重，`review_fraction` 标明评论比例。每个规模都是前一个规模的延伸，必要时在评论中间截断。

全量预测单独检验：只用 `N <= 全量 token 数的 80%` 范围内的拟合点估计参数，再预测全量词表。`heaps_prediction.csv` 记录这个预测模型的参数和误差；`law_parameters.csv` 则记录全部 80 个点拟合的描述模型。

## 看一条文本怎么分词

```bash
python -m part1 --text "I don't like the-best-food 😊!"
```

输出有两组：

```text
surface_tokens:   ["I", "don't", "like", "the", "best", "food", "😊", "!"]
normalized_terms: ["i", "do", "not", "like", "the", "best", "food", "😊"]
```

需要做归一化时，使用 `normalized_terms`；查看原始分词边界时，使用 `surface_tokens`。命令行加 `--tokenizer whitespace` 或 `--tokenizer wordpunct` 可以换方法。

自写规则会保留完整 emoji、ASCII 表情和小数，拆开连字符。归一化时统一 Unicode 等价写法和撇号，把普通词转成小写，再展开明确的缩写。`she's`、`I'd` 这类有歧义的形式保留。停用词也保留，不做词干提取或词形还原。代码在 `text_processing/`，其他组员可以直接复用。

## 结果在哪里

`results/` 中有三张图，每张都提供 PNG 和 SVG：

- `zipf`：词频与排名的双对数图。
- `heaps`：统一的词表增长曲线、80 点拟合和全量预测。
- `tokenizer_comparison`：三种分词器的词表大小。

对应的表格是：

| 文件 | 看什么 |
| --- | --- |
| `tokenizer_comparison.csv` | 各方法的总词数、不同词数和只出现一次的词数 |
| `heaps_samples.csv` | 统一增长曲线的所有观测点、拟合标记和评论比例 |
| `heaps_prediction.csv` | 留出全量结果后的词表预测、参数和误差 |
| `law_parameters.csv` | Zipf、Heaps 的拟合参数、范围和 log-log R² |
| `tokenization_examples.csv` | 同一段文字经过三种分词器后的结果 |

Zipf 使用 `f(r) = C × r^(-s)`，表中的 coefficient、exponent 分别是 C、s；Heaps 使用 `V = k × N^β`，对应 k、β。示例表会注明文字来自真实评论还是合成案例，合成案例不参与词频和词表统计。

Heaps 图按约 85 mm（3.35 英寸）的单栏宽度制作，文字为 9 pt。插入报告时按单栏宽度使用，避免再次缩小导致字体难读。

## 测试和依赖

```bash
python -m unittest discover -s part1/tests -v
```

分词用到 [regex](https://github.com/mrabarnett/mrab-regex) 和 [NLTK](https://www.nltk.org/api/nltk.tokenize.regexp.html)，拟合用 [NumPy](https://numpy.org/)，画图用 [Matplotlib](https://matplotlib.org/)。版本写在两个 `requirements.txt` 中，不需要下载 NLTK 模型。

有序正则的组织方式参考了 [NLTK TweetTokenizer](https://www.nltk.org/_modules/nltk/tokenize/casual.html)，自写分词器使用自己的边界规则，没有调用 TweetTokenizer。
