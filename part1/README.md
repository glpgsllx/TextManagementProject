# Part 1：统计与正则分词器

本部分交付三项结果：**词频与 Zipf 图、词表增长与 Heaps 拟合、自写正则分词器与两种基线的比较**。统计使用 Meridian, Idaho 的全部 37,401 条评论；原始评论始终保留在数据文件中。

## 运行与交付文件

从仓库根目录执行，先按[根目录说明](../README.md)安装依赖并准备数据：

```bash
# 查看一条文本的分词结果
python -m part1 --text "I don't like the-best-food 😊!"
python -m part1 --text "Don't go" --keep-contractions
python -m part1 --tokenizer wordpunct --text "Don't go!"

# 完整实验；也可指定输入目录、输出目录与主采样种子
python -m part1.run
python -m part1.run --input-dir data/meridian_id --output-dir tmp/part1-check --seed 42

# 公共接口、分词边界、缩写开关、采样和已知幂律的测试
python -m unittest discover -v
```

| 输出文件（在 `results/`） | 用途 |
| --- | --- |
| `summary.json` | 语料校验、处理策略、拟合参数和运行环境 |
| `tokenizer_comparison.csv` | 四组配置的 token / type 数量 |
| `term_frequencies_*.csv` | 各方法的完整排序词频，使用全语料出现次数 |
| `heaps_samples.csv` / `heaps_parameters.csv` | 各采样规模与种子的计数和参数 |
| `sampling_manifest.json` | 可复现的评论排序与抽样记录 |
| `tokenization_examples.json` / `regex_rules.json` | 真实及合成输入输出、正则优先级 |
| `zipf` / `heaps` / `tokenizer_comparison` 的 PNG、SVG | 报告用图 |

代码职责：`run.py` 串联数据读取与实验；`statistics.py` 负责计数和拟合；`plots.py` 负责绘图；`boundary_cases.py` 保存合成边界案例。共享分词规则在仓库根目录的 `text_processing/` 中。

## 分词器规则与接口

采用“**识别边界 → 统一归一化**”两步。`surface_tokens()` 用于观察边界，`tokenize()` 返回统计和检索用的术语列表。三个方法共用归一化函数，比较中保留各自边界造成的差异。

```python
from text_processing import get_tokenizer

t = get_tokenizer("custom")
print(t.surface_tokens("I don’t like the-best-food 😊!"))
# ['I', 'don’t', 'like', 'the', 'best', 'food', '😊', '!']
print(t.tokenize("I don’t like the-best-food 😊!"))
# ['i', 'do', 'not', 'like', 'the', 'best', 'food', '😊']

print(get_tokenizer(expand_contractions=False).tokenize("Don’t go"))
# ["don't", 'go']
```

### 边界怎么确定

自写规则在 `text_processing/custom.py`，按以下优先级匹配：

1. 完整 emoji 序列：包括肤色、ZWJ 组合、旗帜和键帽，例如 `👩🏽‍🍳`、`🇸🇬`、`1️⃣` 各为一个 token。
2. ASCII 表情：例如 `:-)`、`:(`、`<3` 各为一个 token。
3. 含字母的词：允许内部撇号和字母数字混合，例如 `don't`、`B52`、`24hr`。
4. 数字：保留内部小数点和逗号，例如 `1,200.50`。
5. 连字符和破折号：作为分隔符丢弃，`the-best-food → the, best, food`。
6. 其余非空白字符：先作为独立标点保留，让归一化阶段决定是否使用。

`regex` 库提供 Unicode 字符属性；以上边界规则由本项目编写。

### 归一化怎么做

| 规则 | 输入 → 术语输出 |
| --- | --- |
| NFC、统一撇号、单词转小写 | `DON’T → do, not`；组合重音字母统一为 NFC |
| 明确缩写展开 | `I'm → i, am`；`we've → we, have`；`won't → will, not` |
| 少量例外词表 | `can't / cannot → can, not`；`shan't → shall, not`；`let's → let, us`；`y'all → you, all` |
| 保留歧义形式 | `she's → she's`；`I'd → i'd`；`Alice's → alice's`；`ain't → ain't` |
| 保留数字、完整 emoji 和 ASCII 表情 | `$4.89 😊 :-) → 4.89, 😊, :-)` |
| 去除其他纯标点 token | `food!!! → food`（custom 的边界规则先拆出 `!`） |

`'re / 've / 'll` 作为后缀展开；`n't` 仅在已列出的助动词后展开，也支持 `wouldn't've → would, not, have`。缩写展开要求整个 token 是一个识别出的词。ASCII 表情保留大小写，以免改变 `:D` 的形状。

保留停用词，不做 stemming、lemmatization 或拼写纠正。歧义缩写需要语法上下文，本规则不猜测其含义。

### 如何比较不同模块

```python
from text_processing import available_tokenizers, get_tokenizer

text = "I don’t like the-best-food 😊!"
for name in available_tokenizers():
    print(name, get_tokenizer(name).tokenize(text))
```

实际输出：

```text
custom ['i', 'do', 'not', 'like', 'the', 'best', 'food', '😊']
whitespace ['i', 'do', 'not', 'like', 'the-best-food']
wordpunct ['i', 'don', 't', 'like', 'the', 'best', 'food']
```

基线使用 NLTK `WhitespaceTokenizer` 和 `WordPunctTokenizer`。共同归一化只删除**纯标点 token**，所以空白分词产生的 `food!` 仍是 `food!`；WordPunct 拆开的 `don`、`t` 也不会重新拼回缩写。`custom_no_expand` 是关闭缩写展开的消融配置，使用同一个 custom 分词器。

## 本次实验结果

以下为共享语料、共享归一化策略下的计数：token 是总出现次数，type 是不同字符串数。

| 方法 | 边界 token 数 | 边界 type 数 | 归一化 token 数 | 归一化 type 数 |
| --- | ---: | ---: | ---: | ---: |
| Custom | 4,243,374 | 49,849 | 3,783,589 | 38,120 |
| Whitespace | 3,728,711 | 120,284 | 3,760,186 | 103,566 |
| WordPunct | 4,346,210 | 48,574 | 3,806,763 | 36,361 |
| Custom，关闭缩写展开 | 4,243,374 | 49,849 | 3,738,918 | 38,169 |

缩写展开覆盖 44,671 次词出现，归一化词表减少 49 个 type。Whitespace 的标点附着产生较多变体；WordPunct 的词表更小，同时会拆开缩写、小数和部分表情，**词表大小本身不代表分词准确率**。具体例子见 `tokenization_examples.json`。

### Zipf：排名和出现次数

使用 custom 归一化词频，考察 `f(r) = C × r^(-s)`。在 log10 空间拟合，预设主范围为排名 10–10,000、出现次数至少 2：

- `s = 1.4695`，log10 空间 `R² = 0.9869`。
- 38,120 个 type 中，14,964 个仅出现一次。
- 排名 1–100、10–1,000、100–10,000 的指数分别约为 0.8667、1.1064、1.5204；不同区段有不同斜率，整体具有长尾结构，但不能声称全局严格满足 `1/r`。

### Heaps：token 增多时词表怎样增长

拟合 `V = k × N^β`，其中 `N` 是**实际累计 token 数**，`V` 是不同术语数。按 `SHA256(seed + NUL + review_id)` 排序，取嵌套前缀；评论规模按比例向上取整。

主种子 42 的四个作业检查点：

| 评论占比 | 评论数 | N：token 数 | V：type 数 |
| --- | ---: | ---: | ---: |
| 10% | 3,741 | 386,662 | 14,091 |
| 30% | 11,221 | 1,137,626 | 22,974 |
| 50% | 18,701 | 1,908,342 | 28,637 |
| 100% | 37,401 | 3,783,589 | 38,120 |

四点拟合得到 `k = 51.3780`、`β = 0.4369`、log10 空间 `R² = 0.9995`。仅用 10% / 30% / 50% 拟合，预测全量词表为 39,007，实际 38,120，高估约 2.33%。另以种子 43–46 检查排序影响，五个种子的 `β` 范围为 0.4239–0.4369；图中也展示更密的采样点。

这些是本城市、当前分词策略下的描述性拟合；嵌套采样点并非独立样本。

### Emoji 覆盖如何验证

真实 Meridian 评论中，当前规则匹配到 6 条评论中的 13 个 Unicode 符号序列：`® × 6`、`‼ × 4`、`™ × 2`、`⁉ × 1`，均为文字样式符号；ASCII 表情出现在 886 条评论中。`😊`、肤色、ZWJ、旗帜和键帽等边界由合成输入单独测试。17 个合成案例用于验证规则，**不加入 Zipf、Heaps 或词表统计**。

## 课件对应

页码为 PDF 物理页码；课件在仓库上一级的 `slides/`。

| 本部分知识点 | 对应课件 |
| --- | --- |
| token / type | Lecture 2 — Text Normalization，p.4 |
| 连字符和分词边界 | Lecture 2，p.11 |
| 规则与正则分词 | Lecture 2，pp.12–13 |
| 大小写归一化 | Lecture 2，p.29 |
| Heaps 定律 | Lecture 8 — Index，pp.23–24 |
| Zipf 定律 | Lecture 8，pp.25–26 |

## 实现参考

- [NLTK TweetTokenizer 源码](https://www.nltk.org/_modules/nltk/tokenize/casual.html)：参考有序正则规则的组织思路；本项目自写规则，未调用 TweetTokenizer。
- [NLTK 正则分词器文档](https://www.nltk.org/api/nltk.tokenize.regexp.html)：两种标准基线。
- [regex 项目](https://github.com/mrabarnett/mrab-regex)：Unicode 属性支持。
- 作业要求：仓库根目录 `GroupAssignment-202609.pdf`，pp.2–3，§3.2。
