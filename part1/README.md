# Part 1 — Dataset Analysis & Custom Tokenization

从仓库根目录运行。共享数据准备和公共分词接口见[总 README](../README.md)。

## 安装和运行

使用 Python 3.11，并激活虚拟环境：

```bash
python -m pip install -r part1/requirements.txt
python -m part1.run
```

输入为 `data/meridian_id/business.json`、`review.json` 与提取脚本生成的 `manifest.json`。两个数据文件均为 UTF-8 JSON Lines，每行一个 JSON 对象。

可选参数：

```bash
python -m part1.run --input-dir data/meridian_id --output-dir part1/results --seed 42
```

`--input-dir` 指向 Meridian 子集；`--output-dir` 指定最终输出目录；`--seed` 控制可复现的嵌套采样顺序，默认 42。Heaps 的 N 使用实际累计 token 数。

## 分词器示例

```bash
python -m part1 --text "I don't like the-best-food 😊!"
python -m part1 --tokenizer wordpunct --text "Don't go!"
python -m part1 --keep-contractions --text "Don't go!"
```

第一条命令输出：

```json
{
  "surface_tokens": ["I", "don't", "like", "the", "best", "food", "😊", "!"],
  "normalized_terms": ["i", "do", "not", "like", "the", "best", "food", "😊"]
}
```

`surface_tokens` 展示分词边界；`normalized_terms` 是归一化后的术语。三种方法共用 NFC、撇号统一、小写和明确缩写展开规则；自写正则保留完整 emoji、ASCII 表情和数字，拆开连字符。保留歧义缩写与停用词，不做词干提取或词形还原。

## 最终输出

`results/` 仅保存报告使用的图表和结果表：

| 文件 | 内容 |
| --- | --- |
| `zipf.png` / `zipf.svg` | 词频—排名双对数图与拟合 |
| `heaps.png` / `heaps.svg` | 词表增长图与拟合 |
| `tokenizer_comparison.png` / `tokenizer_comparison.svg` | 不同分词配置的词表大小 |
| `tokenizer_comparison.csv` | token、type 与单次出现词数量 |
| `heaps_samples.csv` | 主种子的 10%、30%、50%、100% 采样结果 |
| `law_parameters.csv` | 两个定律的参数、拟合范围与 log-log R² |
| `tokenization_examples.csv` | 三种分词器的实际输出对比；标明真实或合成来源 |

合成案例仅用于展示和测试，不参与语料统计。运行过程的完整词频及复现信息保存在本地 `.cache/`，不提交 Git。

## 代码与测试

- `run.py`：读取语料、计数、输出实验结果。
- `statistics.py`：嵌套采样与 log-log 最小二乘拟合。
- `plots.py`：绘图。
- `boundary_cases.py`、`tests/`：边界案例与统计测试。
- `../text_processing/`：团队共用的分词实现与接口。

```bash
python -m unittest discover -s part1/tests -v
```

## 第三方库

- [regex](https://github.com/mrabarnett/mrab-regex)：Unicode 属性匹配，自写分词规则使用。
- [NLTK](https://www.nltk.org/api/nltk.tokenize.regexp.html)：Whitespace、WordPunct 基线。
- [NumPy](https://numpy.org/)：数值拟合。
- [Matplotlib](https://matplotlib.org/)：绘图。

依赖版本见本目录与根目录的 `requirements.txt`，无需下载 NLTK 模型。分词规则采用有序正则的组织方式，可对照 [NLTK TweetTokenizer](https://www.nltk.org/_modules/nltk/tokenize/casual.html)；本项目未调用它。
