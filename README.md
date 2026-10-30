# AI6122 Text Management Project

这个仓库放小组作业的代码。我们统一使用 Meridian, Idaho（ID）的 Yelp 数据，共 1,045 家商户、37,401 条评论。

数据已经放在 `data/meridian_id/` 里，拉取仓库后就有。`business.json` 是商户信息，`review.json` 是评论，两者通过 `business_id` 关联。文件都是 JSON Lines 格式，每行一条记录。`manifest.json` 保存文件数量和校验值。

## 先装环境

使用 Python 3.11，在仓库根目录执行：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

根目录的依赖是公共分词器需要的。运行 Part 1 时，再按 [part1/README.md](part1/README.md) 安装它自己的统计和绘图依赖。

### Windows 上先确认语料没被改过换行符

Windows 版 Git 默认开着 `core.autocrlf=true`，克隆时会把语料的换行符从 LF 改成 CRLF，
每条评论多一个字节，`data/meridian_id/` 下两个 JSON 的字节数和 sha256 就与
`manifest.json` 对不上了——Part 1 读语料时会直接报
`Corpus checksum/count differs from export manifest`（分词统计不受影响，坏的是校验值）。

仓库里的 `.gitattributes` 已经把这两个语料文件标成 `-text`，Git 不会再动它们的换行符。
克隆后请核对一次：

```bash
sha256sum data/meridian_id/review.json
# 应等于 data/meridian_id/manifest.json 里记的 e6588358fc7398dc42fa45bc2504b31785d892f8720770bd580faf7899d8f32f
```

对不上就在仓库里执行 `git config core.autocrlf false` 后重新取回这两个文件。

## 其他部分怎么用分词器

直接从 `text_processing` 导入：

```python
from text_processing import get_tokenizer

tokenizer = get_tokenizer("custom")
terms = tokenizer.tokenize("I don't like the-best-food 😊!")
print(terms)
# ['i', 'do', 'not', 'like', 'the', 'best', 'food', '😊']
```

`custom` 是自写正则分词器，也可以换成 `whitespace` 或 `wordpunct`。

需要做归一化时，调用 `tokenizer.tokenize(text)`；查看原始分词边界时，调用 `tokenizer.surface_tokens(text)`。归一化包括 Unicode 等价写法和撇号统一、小写转换、明确缩写的展开及纯标点过滤。

做搜索时，评论和查询要用同一种分词器；换方法后需要重新建索引。做情感分析时，可以直接读取原始评论，保留标点和大小写。

## 文件放在哪里

- `text_processing/`：大家共用的分词器。
- `data/meridian_id/`：全组使用的数据。
- `part1/`：Part 1 的代码、运行说明和最终图表。
- `part2/`：Part 2 的倒排索引、两个打分函数、50 条查询的实验及运行说明和结果。
  两项分析分开跑：**逐实现分析**（`part2/impl_analysis.py`，作业 3.3.2 的字面要求：
  两个实现各自跑一批 diverse query 并给出分析）与**分歧机制分析**
  （`part2/query_experiment.py`，附带，解释两个实现为什么分歧）。
- `part2/lucene/`：真·Apache Lucene 10.5.2 对照探针（Java），由 `part2/lucene_experiment.py`
  驱动，产出 E8–E11 四个对照实验组。需要 JDK 与三个 Lucene jar，见 part2 README。
- `tests/`：公共分词接口的测试。
- `scripts/prepare_city_subset.py`：从原始 Yelp 数据重新提取城市子集的脚本，已有数据时不用运行。

后续部分的代码和结果放到各自目录里。公共接口的测试命令是：

```bash
python -m unittest tests.test_text_processing -v
```
