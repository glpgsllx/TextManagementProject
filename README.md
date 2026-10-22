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

需要做归一化时，调用 `tokenizer.tokenize(text)`；查看原始分词边界时，调用 `tokenizer.surface_tokens(text)`。缩写展开默认开启，可以通过 `get_tokenizer("custom", expand_contractions=False)` 单独关闭。

做搜索时，评论和查询要用同一种分词器、同一套设置；换设置后需要重新建索引。做情感分析时，可以直接读取原始评论，保留标点和大小写。

## 文件放在哪里

- `text_processing/`：大家共用的分词器。
- `data/meridian_id/`：全组使用的数据。
- `part1/`：Part 1 的代码、运行说明和最终图表。
- `tests/`：公共分词接口的测试。
- `scripts/prepare_city_subset.py`：从原始 Yelp 数据重新提取城市子集的脚本，已有数据时不用运行。

后续部分的代码和结果放到各自目录里。公共接口的测试命令是：

```bash
python -m unittest tests.test_text_processing -v
```
