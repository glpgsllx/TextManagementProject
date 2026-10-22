# AI6122 Text Management Project

团队共用的 Yelp 项目。全组使用 Meridian, Idaho（ID）的 **1,045 家商户、37,401 条评论**。

## 团队入口

| 目录 | 职责 |
| --- | --- |
| `text_processing/` | 共用分词接口：custom、whitespace、wordpunct |
| `data/meridian_id/` | 全组共用的数据与校验清单；本地生成，不提交 Git |
| `scripts/` | 城市子集提取 |
| `tests/` | 公共接口测试 |
| `part1/` | Part 1 的代码、实验依赖、图表、结果和说明 |

各部分的实现和产物放在各自目录。Part 1 的运行方式、规则和实验结果见 [part1/README.md](part1/README.md)。

## 使用公共分词接口

在仓库根目录执行，使用 Python 3.11：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

```python
from text_processing import get_tokenizer

# 可选名称：custom、whitespace、wordpunct。
tokenizer = get_tokenizer("custom", expand_contractions=True)
print(tokenizer.tokenize("I don't like the-best-food 😊!"))
# ['i', 'do', 'not', 'like', 'the', 'best', 'food', '😊']

# 查看归一化前的边界。
print(tokenizer.surface_tokens("Don't go!"))
# ["Don't", 'go', '!']
```

检索模块的文档与查询应使用同一方法和配置；更改配置后重新建索引。情感分析按需要保留原文中的大小写、标点和上下文。

公共接口测试：`python -m unittest tests.test_text_processing -v`。无需下载 NLTK 模型。

## 准备共享数据

将 `yelp_academic_dataset_business.json` 和 `yelp_academic_dataset_review.json` 放入源目录，再执行：

```bash
python scripts/prepare_city_subset.py --city Meridian --state ID \
  --source-dir yelp-dataset/versions/4 --output-dir data/meridian_id
```

脚本保留原字段和文本；输出目录需尚未存在。已有子集可直接使用。筛选方法、数量和文件哈希见 `data/meridian_id/manifest.json`。
