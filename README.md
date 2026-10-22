# AI6122 Text Management Project

团队共用的 Yelp 评论项目。全组使用 **Meridian, Idaho（州代码 ID）** 的 1,045 家商户、37,401 条评论；Part 1 已实现分词器、词频统计、Zipf 与 Heaps 实验。

## 开始运行

在仓库根目录执行，使用 Python 3.11：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m part1 --text "I don't like the-best-food 😊!"
```

依赖版本保存在 `requirements.txt`，无需下载 NLTK 模型。

## 其他组员怎样调用分词器

公共接口在 `text_processing`，不依赖 Part 1 的统计流程：

```python
from text_processing import available_tokenizers, get_tokenizer, tokenize

print(available_tokenizers())  # ('custom', 'whitespace', 'wordpunct')
print(tokenize("I don't like the-best-food 😊!"))
# ['i', 'do', 'not', 'like', 'the', 'best', 'food', '😊']

# 一个实例可以反复用于评论和查询。
tokenizer = get_tokenizer("custom", expand_contractions=True)
document_terms = tokenizer.tokenize("The best-food is here!")
query_terms = tokenizer.tokenize("best food")

# 检查原始分词边界；保留大小写和标点，不做缩写展开。
print(tokenizer.surface_tokens("Don't go!"))
# ["Don't", 'go', '!']
```

- **Part 2 检索：** 文档与查询使用相同的方法和 `expand_contractions` 配置；修改配置后重新构建索引。
- **Part 3 等后续模块：** 保留原始 `review["text"]`。情感分析可能需要大小写、标点和上下文，按任务选择原文或术语流。
- **新增分词方法：** 实现一个 `text -> list[str]` 的边界函数，在 `text_processing/api.py` 的 `_SURFACE_TOKENIZERS` 注册，并补充输入输出测试。现有公共接口和统一归一化可以继续复用。

## 目录职责

| 路径 | 内容 |
| --- | --- |
| `text_processing/custom.py` | 自写的有序正则边界规则 |
| `text_processing/normalization.py` | 共用的大小写、缩写、标点处理 |
| `text_processing/api.py` | 三种分词方法的统一调用接口 |
| `part1/` | Part 1 实验、统计、命令行入口和测试 |
| `part1/plots.py` / `part1/boundary_cases.py` | 实验绘图与独立的合成边界案例 |
| `tests/` | 团队公共分词接口测试 |
| `part1/results/` | 可复查的统计表、参数、示例和图 |
| `scripts/prepare_city_subset.py` | 从原始 Yelp 文件提取城市子集 |
| `data/meridian_id/` | 全组共用的两个 JSONL 数据文件与清单 |

具体规则、实验结果和课件对应见 [Part 1 说明](part1/README.md)。

## 准备数据

Yelp 原始文件和导出的业务/评论 JSONL **不提交到 Git**。每位组员从同一份 Yelp 数据下载或取得文件后，在仓库根目录运行：

```bash
python scripts/prepare_city_subset.py \
  --city Meridian --state ID \
  --source-dir yelp-dataset/versions/4 \
  --output-dir data/meridian_id
```

源目录内需要 `yelp_academic_dataset_business.json` 和 `yelp_academic_dataset_review.json`。脚本先选城市和州对应的商户，再按 `business_id` 提取全部相关评论；保留原字段、文本和记录顺序。输出目录需尚未存在，已有子集可直接使用。

`data/meridian_id/manifest.json` 记录行数、SHA-256 和筛选方法。这里的“100% 数据”指 Meridian 子集全部评论。

## 运行与测试

```bash
python -m part1.run
python -m unittest discover -v
```

全量实验读取 `data/meridian_id/`，结果写入 `part1/results/`。团队开发时把共用文本规则放入 `text_processing`，各 Part 的业务逻辑保留在自己的模块中。
