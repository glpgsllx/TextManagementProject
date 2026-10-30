# 交付物（W7：作业 4.1 报告片段 + 4.2 Readme.txt）

W7 在工作项清单里列的三项，落点如下。代码本身已经在仓库里（`part1/`、`part2/`、
`text_processing/`、`tests/`、`scripts/`），这个目录只放**面向提交的文档**。

| W7 项 | 落在这 | 对应作业条款 |
|---|---|---|
| 报告：主函数代码片段 + 公式逐符号 + 设计辩护 + 第三方库引用 | `report_snippets.md` | 4.1 |
| Readme.txt 四要素（第三方库链接 / 安装与用法 / 样例输出说明） | 仓库根的 `../Readme.txt` | 4.2 |
| 重跑指南 | `../part2/README.md`、`../README.md` | 组内惯例 |

## 为什么不再有打包脚本

作业 4.2 要的是 `Src Gxx.zip`（含 `Readme.txt` + `SourceCode/`），但那是 **NTULearn
提交时**的产物，不是仓库内容。仓库里只留源码；要交 NTULearn 时从干净 checkout 现打，
两条命令（见 `../Readme.txt` 第 33–44 行）：

```bash
mkdir -p Src_Gxx/SourceCode
cp -r TextManagementProject Src_Gxx/SourceCode/ && cp TextManagementProject/Readme.txt Src_Gxx/
python -c "import shutil; shutil.make_archive('Src_Gxx', 'zip', 'Src_Gxx')"
```

## 提交前检查清单

1. 组号 `xx` 与 NTULearn 上的组号一致（打包命令里的 `Src_Gxx`）
2. `Src_Gxx.zip` 顶层只有 `Readme.txt` 与 `SourceCode/`，体积约 130 KB
   （若几 MB，多半是误收了 `part2/.cache/` 或 `data/`）
3. 报告里贴的代码片段取自 `report_snippets.md`，每段控制在 15 行内
4. 报告 References 里第三方库已列全（NLTK / regex / NumPy / Matplotlib / Lucene / Yelp）
5. 报告正文不超过 10 页，50 条逐条明细放附录

## 依赖版本口径

`requirements.txt` 钉的是 nltk 3.10.4 / regex 2026.9.29（Part 1 进仓库时定的），
本次不动它。报告里的数字是用 nltk 3.9.4 / regex 2026.5.9 跑出来的：

- **nltk 只被 Part 1 的参考分词器用**（`text_processing/api.py` 的 Whitespace /
  WordPunct 对比）。Part 2 全程走我们自己的 custom tokenizer，**完全不加载 nltk**，
  所以 nltk 版本与 Part 2 无关系。
- **regex 两侧都用**（custom tokenizer 的正则）。2026.5.9 → 2026.9.29 是同年小版本，
  我们只用基础语法 + Unicode property escapes，改 token 产出的可能性极低。

即：助教按 `requirements.txt` 装完再跑 Part 2，数字应当一致；Part 1 的分词器对比表
若出现小数位差异，属 NLTK 版本差异，不是代码问题。
