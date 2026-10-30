"""Lucene 标准库对照实验：真·Apache Lucene 10.5.2 vs 我们手搓的两个实现。

作业 3.3.2 最后一句：

    "You may also run the same queries and compare your results with those
     returned by a standard library such as Lucene, e.g., the built-in
     ranking modules."

这一句是"可选加分项"，所以本模块不改动 `query_experiment.py` 的任何既有实验组，
而是**新增**一组 E8–E11，把它们作为第三方参照列。

跑的四个 Lucene 配置（两份索引 × 两个相似度）
---------------------------------------------
    ours_tok   用我们自己分词器切好的 token 喂给 Lucene（WhitespaceAnalyzer 只按
               空白切开、不再做任何处理）。两侧 token 序列逐条一致，于是剩下的差异
               纯粹是**打分实现**：idf 公式、长度归一、norm 量化、浮点精度。
    std        StandardAnalyzer 处理原文，Lucene 开箱即用的方式。与 ours_tok 的差
               就是**分词器**的贡献（含英文停用词表）。

    每个索引上跑 BM25Similarity(1.2, 0.75) 与 ClassicSimilarity（Lucene 内置 TF-IDF）。

为什么要把分词这一层单独拆出来
------------------------------
之前 `data/lucene_probe` 只跑 6 条查询、且用 StandardAnalyzer 重新分词，
"公式差异"和"分词差异"混在一起，看到 4/10 的重合说不清是哪一边的锅。这里让两侧
token 完全一致，于是"手搓 vs Lucene"这个数可以直接归因到公式实现上。

怎么跑
------
    python -m part2.lucene_experiment                  # 全量：导出语料 → 编译/跑 Java → 对比
    python -m part2.lucene_experiment --skip-run       # Java 已经跑过，只重算对比
    python -m part2.lucene_experiment --queries Q01,Q45

需要 JDK（自动找 `C:\\Program Files\\Java\\jdk-*`）与 Lucene 10.5.2 的三个 jar
（默认 `C:/tmp/lucene-probe/lib`）。产出写在 `part2/results/lucene_comparison.*`。
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

from .bm25 import COURSE, DEFAULT_B, DEFAULT_K1, LUCENE, OKAPI, bm25_search
from .inverted_index import load_index
from .queries import QUERIES
from .rank_metrics import compare_pair
from .tfidf import LNC_LTC, tfidf_search

ROOT = Path(__file__).resolve().parents[1]
JAVA_SRC = Path(__file__).resolve().parent / "lucene" / "LuceneCompare.java"

POOL = 50          # 深池，与 query_experiment.py 一致
KS = (5, 10, 20, 50)
LUCENE_TOPK = 200  # 向 Lucene 多要一些，重排 tiebreak 后再取前 POOL 条
EXPLAIN_DEPTH = 10  # 每条查询取前 10 篇的 explain（拿回 Lucene 实际用的 dl）

OUR_CONDITIONS = {
    "tfidf_lnc_ltc": lambda index, text, k: tfidf_search(index, text, top_k=k,
                                                         variant=LNC_LTC),
    "bm25_okapi": lambda index, text, k: bm25_search(index, text, top_k=k,
                                                    variant=OKAPI),
    "bm25_course": lambda index, text, k: bm25_search(index, text, top_k=k,
                                                     variant=COURSE),
    "bm25_lucene": lambda index, text, k: bm25_search(index, text, top_k=k,
                                                     variant=LUCENE),
}

# E8–E11：每组差异同样收敛在一处
GROUP_PAIRS = [
    ("E8", "bm25_lucene", "lucene_bm25_ours_tok",
     "手搓的 Lucene idf 复刻 vs 真 Lucene BM25：同 token、同 k1/b，只看复刻忠不忠实"),
    ("E9", "bm25_okapi", "lucene_bm25_ours_tok",
     "交付口径 BM25（Okapi 钳制 idf + (k1+1)）vs 真 Lucene BM25：同 token，差在 idf 与常数"),
    ("E10", "tfidf_lnc_ltc", "lucene_classic_ours_tok",
     "我们的 TF-IDF 余弦 lnc.ltc vs Lucene 内置 ClassicSimilarity：同 token，两套 formula"),
    ("E11", "lucene_bm25_ours_tok", "lucene_bm25_std",
     "同为 Lucene BM25，只换分词器（我们的 custom vs Lucene StandardAnalyzer）"),
]


# ---------------------------------------------------------------- 环境

def find_java_home():
    """找 JDK：先看 JAVA_HOME，再扫 Program Files 下的 jdk-*。"""
    import os
    env = os.environ.get("JAVA_HOME")
    if env and (Path(env) / "bin" / "javac.exe").exists():
        return Path(env)
    for base in (Path("C:/Program Files/Java"), Path("D:/Program Files/Java")):
        if not base.exists():
            continue
        candidates = sorted([p for p in base.iterdir()
                             if p.is_dir() and p.name.startswith("jdk")],
                            reverse=True)
        for candidate in candidates:
            if (candidate / "bin" / "javac.exe").exists():
                return candidate
    return None


def find_lucene_lib():
    for candidate in (Path("C:/tmp/lucene-probe/lib"),
                      Path("C:/tmp/lucene-probe"),
                      ROOT / "part2/lucene/lib"):
        if candidate.exists() and list(candidate.glob("lucene-core-*.jar")):
            return candidate
    return None


# ---------------------------------------------------------------- 导出语料

def export_corpus(input_dir: Path, work_dir: Path, tokenizer_name: str, verbose=True):
    """导出 ids.txt / texts_tok.txt / texts_raw.txt。

    texts_tok.txt 里的 token 必须和索引里的 token **逐条一致**，否则"同 token"这个
    前提就不成立，后面所有差异都无法归因。所以用建索引时那一个分词器重切一遍。
    """
    from text_processing import get_tokenizer

    work_dir.mkdir(parents=True, exist_ok=True)
    ids_path = work_dir / "ids.txt"
    tok_path = work_dir / "texts_tok.txt"
    raw_path = work_dir / "texts_raw.txt"
    if ids_path.exists() and tok_path.exists() and raw_path.exists():
        if verbose:
            print(f"语料已导出，跳过：{work_dir}")
        return ids_path, tok_path, raw_path

    tokenizer = get_tokenizer(tokenizer_name)
    reviews_path = input_dir / "review.json"
    started = time.perf_counter()
    written = 0
    with (open(ids_path, "w", encoding="utf-8", newline="\n") as ids_out,
          open(tok_path, "w", encoding="utf-8", newline="\n") as tok_out,
          open(raw_path, "w", encoding="utf-8", newline="\n") as raw_out):
        for line in open(reviews_path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record["text"]
            tokens = tokenizer.tokenize(text)
            # 换行/制表符压成空格：Lucene 侧一行一篇，格式必须干净
            flat = re.sub(r"[\r\n\t]+", " ", text)
            tokens_flat = [re.sub(r"[\r\n\t]+", " ", t) for t in tokens]
            tokens_flat = [t for t in tokens_flat if t and not t.isspace()]
            ids_out.write(record["review_id"] + "\n")
            tok_out.write(" ".join(tokens_flat) + "\n")
            raw_out.write(flat + "\n")
            written += 1
    if verbose:
        print(f"导出 {written} 篇到 {work_dir}，耗时 {time.perf_counter() - started:.1f}s")
    return ids_path, tok_path, raw_path


def export_queries(work_dir: Path, queries, tokenizer_name: str):
    """queries.tsv：qid \\t 原始查询 \\t 我们的 token。"""
    from text_processing import get_tokenizer

    tokenizer = get_tokenizer(tokenizer_name)
    path = work_dir / "queries.tsv"
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for query in queries:
            tokens = [t for t in tokenizer.tokenize(query["text"])
                      if t and not t.isspace()]
            handle.write(f"{query['id']}\t{query['text']}\t{' '.join(tokens)}\n")
    return path


# ---------------------------------------------------------------- 跑 Java

def run_java(work_dir: Path, out_path: Path, java_home: Path, lucene_lib: Path,
             verbose=True):
    classes = work_dir / "classes"
    classes.mkdir(parents=True, exist_ok=True)
    javac = java_home / "bin" / "javac.exe"
    java = java_home / "bin" / "java.exe"
    classpath = str(lucene_lib / "*")
    classpath = classpath.replace("\\", "/")

    if verbose:
        print(f"编译 {JAVA_SRC.name} ...")
    subprocess.run(
        [str(javac), "-encoding", "UTF-8", "-nowarn", "-cp", classpath,
         "-d", str(classes), str(JAVA_SRC)],
        check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")

    if verbose:
        print("建索引并检索（这一步最慢，约 1–3 分钟）...")
    started = time.perf_counter()
    result = subprocess.run(
        [str(java), "-Xmx4g", "-Dfile.encoding=UTF-8", "-Dstdout.encoding=UTF-8",
         "-cp", f"{classpath};{classes}",
         "LuceneCompare", str(work_dir), str(out_path), str(LUCENE_TOPK),
         str(EXPLAIN_DEPTH)],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        sys.stderr.write(result.stdout[-4000:] + "\n" + result.stderr[-4000:] + "\n")
        raise SystemExit(f"Lucene 探针失败，退出码 {result.returncode}")
    if verbose:
        print(f"Lucene 跑完，耗时 {time.perf_counter() - started:.1f}s")
    return out_path


# ---------------------------------------------------------------- 读回与对齐

def load_lucene(path: Path):
    """读 Lucene 产物 → {配置名: {查询 id: [(doc_id, score), ...]}}，并按我们的
    tiebreak（分数降序、同分按 doc_id 升序）重排。

    为什么要重排：Lucene 的同分 tiebreak 是内部 doc id（= 插入顺序），我们的是
    review_id 字符串升序。不统一的话，"同分块内部的顺序"会被当成排名差异算进 τ。
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    runs = {cfg: {} for cfg in
            ["lucene_bm25_ours_tok", "lucene_classic_ours_tok",
             "lucene_bm25_std", "lucene_classic_std"]}
    meta = {"topK": payload["topK"], "indexes": payload["indexes"],
            "lucene": payload["lucene"], "java": payload.get("java")}
    for query in payload["queries"]:
        qid = query["id"]
        for cfg in runs:
            block = query["results"].get(cfg, {})
            ranked = [(item["id"], item["score"]) for item in block.get("top", [])]
            ranked.sort(key=lambda item: (-item[1], item[0]))
            runs[cfg][qid] = {"ranked": ranked, "totalHits": block.get("totalHits", 0),
                              "explain": block.get("explain", [])}
    return runs, meta


def lucene_dl_from_explain(explain):
    """从 explain 原文里抠出 Lucene 打分时实际用的 dl（它标注为 approximate）。"""
    match = re.search(r"([0-9]+(?:\.0)?) = dl, length of field", explain or "")
    if not match:
        return None
    return float(match.group(1))


def score_check(index, lucene_runs, queries, verbose=True):
    """分数级校验：用我们索引里的 tf / df / avgdl 复算 Lucene 的 BM25 分数。

    这是"手搓能不能被 Lucene 顶替"最硬的一条证据，分三步归因，前一步的残差就是
    后一步要解释的东西：

        第 1 步  用**我们算的真实 dl** 复算，得到残差 A。
        第 2 步  用 **Lucene explain 里报的 dl** 复算，得到残差 B。
        第 3 步  残差 B 应当只剩浮点误差（Lucene 全程 float32，我们全程 float64），
                 量级 1e-7；如果 B ≈ 0，说明 idf、tf 饱和、长度归一、N、avgdl
                 这一整套公式我们理解得和 Lucene **逐项一致**。
        残差 A − 残差 B = Lucene 把文档长度量化（norm 里存的是近似长度）带来的损失。

    顺带把"dl 量化损失"本身也统计出来：Lucene 用的 dl 与真实 dl 的相对差。
    """
    import math
    from text_processing import get_tokenizer

    tokenizer = get_tokenizer(index.tokenizer)
    n_docs = len(index.doc_lengths)
    rows = []
    duplicate_checks = []
    for query in queries:
        qid = query["id"]
        tokenizer = get_tokenizer(index.tokenizer)
        tokens_all = tokenizer.tokenize(query["text"])
        terms = set(tokens_all)
        per_term = []
        for term in terms:
            plist = index.postings.get(term)
            if not plist:
                continue
            df = len(plist)
            per_term.append((term, plist, math.log(1 + (n_docs - df + 0.5) / (df + 0.5)),
                             tokens_all.count(term)))
        if not per_term:
            continue
        has_duplicate = len(tokens_all) != len(terms)
        block = lucene_runs["lucene_bm25_ours_tok"][qid]
        for position, (doc_id, actual) in enumerate(block["ranked"][:10]):
            dl = index.doc_lengths[doc_id]
            explains = block.get("explain", [])
            dl_lucene = (lucene_dl_from_explain(explains[position])
                         if position < len(explains) else None)

            def score_with(length, weighted=False):
                total = 0.0
                for _term, plist, idf, multiplicity in per_term:
                    tf = plist.get(doc_id)
                    if not tf:
                        continue
                    total += idf * tf / (tf + DEFAULT_K1 * (
                        1 - DEFAULT_B + DEFAULT_B * length / index.avg_doc_length)) \
                        * (multiplicity if weighted else 1)
                return total

            exact = score_with(dl)
            as_lucene = score_with(dl_lucene) if dl_lucene else None
            rows.append({
                "query": qid, "doc": doc_id, "lucene": actual,
                "dl_ours": dl, "dl_lucene": dl_lucene,
                "pred_dl_exact": exact, "pred_dl_lucene": as_lucene,
                "rel_err_exact": abs(exact - actual) / actual if actual else None,
                "rel_err_lucene_dl": (abs(as_lucene - actual) / actual
                                      if as_lucene is not None and actual else None),
                "dl_rel_loss": (abs(dl - dl_lucene) / dl
                                if dl_lucene is not None and dl else None),
            })
            if has_duplicate and position == 0 and as_lucene is not None:
                # 重复词查询：把"不去重"的分数也算出来，看是不是正好翻倍
                weighted = score_with(dl_lucene, weighted=True)
                duplicate_checks.append({
                    "query": qid, "text": query["text"], "doc": doc_id,
                    "tokens": tokens_all,
                    "score_dedup": as_lucene, "score_with_duplicates": weighted,
                    "lucene_actual": actual,
                    "ratio": weighted / as_lucene if as_lucene else None})

    def summarize(key):
        values = [row[key] for row in rows if row[key] is not None]
        if not values:
            return None
        values = sorted(values)
        return {"n": len(values), "median": round(statistics.median(values), 10),
                "p90": round(values[int(len(values) * 0.9)], 10),
                "max": round(values[-1], 10)}

    out = {"n_compared": len(rows),
           "rel_err_dl_exact": summarize("rel_err_exact"),
           "rel_err_dl_lucene_actual": summarize("rel_err_lucene_dl"),
           "dl_quantization_loss": summarize("dl_rel_loss"),
           "duplicate_term_queries": duplicate_checks,
           "sample": rows[:8]}
    if verbose:
        print(f"分数级校验：比对了 {len(rows)} 个 (查询, 文档) 对")
        print(f"  残差 A（用真实 dl 复算）      ：{out['rel_err_dl_exact']}")
        print(f"  残差 B（用 Lucene 实际 dl 复算）：{out['rel_err_dl_lucene_actual']}")
        print(f"  dl 量化相对损失                ：{out['dl_quantization_loss']}")
    return out


# ---------------------------------------------------------------- 指标

def compute_groups(index, our_runs, lucene_runs, queries, ks=KS, pool=POOL):
    """把 E8–E11 的逐条指标算出来。我们的条件是完整排序，Lucene 是 top-K。"""
    groups = {group: {"pair": [name_a, name_b], "note": note, "per_query": {}}
              for group, name_a, name_b, note in GROUP_PAIRS}
    for query in queries:
        qid = query["id"]
        for group, name_a, name_b, _note in GROUP_PAIRS:
            ranked_a = (our_runs[name_a][qid] if name_a in our_runs
                        else lucene_runs[name_a][qid]["ranked"])
            ranked_b = (our_runs[name_b][qid] if name_b in our_runs
                        else lucene_runs[name_b][qid]["ranked"])
            metrics = compare_pair(ranked_a, ranked_b, ks=ks, pool=pool)
            metrics["axis"] = query["axis"]
            metrics["text"] = query["text"]
            metrics["df_band"] = query.get("df_band", "")
            groups[group]["per_query"][qid] = metrics
    return groups


def _mean(values):
    values = [v for v in values if v is not None]
    return round(statistics.fmean(values), 4) if values else None


def aggregate(group, queries, ks=KS):
    per_query = group["per_query"]
    rows = []

    def summarize(label, ids):
        subset = [per_query[qid] for qid in ids if qid in per_query]
        if not subset:
            return None
        row = {"slice": label, "n_queries": len(subset)}
        for k in ks:
            row[f"overlap@{k}"] = _mean([m["overlap"][f"@{k}"] for m in subset])
        row["tau_union"] = _mean([m["tau_b_union"]["tau"] if m["tau_b_union"] else None
                                  for m in subset])
        row["tau_intersection"] = _mean(
            [m["tau_b_intersection"]["tau"] if m["tau_b_intersection"] else None
             for m in subset])
        row["top1_same_rate"] = _mean([1.0 if m["top1_same"] else 0.0 for m in subset])
        rows.append(row)

    summarize("全部查询", [q["id"] for q in queries])
    axes = {}
    for query in queries:
        axes.setdefault(query["axis"], []).append(query["id"])
    for axis in sorted(axes):
        summarize(f"axis={axis}", axes[axis])
    return rows


def write_markdown(path, meta, groups, aggregates, checks, index_meta):
    lines = ["# Lucene 标准库对照（作业 3.3.2 的 \"compare with Lucene\"）", "",
             f"- Lucene {meta['lucene']} / Java {meta['java']}，向每个配置要前 "
             f"{meta['topK']} 条后按我们的 tiebreak 重排",
             f"- 我们的实现：{index_meta['reviews']} 篇，分词器 {index_meta['tokenizer']}，"
             f"avgdl={index_meta['avgdl']}", ""]

    lines += ["## 1. 两份 Lucene 索引的统计", "",
              "| 索引 | analyzer | numDocs | sumTotalTermFreq | avgdl |",
              "|---|---|---:|---:|---:|"]
    for name, stats in meta["indexes"].items():
        lines.append(f"| `{name}` | {stats['analyzer']} | {stats['numDocs']} | "
                     f"{stats['sumTotalTermFreq']} | {round(stats['avgdl'], 4)} |")
    lines.append("")

    lines += ["## 2. 实验组与聚合结果", ""]
    for group, rows in aggregates.items():
        payload = groups[group]
        lines += [f"### {group}：`{payload['pair'][0]}` vs `{payload['pair'][1]}`", "",
                  f"差异只在：{payload['note']}", "",
                  "| 切片 | 查询数 | " + " | ".join(f"overlap@{k}" for k in KS) +
                  " | τ 并集 | τ 交集 | 首条相同率 |",
                  "|---|---:|" + "---:|" * (len(KS) + 3)]
        for row in rows:
            cells = [str(row[f"overlap@{k}"]) for k in KS]
            lines.append(f"| {row['slice']} | {row['n_queries']} | " +
                         " | ".join(cells) +
                         f" | {row['tau_union']} | {row['tau_intersection']} | "
                         f"{row['top1_same_rate']} |")
        lines.append("")

    lines += ["## 3. 分数级校验（BM25，ours_tok 索引，每条查询的前 10 篇）", "",
              "用我们索引里的 tf / df / avgdl 复算 Lucene 的 BM25 分数，与 Lucene 实际返回的分数比。",
              "两步归因：残差 A 用我们算的真实 dl，残差 B 用 Lucene explain 里报的 dl；",
              "A − B 就是 Lucene 量化文档长度造成的损失。", "",
              "| 量 | 比对数 | 中位数 | p90 | 最大 |",
              "|---|---:|---:|---:|---:|"]
    for label, key in [("残差 A：用真实 dl 复算", "rel_err_dl_exact"),
                       ("残差 B：用 Lucene 实际 dl 复算", "rel_err_dl_lucene_actual"),
                       ("dl 本身被量化的相对损失", "dl_quantization_loss")]:
        stats = checks[key]
        if stats:
            lines.append(f"| {label} | {stats['n']} | {stats['median']} | "
                         f"{stats['p90']} | {stats['max']} |")
    lines.append("")
    lines += ["样本（前 8 个）：", "",
              "| 查询 | Lucene 分数 | 复算（真实 dl） | 复算（Lucene 的 dl） | 真实 dl | Lucene 的 dl |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in checks["sample"]:
        dl_luc = row["dl_lucene"] if row["dl_lucene"] is not None else "n/a"
        pred_luc = (f"{row['pred_dl_lucene']:.6f}"
                    if row["pred_dl_lucene"] is not None else "n/a")
        lines.append(f"| {row['query']} | {row['lucene']:.6f} | "
                     f"{row['pred_dl_exact']:.6f} | {pred_luc} | "
                     f"{row['dl_ours']} | {dl_luc} |")
    lines.append("")

    if checks.get("duplicate_term_queries"):
        lines += ["### 3.1 已知语义差异：重复查询词", "",
                  "教科书 BM25 写的是 Σ_{t∈Q}，查询被当成**词项的集合**，所以 "
                  "`pizza pizza` 与 `pizza` 是同一个信息需求，我们的实现去重。",
                  "Lucene 的 BooleanQuery **不去重**：同一个 TermQuery 加两次 SHOULD，",
                  "分数正好翻倍。下表是实测（top-1 文档）：", "",
                  "| 查询 | token | 去重分数 | 不去重分数 | 不去重 / 去重 |",
                  "|---|---|---:|---:|---:|"]
        for row in checks["duplicate_term_queries"]:
            lines.append(f"| {row['query']} `{row['text']}` | {row['tokens']} | "
                         f"{row['score_dedup']:.6f} | {row['score_with_duplicates']:.6f} | "
                         f"{row['ratio']:.4f} |")
        lines.append("")
        lines += ["对照用的 Lucene 结果已在 Java 侧显式去重（两边查询语义一致），",
                  "所以这一条不会污染上面的 overlap / τ 读数，但它本身是个要写进报告的",
                  "实现差异。", ""]

    lines += ["## 4. E9（交付 BM25 vs 真 Lucene BM25）逐条明细", "",
              "| 查询 | 文本 | 轴 | overlap@10 | overlap@50 | τ 并集 | 首条相同 | "
              "BM25 未召回 | Lucene 未召回 |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for qid, metrics in groups["E9"]["per_query"].items():
        tau = metrics["tau_b_union"]["tau"] if metrics["tau_b_union"] else None
        tau_text = f"{tau:.4f}" if tau is not None else "n/a"
        lines.append(f"| {qid} | {metrics['text']} | {metrics['axis']} | "
                     f"{metrics['overlap']['@10']} | {metrics['overlap']['@50']} | "
                     f"{tau_text} | {'是' if metrics['top1_same'] else '否'} | "
                     f"{metrics['n_missing_a']} | {metrics['n_missing_b']} |")
    lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- main

def load_index_from_disk(index_file, input_dir, tokenizer):
    reviews_path = input_dir / "review.json"
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    source = {"file": reviews_path.name, "bytes": reviews_path.stat().st_size,
              "reviews": manifest["files"]["review.json"]["records"],
              "tokenizer": tokenizer}
    return load_index(index_file, source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/meridian_id")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "part2/results")
    parser.add_argument("--index-file", type=Path, default=ROOT / "part2/.cache/index.pkl")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "part2/.cache/lucene")
    parser.add_argument("--tokenizer", default="custom")
    parser.add_argument("--lucene-lib", type=Path, default=None)
    parser.add_argument("--java-home", type=Path, default=None)
    parser.add_argument("--queries", default="", help="只跑指定查询 id，逗号分隔")
    parser.add_argument("--skip-run", action="store_true", help="不重跑 Java，只用已有产物")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    verbose = not args.quiet

    queries = QUERIES
    if args.queries:
        wanted = {q.strip() for q in args.queries.split(",") if q.strip()}
        queries = [q for q in queries if q["id"] in wanted]

    java_home = args.java_home or find_java_home()
    lucene_lib = args.lucene_lib or find_lucene_lib()
    if java_home is None:
        raise SystemExit("找不到 JDK，用 --java-home 指定")
    if lucene_lib is None:
        raise SystemExit("找不到 Lucene jar，用 --lucene-lib 指定（需 lucene-core / "
                         "-analysis-common / -queryparser 三个 jar）")
    if verbose:
        print(f"JDK: {java_home}\nLucene lib: {lucene_lib}")

    index = load_index_from_disk(args.index_file, args.input_dir, args.tokenizer)
    for query in queries:
        from .queries import profile_query
        query["df_band"] = profile_query(index, query["text"])["df_band"]

    export_corpus(args.input_dir, args.work_dir, args.tokenizer, verbose)
    export_queries(args.work_dir, queries, args.tokenizer)

    out_json = args.work_dir / "lucene_results.json"
    if not args.skip_run:
        run_java(args.work_dir, out_json, java_home, lucene_lib, verbose)
    lucene_runs, lucene_meta = load_lucene(out_json)

    # 我们的四个条件：完整排序（n_docs 条），保证 τ 有真实名次
    n_docs = len(index.doc_lengths)
    our_runs = {name: {} for name in OUR_CONDITIONS}
    started = time.perf_counter()
    for query in queries:
        for name, runner in OUR_CONDITIONS.items():
            our_runs[name][query["id"]] = runner(index, query["text"], n_docs)
    if verbose:
        print(f"我们的四个条件跑完，耗时 {time.perf_counter() - started:.1f}s")

    groups = compute_groups(index, our_runs, lucene_runs, queries)
    aggregates = {group: aggregate(groups[group], queries) for group in groups}
    checks = score_check(index, lucene_runs, queries, verbose)

    index_meta = {"reviews": len(index.doc_lengths), "tokenizer": index.tokenizer,
                  "avgdl": round(index.avg_doc_length, 4)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "lucene_comparison.json").write_text(
        json.dumps({"meta": lucene_meta, "index": index_meta,
                    "groups": {g: {"pair": groups[g]["pair"], "note": groups[g]["note"],
                                   "aggregate": aggregates[g],
                                   "per_query": groups[g]["per_query"]}
                               for g in groups},
                    "score_check": checks}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8")
    write_markdown(args.output_dir / "lucene_comparison.md", lucene_meta, groups,
                   aggregates, checks, index_meta)

    if verbose:
        for group in ["E8", "E9", "E10", "E11"]:
            row = aggregates[group][0]
            print(f"{group} {groups[group]['pair']}: overlap@10={row['overlap@10']} "
                  f"overlap@50={row['overlap@50']} tau_union={row['tau_union']} "
                  f"top1_same={row['top1_same_rate']}")
        print(f"产出写到 {args.output_dir}")


if __name__ == "__main__":
    main()
