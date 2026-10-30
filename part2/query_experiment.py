"""查询实验 runner：50 条查询 × 若干检索条件 → 落 JSON / CSV / Markdown。

跑法（仓库根目录，索引已建好）：

    python -m part2.query_experiment                 # 全量：50 条 × 6 个基础条件 + 16 组参数网格
    python -m part2.query_experiment --skip-grid     # 只跑基础条件
    python -m part2.query_experiment --queries Q01,Q19,Q46   # 只跑指定几条
    python -m part2.query_experiment --axes edge     # 只跑某个轴

产出（默认写在 part2/results/）：
    query_experiment_runs.json       原始产物：每个条件每条查询的 top-50 与耗时
    query_experiment_summary.json    指标：逐条 + 聚合（按轴、按 df 分档）
    query_experiment_summary.md      给人看的表格
    query_experiment_overlap.csv     长表，画图用

本脚本是**附带分析**，不是作业的字面要求。作业 3.3.2 写的是 "Run a set of at least
five diverse queries on **both implementations**"——"两个实现各自跑"和"两个实现拉到一起
比"是两件事，前者由 impl_analysis.py 承担（逐实现画像，不需要 qrels 就能判断实现行为
对不对）。本脚本做的是后者：解释两个实现**为什么分歧、分歧由什么变量控制**。

它测的是排序一致性，**没有 qrels 就判不了谁更好**——这是它的天花板，不是它的缺点。

实验组（GROUP_PAIRS 里逐条定义，README 的表格就是它们）
------------------------------------------------------
每个实验组是"两个检索条件的配对比较"，差异只在**一处**（见每组的 note），
这样才能把 overlap / τ 的差异归到那一个因素上：

    E1  两个实现的分歧机制（附带，作业未要求）：TF-IDF 余弦 vs BM25
    E2  TF-IDF 内部：文档侧要不要乘 idf（lnc.ltc vs ltc.ltc）
    E3  BM25 内部：idf 用 Okapi 还是课件版
    E4  BM25 内部：idf 用 Okapi 还是 Lucene 版
    E5  BM25 内部：idf 用课件版还是 Lucene 版（两者应当几乎重合）
    E6  钳制消融：Okapi 钳制 vs 不钳制（走 idf_fn 后门，只有含负 idf 词的查询会动）
    E7  参数敏感性：k1 × b 的 4×4 网格，每组都与默认 (1.2, 0.75) 比

度量只有两个（rank_metrics.py 有完整说明）：overlap@k 看集合重合、Kendall τ-b
看顺序一致。没有相关性标注，所以 P@10 / MAP / NDCG 一律不算——算不出来就不要
假装算出来。
"""

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

from .bm25 import (
    COURSE,
    DEFAULT_B,
    DEFAULT_K1,
    LUCENE,
    OKAPI,
    bm25_search,
    okapi_idf_unclamped,
)
from .inverted_index import load_index
from .queries import QUERIES, profile_query
from .rank_metrics import compare_pair, self_check
from .tfidf import LNC_LTC, LTC_LTC, tfidf_search

ROOT = Path(__file__).resolve().parents[1]

# 深池：两侧各取前 POOL 条，overlap@50 和 τ 的并集/交集都在它上面算。
POOL = 50
KS = (5, 10, 20, 50)


# ---------------------------------------------------------------- 检索条件

def _run_tfidf(index, text, top_k, variant):
    return tfidf_search(index, text, top_k=top_k, variant=variant)


def _run_bm25(index, text, top_k, variant=OKAPI, k1=DEFAULT_K1, b=DEFAULT_B,
              idf_fn=None):
    return bm25_search(index, text, top_k=top_k, variant=variant, k1=k1, b=b,
                       idf_fn=idf_fn)


def base_conditions():
    """六个基础条件：两个实现 + 各自的变体消融。"""
    return [
        {"name": "tfidf_lnc_ltc", "family": "tfidf",
         "label": "TF-IDF 余弦 lnc.ltc",
         "note": "文档侧 1+ln(tf) 不含 idf（课上那套，也是 Manning 教材默认）",
         "run": lambda index, text, k: _run_tfidf(index, text, k, LNC_LTC)},
        {"name": "tfidf_ltc_ltc", "family": "tfidf",
         "label": "TF-IDF 余弦 ltc.ltc",
         "note": "文档侧也乘 idf（教科书正文最常见的那版，分子等价于 idf²）",
         "run": lambda index, text, k: _run_tfidf(index, text, k, LTC_LTC)},
        {"name": "bm25_okapi", "family": "bm25",
         "label": "BM25 okapi (k1=1.2, b=0.75)",
         "note": "idf = max(0, ln((N−df+0.5)/(df+0.5)))，交付口径，默认参数",
         "run": lambda index, text, k: _run_bm25(index, text, k, OKAPI)},
        {"name": "bm25_course", "family": "bm25",
         "label": "BM25 course",
         "note": "idf = ln(N/df)，课件 Lecture 9.1 原文，与 TF-IDF 同一个 idf",
         "run": lambda index, text, k: _run_bm25(index, text, k, COURSE)},
        {"name": "bm25_lucene", "family": "bm25",
         "label": "BM25 lucene",
         "note": "idf = ln(1+(N−df+0.5)/(df+0.5))，Lucene 8.x/9.x 实际实现",
         "run": lambda index, text, k: _run_bm25(index, text, k, LUCENE)},
        {"name": "bm25_okapi_unclamped", "family": "bm25",
         "label": "BM25 okapi 不钳制（实验接口）",
         "note": "走 idf_fn 后门，idf 可为负；只用于量化钳制的影响，不进交付",
         "run": lambda index, text, k: _run_bm25(index, text, k, OKAPI,
                                                 idf_fn=okapi_idf_unclamped)},
    ]


def grid_conditions(k1_values=(0.0, 0.9, 1.2, 2.0), b_values=(0.0, 0.3, 0.75, 1.0)):
    """k1 × b 的参数网格（作业要求 "correctly handle the parameters k1 and b"）。

    每一组都与默认 (1.2, 0.75) 比，所以差异只有一个参数维度。k1=0 是退化点
    （词频饱和因子恒为 1，只剩 idf 求和），b=0 是完全不做长度归一。
    """
    return [
        {"name": f"bm25_k1_{k1}_b_{b}", "family": "bm25_grid",
         "label": f"BM25 okapi k1={k1}, b={b}", "note": f"参数网格：k1={k1}, b={b}",
         "k1": k1, "b": b, "is_default": (k1 == DEFAULT_K1 and b == DEFAULT_B),
         "run": lambda index, text, k, _k1=k1, _b=b: _run_bm25(
             index, text, k, OKAPI, k1=_k1, b=_b)}
        for k1 in k1_values for b in b_values
    ]


def group_pairs():
    """实验组 = 两个条件的配对。差异都收敛在一处，见每组的 note。"""
    pairs = [
        ("E1", "tfidf_lnc_ltc", "bm25_okapi",
         "跨实现对比（附带分析，作业未要求；字面要求是 impl_analysis.py 的逐实现画像）："
         "差异是整套打分公式"),
        ("E2", "tfidf_lnc_ltc", "tfidf_ltc_ltc",
         "TF-IDF 内部消融：文档侧要不要乘 idf，其余全部相同"),
        ("E3", "bm25_okapi", "bm25_course",
         "BM25 内部消融：只换 idf（Okapi 钳制式 vs 课件 ln(N/df)）"),
        ("E4", "bm25_okapi", "bm25_lucene",
         "BM25 内部消融：只换 idf（Okapi 钳制式 vs Lucene 加 1 式）"),
        ("E5", "bm25_course", "bm25_lucene",
         "BM25 内部消融：只换 idf（课件式 vs Lucene 式，预期几乎重合）"),
        ("E6", "bm25_okapi", "bm25_okapi_unclamped",
         "钳制消融：只去掉 max(0,·)，预期只有含 df>(N+1)/2 词项的查询会变"),
    ]
    return pairs


# ---------------------------------------------------------------- 跑批

def run_condition(index, condition, query_text, top_k):
    """跑一个条件一次，返回 (排序结果, 耗时毫秒)。"""
    started = time.perf_counter()
    ranked = condition["run"](index, query_text, top_k)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return ranked, round(elapsed_ms, 2)


def run_query(index, conditions, query):
    """一条查询 × 所有条件，把**完整排序**留在内存里给指标用。

    每个条件跑两趟，两趟各有用途，别合成一趟：

        完整排序    top_k = 全库篇数，拿到每个候选文档的真实名次。τ 要在两侧的
                    top-50 并集上比较名次，如果只取前 50 条，"这一侧没进前 50、
                    另一侧进了"的文档在这一侧就没有名次，τ 算不下去（直接 KeyError）。
        top-10     只取 10 条，这一趟的耗时才是报告里能写的"单条查询耗时"——
                   真实检索只要前 10 条，没人对 37,401 篇做全排序。
    """
    n_docs = len(index.doc_lengths)
    out = {}
    for condition in conditions:
        ranked_full, ms_full = run_condition(index, condition, query["text"], n_docs)
        _, ms_10 = run_condition(index, condition, query["text"], 10)
        out[condition["name"]] = {"ranked": ranked_full, "ms": ms_10,
                                  "ms_full_sort": ms_full}
    return out


def pair_metrics_for_query(query_runs, pairs, ks=KS, pool=POOL):
    """一条查询上，把所有实验组的指标一次算完（此时完整排序还在内存里）。"""
    return {group: compare_pair(query_runs[name_a]["ranked"],
                                query_runs[name_b]["ranked"], ks=ks, pool=pool)
            for group, name_a, name_b, _note in pairs}


# ---------------------------------------------------------------- 聚合

def _mean(values):
    values = [v for v in values if v is not None]
    return round(statistics.fmean(values), 4) if values else None


def aggregate(group, queries, ks=KS):
    """把一个实验组的逐条指标聚成几行数。按轴、按 df 分档各一组。"""
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
        row["top1_same_rate"] = _mean([1.0 if m["top1_same"] else 0.0
                                       for m in subset])
        row["n_intersection"] = _mean([m["n_intersection"] for m in subset])
        return row

    rows.append(summarize("全部查询", [q["id"] for q in queries]))

    axes = {}
    for query in queries:
        axes.setdefault(query["axis"], []).append(query["id"])
    for axis, ids in sorted(axes.items()):
        row = summarize(f"axis={axis}", ids)
        if row:
            rows.append(row)

    bands = {}
    for query in queries:
        bands.setdefault(query.get("df_band", "n/a"), []).append(query["id"])
    for band, ids in sorted(bands.items()):
        row = summarize(f"band={band}", ids)
        if row:
            rows.append(row)

    return [row for row in rows if row]


def condition_summary(runs, conditions, queries):
    """每个条件自身的画像：平均耗时、平均返回条数、top-10 文档长度中位数。"""
    rows = []
    for condition in conditions:
        name = condition["name"]
        per_query = runs[name]
        returned = [per_query[q["id"]]["n_returned"] for q in queries]
        ms = [per_query[q["id"]]["ms"] for q in queries]
        dls = [per_query[q["id"]]["top10_dl_median"] for q in queries
               if per_query[q["id"]]["top10_dl_median"] is not None]
        rows.append({"condition": name, "label": condition["label"],
                     "note": condition["note"],
                     "mean_ms": _mean(ms), "max_ms": round(max(ms), 2),
                     "mean_n_returned": _mean(returned),
                     "median_top10_dl": round(statistics.median(dls), 1) if dls else None,
                     "n_empty_results": sum(1 for n in returned if n == 0)})
    return rows


# ---------------------------------------------------------------- 输出

def _fmt_tau(metrics_union, metrics_intersection):
    """τ 在 Markdown 里统一保留 4 位小数；算不出来就写 n/a。"""
    union = metrics_union["tau"] if metrics_union else None
    intersection = metrics_intersection["tau"] if metrics_intersection else None
    return (f"{union:.4f}" if union is not None else "n/a",
            f"{intersection:.4f}" if intersection is not None else "n/a")


def write_markdown(path, meta, profiles, condition_rows, groups, aggregates,
                   grid_rows):
    """写给人看的汇总表格。"""
    lines = ["# 查询实验结果汇总", ""]
    lines += [
        f"- 语料：{meta['reviews']} 条评论，分词器 {meta['tokenizer']}，"
        f"avgdl={meta['avgdl']}",
        f"- 查询数：{meta['n_queries']}（作业要求 ≥5；IIR §8.1 的经验下限是 50 条信息需求）",
        f"- 深池：每侧取前 {meta['pool']} 条；overlap@k 的 k = "
        + ", ".join(str(k) for k in meta["ks"]),
        f"- 排序 tiebreak：{meta['tiebreak']}",
        "- 指标：overlap@k（集合重合，分母固定为 k）+ Kendall τ-b（顺序一致，"
        "并集与交集各报一次）",
        "",
    ]

    lines += ["## 1. 检索条件", "",
              "| 条件 | 说明 | 平均耗时 ms | 最大耗时 ms | 平均返回条数 | "
              "top-10 文档长度中位数 | 空结果条数 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in condition_rows:
        lines.append(
            f"| `{row['condition']}` | {row['note']} | {row['mean_ms']} | "
            f"{row['max_ms']} | {row['mean_n_returned']} | {row['median_top10_dl']} | "
            f"{row['n_empty_results']} |")
    lines.append("")

    lines += ["## 2. 实验组与聚合结果", ""]
    for group, agg_rows in aggregates.items():
        if group.startswith("E7::"):
            continue  # 参数网格的 16 组不在这里展开，统一走第 3 节的紧凑表
        note = groups[group]["note"]
        name_a, name_b = groups[group]["pair"]
        lines += [f"### {group}：`{name_a}` vs `{name_b}`", "",
                  f"差异只在：{note}", "",
                  "| 切片 | 查询数 | " +
                  " | ".join(f"overlap@{k}" for k in meta["ks"]) +
                  " | τ 并集 | τ 交集 | 首条相同率 |",
                  "|---|---:|" + "---:|" * (len(meta["ks"]) + 3)]
        for row in agg_rows:
            cells = [str(row[f"overlap@{k}"]) for k in meta["ks"]]
            lines.append(
                f"| {row['slice']} | {row['n_queries']} | " + " | ".join(cells) +
                f" | {row['tau_union']} | {row['tau_intersection']} | "
                f"{row['top1_same_rate']} |")
        lines.append("")

    if grid_rows:
        lines += ["## 3. 参数敏感性网格（k1 × b，均与默认 k1=1.2 / b=0.75 比）", "",
                  "| k1 | b | overlap@10 | τ 并集 | 首条相同率 | 说明 |",
                  "|---:|---:|---:|---:|---:|---|"]
        for row in grid_rows:
            lines.append(
                f"| {row['k1']} | {row['b']} | {row['overlap@10']} | "
                f"{row['tau_union']} | {row['top1_same_rate']} | {row['note']} |")
        lines.append("")

    lines += ["## 4. 逐条查询画像", "",
              "| 查询 | 文本 | 轴 | 词项 | df 范围 | df 分档 | 候选文档数 |",
              "|---|---|---|---|---|---|---:|"]
    for query, profile in profiles:
        dfs = [str(item["df"]) for item in profile["term_df"]]
        lines.append(
            f"| {query['id']} | {query['text']} | {query['axis']} | "
            f"{profile['n_unique_terms']} | {' / '.join(dfs)} | {profile['df_band']} | "
            f"{profile['n_candidates']} |")
    lines.append("")

    lines += ["## 5. E1（两个实现）逐条明细", "",
              "| 查询 | 文本 | 轴 | df 分档 | overlap@5 | overlap@10 | overlap@20 | "
              "overlap@50 | τ 并集 | τ 交集 | 首条相同 | TF-IDF 未召回 | BM25 未召回 |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for qid, metrics in groups["E1"]["per_query"].items():
        tau_u, tau_i = _fmt_tau(metrics["tau_b_union"], metrics["tau_b_intersection"])
        lines.append(
            f"| {qid} | {metrics['text']} | {metrics['axis']} | {metrics['df_band']} | "
            f"{metrics['overlap']['@5']} | {metrics['overlap']['@10']} | "
            f"{metrics['overlap']['@20']} | {metrics['overlap']['@50']} | "
            f"{tau_u} | {tau_i} | {'是' if metrics['top1_same'] else '否'} | "
            f"{metrics['n_missing_a']} | {metrics['n_missing_b']} |")
    lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path, groups, ks=KS):
    """长表：group × query × k → overlap，画图直接读这个。"""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "query_id", "text", "axis", "df_band", "k",
                         "overlap", "tau_union", "tau_intersection", "top1_same"])
        for group, payload in groups.items():
            for qid, metrics in payload["per_query"].items():
                tau_u = metrics["tau_b_union"]["tau"] if metrics["tau_b_union"] else ""
                tau_i = (metrics["tau_b_intersection"]["tau"]
                         if metrics["tau_b_intersection"] else "")
                for k in ks:
                    writer.writerow([group, qid, metrics["text"], metrics["axis"],
                                     metrics["df_band"], k, metrics["overlap"][f"@{k}"],
                                     tau_u, tau_i,
                                     "1" if metrics["top1_same"] else "0"])


# ---------------------------------------------------------------- main

def load_index_from_disk(index_file, input_dir, tokenizer):
    """读已落盘的索引；语料/分词器对不上会直接报错，不会拿旧索引跑新语料。"""
    import json as _json
    reviews_path = input_dir / "review.json"
    manifest = _json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    source = {"file": reviews_path.name, "bytes": reviews_path.stat().st_size,
              "reviews": manifest["files"]["review.json"]["records"],
              "tokenizer": tokenizer}
    return load_index(index_file, source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/meridian_id")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "part2/results")
    parser.add_argument("--index-file", type=Path, default=ROOT / "part2/.cache/index.pkl")
    parser.add_argument("--tokenizer", default="custom")
    parser.add_argument("--pool", type=int, default=POOL)
    parser.add_argument("--ks", type=int, nargs="+", default=list(KS))
    parser.add_argument("--queries", default="", help="只跑指定查询 id，逗号分隔")
    parser.add_argument("--axes", default="", help="只跑指定轴，逗号分隔")
    parser.add_argument("--skip-grid", action="store_true", help="不跑 k1/b 参数网格")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet
    index = load_index_from_disk(args.index_file, args.input_dir, args.tokenizer)

    queries = QUERIES
    if args.queries:
        wanted = {q.strip() for q in args.queries.split(",") if q.strip()}
        queries = [q for q in queries if q["id"] in wanted]
    if args.axes:
        wanted = {a.strip() for a in args.axes.split(",") if a.strip()}
        queries = [q for q in queries if q["axis"] in wanted]
    if not queries:
        raise SystemExit("没有选中任何查询，检查 --queries / --axes")

    # 画像：每条查询的分词、df、候选数。实测，不写死。
    profiles = []
    for query in queries:
        profile = profile_query(index, query["text"])
        query["df_band"] = profile["df_band"]
        profiles.append((query, profile))

    if verbose:
        print(f"τ 自检：{self_check()}")
        print(f"条件 {6 + (0 if args.skip_grid else 16)} 个 × 查询 {len(queries)} 条")

    conditions = base_conditions()
    grid = [] if args.skip_grid else grid_conditions()
    pairs = group_pairs()
    for condition in grid:
        if condition["is_default"]:
            continue  # 默认参数那一组就是 bm25_okapi 本身，不用和自己比
        pairs.append((f"E7::{condition['name']}", "bm25_okapi", condition["name"],
                      f"参数敏感性：只改 k1={condition['k1']} / b={condition['b']}"))

    all_conditions = conditions + grid
    runs = {condition["name"]: {} for condition in all_conditions}
    groups = {group: {"pair": [name_a, name_b], "note": note, "per_query": {}}
              for group, name_a, name_b, note in pairs}

    # 逐条查询跑：完整排序只在内存里活一轮，算完指标就只留前 POOL 条落盘。
    started = time.perf_counter()
    for position, query in enumerate(queries, 1):
        query_runs = run_query(index, all_conditions, query)
        qid = query["id"]
        for name, payload in query_runs.items():
            ranked = payload["ranked"]
            runs[name][qid] = {
                "top": [[doc_id, round(score, 6)] for doc_id, score
                        in ranked[:args.pool]],
                "n_returned": len(ranked),
                "ms": payload["ms"],
                "ms_full_sort": payload["ms_full_sort"],
                "top10_dl_median": (
                    statistics.median([index.doc_lengths[doc_id]
                                       for doc_id, _ in ranked[:10]])
                    if ranked else None),
            }
        for group, metrics in pair_metrics_for_query(
                query_runs, pairs, ks=tuple(args.ks), pool=args.pool).items():
            metrics["axis"] = query["axis"]
            metrics["text"] = query["text"]
            metrics["df_band"] = query.get("df_band", "")
            name_a, name_b = groups[group]["pair"]
            metrics["n_returned_a"] = runs[name_a][qid]["n_returned"]
            metrics["n_returned_b"] = runs[name_b][qid]["n_returned"]
            groups[group]["per_query"][qid] = metrics
        if verbose and position % 10 == 0:
            print(f"  ... {position}/{len(queries)} 条查询")
    elapsed = time.perf_counter() - started

    aggregates = {group: aggregate(groups[group], queries, ks=tuple(args.ks))
                  for group in groups}
    condition_rows = condition_summary(runs, conditions, queries)

    grid_rows = []
    for condition in grid:
        group = groups.get(f"E7::{condition['name']}")
        row = aggregate(group, queries, ks=tuple(args.ks))[0] if group else None
        if row:
            grid_rows.append({"k1": condition["k1"], "b": condition["b"],
                              "overlap@10": row.get("overlap@10"),
                              "tau_union": row["tau_union"],
                              "top1_same_rate": row["top1_same_rate"],
                              "note": ("默认参数（与 bm25_okapi 同一条件）"
                                       if condition["is_default"] else "")})
        elif condition["is_default"]:
            grid_rows.append({"k1": condition["k1"], "b": condition["b"],
                              "overlap@10": "—", "tau_union": "—",
                              "top1_same_rate": "—",
                              "note": "默认参数，与自己比恒为 10/10、τ=1"})

    meta = {
        "generated_by": "python -m part2.query_experiment",
        "reviews": len(index.doc_lengths),
        "tokenizer": index.tokenizer,
        "avgdl": round(index.avg_doc_length, 4),
        "n_queries": len(queries),
        "query_axes": {axis: sum(1 for q in queries if q["axis"] == axis)
                       for axis in sorted({q["axis"] for q in queries})},
        "pool": args.pool,
        "ks": args.ks,
        "tiebreak": "分数降序，同分按 doc_id 升序",
        "elapsed_seconds": round(elapsed, 1),
        "citation_query_count": (
            "IIR (Manning/Raghavan/Schütze 2008) §8.1: 'As a rule of thumb, 50 "
            "information needs has usually been found to be a sufficient minimum.'"),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "query_experiment_runs.json").write_text(
        json.dumps({"meta": meta, "runs": runs}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8")
    (args.output_dir / "query_experiment_summary.json").write_text(
        json.dumps({"meta": meta,
                    "profiles": [dict(query, profile=profile)
                                 for query, profile in profiles],
                    "conditions": condition_rows,
                    "groups": {g: {"pair": groups[g]["pair"], "note": groups[g]["note"],
                                   "aggregate": aggregates[g],
                                   "per_query": groups[g]["per_query"]}
                               for g in groups},
                    "grid": grid_rows},
                   ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8")
    write_csv(args.output_dir / "query_experiment_overlap.csv", groups,
              ks=tuple(args.ks))
    write_markdown(args.output_dir / "query_experiment_summary.md", meta, profiles,
                   condition_rows, groups, aggregates, grid_rows)

    if verbose:
        print(f"\n耗时 {elapsed:.1f}s，产出写到 {args.output_dir}")
        for group in ["E1", "E2", "E3", "E4", "E5", "E6"]:
            if group not in aggregates:
                continue
            row = aggregates[group][0]
            print(f"{group} {groups[group]['pair']}: overlap@10={row['overlap@10']} "
                  f"overlap@50={row['overlap@50']} tau_union={row['tau_union']} "
                  f"top1_same={row['top1_same_rate']}")


if __name__ == "__main__":
    main()
