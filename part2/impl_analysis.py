"""逐实现分析：两个交付实现各自在 50 条查询上的行为画像。

为什么有这个脚本（以及它和 query_experiment.py 的分工）
------------------------------------------------------
作业 3.3.2 的原文是 "Run a set of at least five diverse queries on **both
implementations**"——要求的是"两个实现**各自**都要跑并给出分析"，**不是**要求把两个
实现拉到一起比个高下。所以：

    本脚本      = 作业的字面要求：每个实现单独跑，回答"这个实现行为对不对"。
    query_      = 额外做的：两个实现互相比，回答"它们为什么分歧"。
    experiment     没有相关性标注，所以那边测的是排序一致性，**测不了谁更好**。

这边算的指标全都**不需要相关性标注**，但都直接指向"实现的行为是否符合公式预期"：

    n_returned      召回条数。BM25 钳制到 0 会让整条查询返回 0 条，这条能抓到。
    margin_rel      首条与第二条的相对分差 (s1−s2)/|s1|。接近 0 说明第一名不是
                    "赢出来的"，是从并列堆里随手取的——此时换 tiebreak 就换第一名。
    n_tied_top1     与首条同分的文档数。同上，越多说明 top-1 越不可信。
    decay_10        s10/s1，前十条分数的集中度。接近 1 说明分数几乎拉不开。
    length_bias     top-10 的文档长度中位数 ÷ 全库平均长度。1.0 = 不偏长也不偏短；
                    >1 偏长文档、<1 偏短文档。用来验证归一化是不是真的生效。
    ms              单条查询取前 10 条的耗时。

`length_bias` 这一列还额外跑一次 **b 的扫描**（k1 固定 1.2，b = 0 / 0.3 /
0.75 / 1.0）：如果"长度归一"是真的在起作用，那么 top-10 的长度中位数应当随 b
单调下降。这是一条可以**证伪**的预期——不成立就说明实现有问题。

跑法（仓库根目录，索引已建好）：

    python -m part2.impl_analysis                    # 全量 50 条
    python -m part2.impl_analysis --queries Q01,Q45  # 只跑几条
    python -m part2.impl_analysis --quiet

产出（part2/results/）：
    impl_analysis.md    三张表：画像 / b 扫描 / 代表查询抽查
    impl_analysis.json  逐条原始指标，画图或复核用
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from .bm25 import DEFAULT_B, DEFAULT_K1, OKAPI, bm25_search
from .inverted_index import load_index
from .queries import QUERIES, df_band, profile_query
from .tfidf import LNC_LTC, tfidf_search

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "part2" / "results"
CORPUS = ROOT / "data" / "meridian_id" / "review.json"

# 抽查用的代表查询：五个轴各挑一条，外加 df=0 那条。
SPOTLIGHT = ["Q01", "Q25", "Q35", "Q43", "Q45", "Q50"]
SPOTLIGHT_TOPN = 5
SNIPPET_CHARS = 120

# 同分判定：相对差在 1e-9 以内算并列（全程 double，只挡浮点噪声）。
TIE_REL_TOL = 1e-9


# ---------------------------------------------------------------- 检索条件

def analysis_conditions():
    """本次要画像的条件：两个交付实现 + BM25 的 b 扫描（b=0.75 与交付口径重合）。"""
    conditions = [
        {"name": "tfidf_lnc_ltc", "label": "TF-IDF 余弦 lnc.ltc",
         "note": "交付实现 1：文档侧 1+ln(tf)，余弦归一化",
         "run": lambda index, text, k: tfidf_search(index, text, top_k=k,
                                                    variant=LNC_LTC)},
    ]
    for b in (0.0, 0.3, 0.75, 1.0):
        is_default = abs(b - DEFAULT_B) < 1e-12
        conditions.append({
            "name": f"bm25_b_{b}",
            "label": f"BM25 k1={DEFAULT_K1}, b={b}" + ("（交付口径）" if is_default else ""),
            "note": f"交付实现 2 的 b 扫描：b={b}",
            "b": b, "is_default": is_default,
            "run": lambda index, text, k, _b=b: bm25_search(
                index, text, top_k=k, variant=OKAPI, k1=DEFAULT_K1, b=_b),
        })
    return conditions


# ---------------------------------------------------------------- 单条指标

def query_metrics(index, condition, query, texts):
    """跑一条查询，算出这个条件在这条查询上的全部行为指标。"""
    n_docs = len(index.doc_lengths)
    started = time.perf_counter()
    ranked_full = condition["run"](index, query["text"], n_docs)
    ms_full = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    condition["run"](index, query["text"], 10)
    ms = (time.perf_counter() - started) * 1000

    top10 = ranked_full[:10]
    scores = [score for _doc, score in top10]
    s1 = scores[0] if scores else None
    s2 = scores[1] if len(scores) > 1 else None
    s10 = scores[-1] if len(scores) >= 10 else None

    n_tied = 0
    if s1 is not None:
        tol = TIE_REL_TOL * max(1.0, abs(s1))
        n_tied = sum(1 for _doc, score in ranked_full if abs(score - s1) <= tol)

    dls = [index.doc_lengths[doc_id] for doc_id, _ in top10]
    return {
        "text": query["text"],
        "axis": query["axis"],
        "df_band": df_band(profile_query(index, query["text"])["df_max"]),
        "n_returned": len(ranked_full),
        "top1_score": s1,
        "top2_score": s2,
        "margin_rel": ((s1 - s2) / abs(s1)) if s1 and s2 is not None else None,
        "n_tied_top1": n_tied,
        "decay_10": (s10 / s1) if s1 and s10 is not None and s1 else None,
        "top10_dl_median": statistics.median(dls) if dls else None,
        "top10_dl_mean": round(statistics.fmean(dls), 1) if dls else None,
        "length_bias": (round(statistics.median(dls) / index.avg_doc_length, 4)
                        if dls else None),
        "ms": round(ms, 2),
        "ms_full_sort": round(ms_full, 2),
        "top": [{"doc": doc_id, "score": round(score, 6),
                 "dl": index.doc_lengths[doc_id],
                 # 原文里可能有竖线，不转义会把 Markdown 表格撑坏
                 "snippet": (texts.get(doc_id) or "").replace("\n", " ")
                            .replace("|", r"\|")[:SNIPPET_CHARS]}
                for doc_id, score in top10],
    }


def run_all(index, conditions, queries, texts, verbose=True):
    """跑完 条件 × 查询 的网格。"""
    runs = {}
    total = len(conditions) * len(queries)
    done = 0
    for condition in conditions:
        per_query = {}
        for query in queries:
            per_query[query["id"]] = query_metrics(index, condition, query, texts)
            done += 1
            if verbose and done % 50 == 0:
                print(f"  ... {done}/{total}")
        runs[condition["name"]] = per_query
    return runs


# ---------------------------------------------------------------- 聚合

BAND_ORDER = ["df=0", "稀有", "低频", "中频", "高频", "超高频"]
NUMERIC_KEYS = ["n_returned", "margin_rel", "n_tied_top1", "decay_10",
                "top10_dl_median", "length_bias", "ms"]


def _median(values):
    values = [v for v in values if v is not None]
    return round(statistics.median(values), 4) if values else None


def aggregate(per_query, queries, slice_by):
    """按某个维度切片聚合。slice_by: 'all' | 'axis' | 'df_band'。"""
    profiles = {query["id"]: per_query[query["id"]] for query in queries}
    if slice_by == "all":
        groups = {"全部": [query["id"] for query in queries]}
    else:
        groups = {}
        for query in queries:
            key = profiles[query["id"]][slice_by]
            groups.setdefault(key, []).append(query["id"])
    rows = []
    for key, qids in groups.items():
        row = {"slice": key, "n_queries": len(qids)}
        for metric in NUMERIC_KEYS:
            row[metric] = _median([profiles[qid][metric] for qid in qids])
        rows.append(row)
    if slice_by == "df_band":
        # 切片名形如 "1000 <= df < 3000 (中频)"，按括号里的档位名排序，
        # 否则同一张表每次跑出来的行顺序会不一样。
        def band_rank(name):
            short = name.split("(")[-1].rstrip(")") if "(" in name else name
            return BAND_ORDER.index(short) if short in BAND_ORDER else 99

        rows.sort(key=lambda row: band_rank(row["slice"]))
    elif slice_by == "axis":
        rows.sort(key=lambda row: row["slice"])
    return rows


def build_profiles(runs, queries, condition_names):
    """{条件名: {'all': [...], 'axis': [...], 'df_band': [...]}}"""
    out = {}
    for name in condition_names:
        out[name] = {key: aggregate(runs[name], queries, key)
                     for key in ("all", "axis", "df_band")}
    return out


def b_scan(profiles):
    """b 扫描：length_bias 与 top-10 中位长度应当随 b 单调下降。"""
    rows = []
    for name in profiles:
        if not name.startswith("bm25_b_"):
            continue
        row = profiles[name]["all"][0]
        rows.append({"b": float(name.split("_")[-1]), "label": name,
                     "length_bias": row["length_bias"],
                     "top10_dl_median": row["top10_dl_median"],
                     "margin_rel": row["margin_rel"], "ms": row["ms"]})
    rows.sort(key=lambda row: row["b"])
    biases = [row["length_bias"] for row in rows if row["length_bias"] is not None]
    monotonic = all(biases[i] > biases[i + 1] for i in range(len(biases) - 1))
    return {"rows": rows, "monotonic_decreasing": monotonic}


def notable(runs, queries, condition_names):
    """值得单独拎出来的异常查询：零命中、首条并列、分数几乎拉不开。"""
    out = {}
    for name in condition_names:
        zero_hit, tied, flat = [], [], []
        for query in queries:
            metrics = runs[name][query["id"]]
            if metrics["n_returned"] == 0:
                zero_hit.append({"id": query["id"], "text": query["text"]})
            if metrics["n_tied_top1"] and metrics["n_tied_top1"] > 1:
                tied.append({"id": query["id"], "text": query["text"],
                             "n_tied_top1": metrics["n_tied_top1"]})
            if metrics["decay_10"] is not None and metrics["decay_10"] > 0.95:
                flat.append({"id": query["id"], "text": query["text"],
                             "decay_10": metrics["decay_10"]})
        out[name] = {"zero_hit": zero_hit, "tied_top1": tied, "flat_top10": flat}
    return out


# ---------------------------------------------------------------- 输出

def write_markdown(path, meta, profiles, scan, spotlight, anomalies, conditions):
    labels = {c["name"]: c["label"] for c in conditions}
    deliverables = [c["name"] for c in conditions
                    if c["name"] == "tfidf_lnc_ltc" or c.get("is_default")]
    metric_cols = [("召回条数", "n_returned"), ("首条相对分差", "margin_rel"),
                   ("首条并列数", "n_tied_top1"), ("s10/s1", "decay_10"),
                   ("top-10 长度中位", "top10_dl_median"),
                   ("长度倾向", "length_bias"), ("耗时 ms", "ms")]

    lines = ["# 逐实现分析（作业 3.3.2 的字面要求）", "",
             f"语料 {meta['n_docs']:,} 篇，平均长度 {meta['avgdl']}，"
             f"查询 {meta['n_queries']} 条，{meta['generated_at']}。", "",
             "这一节回答的是**每个实现各自的行为对不对**，不是两个实现谁更好——"
             "没有相关性标注，谁更好测不出来。", ""]

    for name in deliverables:
        lines += [f"## {labels[name]}", "",
                  "| 切片 | 查询数 | " + " | ".join(c for c, _k in metric_cols) + " |",
                  "|---|---:|" + "---:|" * len(metric_cols)]
        for key in ("all", "axis", "df_band"):
            for row in profiles[name][key]:
                cells = [("—" if row[k] is None else f"{row[k]}") for _c, k in metric_cols]
                lines.append(f"| {row['slice']} | {row['n_queries']} | "
                             + " | ".join(cells) + " |")
            lines.append("")

    lines += ["## b 扫描：长度归一是不是真的在压长文档", "",
              "k1 固定 1.2，只动 b。可证伪的预期是：**b 越大，top-10 的文档长度"
              "中位数越小**。", "",
              "| b | top-10 长度中位 | 长度倾向（÷ 全库均值） | 首条相对分差 | 耗时 ms |",
              "|---:|---:|---:|---:|---:|"]
    for row in scan["rows"]:
        lines.append(f"| {row['b']} | {row['top10_dl_median']} | "
                     f"{row['length_bias']} | {row['margin_rel']} | {row['ms']} |")
    lines += ["",
              f"单调下降：{'**是**，与预期一致' if scan['monotonic_decreasing'] else '**否**，预期被推翻，需要排查'}。",
              ""]

    lines += ["## 代表查询抽查（top-5，人工看相关性）", "",
              "每条列出两个实现各自的前 5 篇，附原文片段。**这是唯一能直接看出「检索"
              "出来的东西到底对不对」的地方**，但它靠人眼，不靠指标。", ""]
    for qid in spotlight:
        lines += [f"### {qid}", ""]
        for name in deliverables:
            metrics = spotlight[qid][name]
            lines += [f"**{labels[name]}**（召回 {metrics['n_returned']:,} 条）：", "",
                      "| # | 文档 | 分数 | 长度 | 原文片段 |",
                      "|---:|---|---:|---:|---|"]
            for position, item in enumerate(metrics["top"][:SPOTLIGHT_TOPN], 1):
                snippet = item["snippet"] or "（无文本）"
                lines.append(f"| {position} | `{item['doc'][:12]}…` | "
                             f"{item['score']} | {item['dl']} | {snippet} |")
            lines.append("")

    lines += ["## 需要单独说明的异常", ""]
    for name in deliverables:
        info = anomalies[name]
        lines += [f"### {labels[name]}", ""]
        lines.append(f"- **零命中**（返回 0 条）："
                     + ("、".join(f"`{x['id']}` {x['text']}" for x in info["zero_hit"])
                        or "无"))
        lines.append(f"- **首条并列**（第一名不是唯一最高分）："
                     + ("、".join(f"`{x['id']}`（{x['n_tied_top1']} 篇同分）"
                                  for x in info["tied_top1"][:8])
                        or "无"))
        lines.append(f"- **分数拉不开**（s10/s1 > 0.95）："
                     + ("、".join(f"`{x['id']}`" for x in info["flat_top10"][:8])
                        or "无"))
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- main

def load_texts():
    texts = {}
    with CORPUS.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            texts[record["review_id"]] = record.get("text") or ""
    return texts


def main(argv=None):
    parser = argparse.ArgumentParser(description="逐实现分析")
    parser.add_argument("--queries", help="只跑指定查询，逗号分隔，如 Q01,Q45")
    parser.add_argument("--skip-run", action="store_true",
                        help="已有 impl_analysis.json，只重新出 Markdown")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    verbose = not args.quiet
    RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS / "impl_analysis.json"
    md_path = RESULTS / "impl_analysis.md"

    queries = QUERIES
    if args.queries:
        wanted = {item.strip() for item in args.queries.split(",")}
        queries = [q for q in QUERIES if q["id"] in wanted]

    if verbose:
        print(f"载入语料文本（{CORPUS.name}）...")
    texts = load_texts()

    source = {"file": "review.json", "bytes": CORPUS.stat().st_size,
              "reviews": len(texts), "tokenizer": "custom"}
    index = load_index(ROOT / "part2" / ".cache" / "index.pkl", source)
    if verbose:
        print(f"索引 {len(index.doc_lengths):,} 篇，平均长度 {index.avg_doc_length}")

    conditions = analysis_conditions()
    names = [c["name"] for c in conditions]

    if verbose:
        print(f"跑 {len(conditions)} 个条件 × {len(queries)} 条查询...")
    started = time.perf_counter()
    runs = run_all(index, conditions, queries, texts, verbose)
    elapsed = time.perf_counter() - started

    profiles = build_profiles(runs, queries, names)
    scan = b_scan(profiles)
    anomalies = notable(runs, queries, names)
    spotlight = {qid: {name: runs[name][qid] for name in names}
                 for qid in SPOTLIGHT if qid in runs[names[0]]}

    payload = {
        "meta": {"n_docs": len(index.doc_lengths),
                 "avgdl": index.avg_doc_length,
                 "n_queries": len(queries),
                 "conditions": [{"name": c["name"], "label": c["label"],
                                 "note": c["note"]} for c in conditions],
                 "elapsed_seconds": round(elapsed, 1),
                 "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")},
        "profiles": profiles, "b_scan": scan,
        "anomalies": anomalies, "spotlight": spotlight,
        "per_query": runs,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    write_markdown(md_path, payload["meta"], profiles, scan, spotlight,
                   anomalies, conditions)

    if verbose:
        print(f"耗时 {elapsed:.1f}s → {md_path}")
        print(f"b 扫描单调下降：{scan['monotonic_decreasing']}")
        for row in scan["rows"]:
            print(f"  b={row['b']}: 长度倾向 {row['length_bias']}, "
                  f"top-10 中位长度 {row['top10_dl_median']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
