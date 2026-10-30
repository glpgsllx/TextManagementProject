"""W5 对比分析：50 条查询的定性判读 + 钳制前后对比。

W5 在工作项清单里的五项，这个脚本承担第 4 项（定性判读 top 结果，每 query 一句）
和第 2 项（钳制前后对比）；其余三项已由别的脚本完成：

    1  overlap@k + Kendall τ（每 query）   → query_experiment.py（E1–E7）
    3  参数敏感性 k1 × b                   → query_experiment.py（E7，4×4 全网格）
    5  Lucene 三方对比                     → lucene_experiment.py（E8–E11）

定性判读为什么必须由人写、又为什么不能当真
------------------------------------------
没有 qrels，"相关性"没有标准答案。这里做的是**人工粗判**（一个判读者，只看 top-3
的原文片段），用途是给聚合数字一个可对照的具体例子，让读者知道 overlap@10 = 5.5
这种数字背后到底发生了什么。它**不是 ground truth**，不能拿来算 P@10 或宣称
"BM25 更好"。凡是要下质量结论的地方，报告里必须回到"无标注 → 只能谈一致性"。

判读时反复出现的四个现象（这也是 50 条判读的归纳，报告里可引）：
    A. TF-IDF 的首条几乎总是极短（4–20 词）、逐字复述查询词的评价。
    B. BM25 的首条更长、信息更多，但在高 df 查询上更容易漂到不相关的长评。
    C. 语料本身有重复/近重复评价，会同时污染两个实现的 top-k 多样性。
    D. 多词查询里 TF-IDF 可能漏掉核心词（Q19 首条是 burgers 不是 pizza）。

跑法（仓库根目录）：

    python -m part2.qualitative_reading

产出：part2/results/qualitative_reading.md
"""

import json
import time
from pathlib import Path

from .bm25 import DEFAULT_B, DEFAULT_K1, OKAPI, bm25_search, okapi_idf_unclamped
from .inverted_index import load_index
from .queries import QUERIES
from .tfidf import LNC_LTC, tfidf_search

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "part2" / "results"
CORPUS = ROOT / "data" / "meridian_id" / "review.json"

SNIPPET_CHARS = 90
TOP_N = 3

# 钳制前后对比的示例查询。第一个是工作项清单里点名的例子，后两个是本查询集里
# 真正触发钳制的边界查询。
CLAMP_DEMOS = ["fish and chips", "the and of", "pizza and pasta"]

# ---------------------------------------------------------------- 判读
# 每条一句。写法固定为"两个实现各自抓到了什么 + 这个差别意味着什么"，
# 不写"谁更好"（没有 qrels，判不了）。
READINGS = {
    "Q01": "两实现都在讲 pizza，但类型完全不同：TF-IDF 首条 17 词短评，BM25 首条 49 词的实际点餐体验。",
    "Q02": "首条两实现同一篇；BM25 第 2 条却是负面评价——两个实现都不区分情感极性，只匹配词。",
    "Q03": "TF-IDF 三条全是「Best Thai food in…」高度贴题；BM25 首条是讲「我爸妈如何评价泰餐」的段子，明显更差。",
    "Q04": "两实现 top-3 完全相同，是 50 条里最一致的一条（查询词都低频，候选少、分数稀疏）。",
    "Q05": "TF-IDF 三条都是十几词的「喜欢这家咖啡店」；BM25 首条 74 词描述店铺格局，信息更多但更散。",
    "Q06": "首条互不相同但都命中早餐 burrito；BM25 第 2、3 条是长评里被顺带提到的旁枝。",
    "Q07": "首条两实现同一篇；BM25 整体更长（52/18/44 词 vs 18/19/20 词）。",
    "Q08": "TF-IDF 第 1、2 条是**同一段文本的重复评价**——这正是检测到的「首条 2 篇同分」；BM25 首条换成了 46 词的正经 BBQ 评价。",
    "Q09": "两实现首条一致，都是理发店正面评价，只是第 2、3 条顺序不同。",
    "Q10": "两实现都命中看牙体验，首条长度接近（35 / 33 词），分歧只在排序。",
    "Q11": "前两条两实现完全一致，都是换油体验的正面评价。",
    "Q12": "首条一致（63 词问刹车维修）；TF-IDF 第 2、3 条是「iPhone repair」「Chip Repair」——只命中 repair，是部分匹配。",
    "Q13": "首条一致；BM25 第 3 条（57 词）漂到了保险公司，与餐饮无关。",
    "Q14": "TF-IDF 三条都字面含 rude/staff；BM25 第 2 条（40 词）完全没有这两个词，是典型漂移。",
    "Q15": "TF-IDF 首条仅 9 词（「…and produce, friendly staff.」）；BM25 首条 20 词、第 2 条 279 词。",
    "Q16": "TF-IDF 首条只有 4 词「Great food, slow service....」，字面完美匹配但信息量为零；BM25 首条 30 词。",
    "Q17": "首条一致；BM25 第 3 条（98 词）是通篇抱怨但没提等待。",
    "Q18": "首条一致；BM25 第 3 条（45 词）讲的是店里电视多，与 overpriced 无关。",
    "Q19": "**TF-IDF 首条是「best burgers in town」——四个查询词命中 best/in/town 却漏掉核心词 pizza**；召回 22,546 vs BM25 7,751（「in」被钳掉）。",
    "Q20": "前两条两实现一致；第 3 条不同但都是负面体验。",
    "Q21": "TF-IDF 第 3 条是「Great food and cold beer」（影院），把 cold 与 food 分开命中；BM25 第 2 条（46 词）没提 cold。",
    "Q22": "两实现 top-3 高度重合，都是「上错菜/送错单」的抱怨。",
    "Q23": "前两条一致；BM25 第 3 条（41 词）讲房间脏乱，是酒店不是餐厅。",
    "Q24": "TF-IDF 三条都是十几词的「Great … atmosphere」；BM25 首条 56 词提到披萨，与 atmosphere 关系弱。",
    "Q25": "首条一致（65 词）；BM25 第 3 条（42 词）真的在讲无麸质松饼，比 TF-IDF 第 3 条（8 词只提 gluten free options）更贴题。",
    "Q26": "TF-IDF 第 1、2 条是同一段文本的重复评价（即检测到的同分并列）；BM25 首条 57 词在评价素汉堡。",
    "Q27": "首条一致，且**首条是负面的**（「they don't allow dogs」）——两实现都不理解「dog friendly」是正向诉求。",
    "Q28": "两实现 top-3 完全一致（查询词低频，候选少）。",
    "Q29": "首条一致，且是负面评价（drive thru 慢且 rude）；BM25 第 2 条（54 词）跑到了星巴克。",
    "Q30": "BM25 首条（87 词）是「这里不适合用 WiFi 工作」的负面评价；TF-IDF 首条 17 词正面提到 free wifi。",
    "Q31": "首条一致；BM25 第 3 条与首条是近重复（34 / 48 词，同一段文本的两个版本）。",
    "Q32": "前两条两实现一致，都字面含 kid friendly。",
    "Q33": "TF-IDF 首条最贴题（「Seating is accessible for a scooter/wheelchair」）；BM25 第 2、3 条是同一段文本的重复。",
    "Q34": "两实现 top-3 完全一致。",
    "Q35": "TF-IDF 首条是 4 词的「Great food, slow service....」；BM25 首条 65 词却在说「food 是墨西哥菜的糟糕借口」——负面。",
    "Q36": "TF-IDF 首条 4 词、第 2 条 5 词，全是极短评价；BM25 首条 64 词讲最差客服。",
    "Q37": "两实现 top-3 完全一致（18/20/21 词，三条都是同一批短评）。",
    "Q38": "TF-IDF 首条 14 词「Worth the wait!」；BM25 首条 55 词讲排队等位。",
    "Q39": "两实现首条互换（TF1=BM2、TF2=BM1），成员相同只是顺序不同。",
    "Q40": "TF-IDF 前三条都是十几词短评；BM25 首条 135 词在横向比较汉堡店。",
    "Q41": "BM25 首条（47 词）讲「crabby teenager」，确属 rude 相关但没出现 rude 这个词；TF-IDF 首条字面含 rude。",
    "Q42": "首条一致（26 词正面）；BM25 第 2 条 146 词讲预订 patio 的纠纷。",
    "Q43": "稀有词（df=76），两实现候选都只有 76 篇；TF-IDF 首条 23 词直接提到 espresso bar，BM25 首条 117 词只在讲 drinks。",
    "Q44": "全库只有 1 篇命中，两实现一致返回同一篇（484 词）——df=1 时公式怎么算都只有它。",
    "Q45": "BM25 三个词 idf 全被钳成 0 → **返回 0 条**；TF-IDF 返回 36,558 条。这条查询本身没有信息需求，返回什么都无意义。",
    "Q46": "BM25 钳掉「the」（召回 13,139 = 只剩 food 的 df），TF-IDF 保留 34,169；BM25 首条与 Q35 首条相同。",
    "Q47": "TF-IDF 召回 33,613（含 and），BM25 2,165（钳掉 and）；两实现前两条仍一致，说明「and」对排序几乎没贡献。",
    "Q48": "重复词查询：两实现都按教科书记号把查询当**集合**去重，结果与 Q01 完全相同（Lucene 不去重，分数会翻倍）。",
    "Q49": "两实现 top-3 成员相同、顺序不同，都精准命中字面的「don't waste your money」。",
    "Q50": "df=0，两实现都返回 0 条，符合预期（生僻词不参与打分）。",
}


# ---------------------------------------------------------------- 跑批

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


def top_n(index, run, text, texts, n=TOP_N):
    ranked = run(text, len(index.doc_lengths))
    out = []
    for doc_id, score in ranked[:n]:
        out.append({"doc": doc_id, "score": round(score, 6),
                    "dl": index.doc_lengths[doc_id],
                    "snippet": (texts.get(doc_id) or "").replace("\n", " ")
                              .replace("|", r"\|")[:SNIPPET_CHARS]})
    return {"n_returned": len(ranked), "top": out}


def collect(index, texts, verbose=True):
    """两个交付实现 × 50 条查询的 top-3。"""
    runs = {"tfidf": {}, "bm25": {}}
    for query in QUERIES:
        runs["tfidf"][query["id"]] = top_n(
            index, lambda t, k: tfidf_search(index, t, top_k=k, variant=LNC_LTC),
            query["text"], texts)
        runs["bm25"][query["id"]] = top_n(
            index, lambda t, k: bm25_search(index, t, top_k=k, variant=OKAPI,
                                            k1=DEFAULT_K1, b=DEFAULT_B),
            query["text"], texts)
    return runs


def clamp_demo(index, texts):
    """钳制前后对比：同一条查询在「钳制」与「不钳制」两种 idf 下的结果。"""
    out = []
    for text in CLAMP_DEMOS:
        clamped = top_n(index, lambda t, k: bm25_search(
            index, t, top_k=k, variant=OKAPI, k1=DEFAULT_K1, b=DEFAULT_B), text, texts)
        unclamped = top_n(index, lambda t, k: bm25_search(
            index, t, top_k=k, variant=OKAPI, k1=DEFAULT_K1, b=DEFAULT_B,
            idf_fn=okapi_idf_unclamped), text, texts)
        out.append({"text": text, "clamped": clamped, "unclamped": unclamped})
    return out


def agreement(runs):
    """两实现的 top-1 / top-3 集合一致的查询有多少条——给判读一个数量打底。"""
    same1 = same3 = 0
    for qid in runs["tfidf"]:
        a = [item["doc"] for item in runs["tfidf"][qid]["top"]]
        b = [item["doc"] for item in runs["bm25"][qid]["top"]]
        if a and b and a[0] == b[0]:
            same1 += 1
        if set(a) == set(b):
            same3 += 1
    return {"n": len(runs["tfidf"]), "same_top1": same1, "same_top3_set": same3}


# ---------------------------------------------------------------- 输出

def write_markdown(path, runs, demos, agree):
    lines = ["# W5 定性判读：50 条查询逐条一句", "",
             "人工粗判（一个判读者，只看两个实现各自的 top-3 原文片段）。**这不是 ground "
             "truth**，只是给聚合数字配的具体例子——没有 qrels，质量判不了。", "",
             f"两实现 **top-1 相同 {agree['same_top1']}/{agree['n']} 条**，"
             f"**top-3 集合完全相同 {agree['same_top3_set']}/{agree['n']} 条**。", "",
             "判读里反复出现的四个现象：", "",
             "- **A**：TF-IDF 首条几乎总是极短（4–20 词）、逐字复述查询词；",
             "- **B**：BM25 首条更长、信息更多，但在高 df 查询上更容易漂到不相关的长评；",
             "- **C**：语料本身有重复/近重复评价，同时污染两个实现的 top-k 多样性；",
             "- **D**：多词查询里 TF-IDF 可能漏掉核心词（见 Q19）。", "",
             "## 逐条判读", "",
             "| 查询 | 轴 | TF-IDF 首条（长度） | BM25 首条（长度） | 判读 |",
             "|---|---|---:|---:|---|"]
    for query in QUERIES:
        qid = query["id"]
        tf = runs["tfidf"][qid]["top"]
        bm = runs["bm25"][qid]["top"]
        tf1 = f"{tf[0]['snippet']}（{tf[0]['dl']}）" if tf else "（0 条）"
        bm1 = f"{bm[0]['snippet']}（{bm[0]['dl']}）" if bm else "（0 条）"
        lines.append(f"| `{qid}` {query['text']} | {query['axis']} | {tf1} | "
                     f"{bm1} | {READINGS.get(qid, '')} |")
    lines += ["", "## 钳制前后对比", "",
              "同一条查询、同一套 k1/b，只切换 `max(0, ·)` 钳制是否生效。"
              "第一条是工作项清单里点名的例子。", ""]
    for demo in demos:
        lines += [f"### `{demo['text']}`", "",
                  "| | 召回条数 | 首条（长度） | 第 2 条（长度） | 第 3 条（长度） |",
                  "|---|---:|---|---|---|"]
        for label, key in [("钳制（交付口径）", "clamped"), ("不钳制", "unclamped")]:
            block = demo[key]
            cells = []
            for item in block["top"]:
                cells.append(f"{item['snippet']}（{item['dl']}）")
            while len(cells) < 3:
                cells.append("—")
            lines.append(f"| {label} | {block['n_returned']:,} | " + " | ".join(cells) + " |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    verbose = True
    RESULTS.mkdir(parents=True, exist_ok=True)
    texts = load_texts()
    source = {"file": "review.json", "bytes": CORPUS.stat().st_size,
              "reviews": len(texts), "tokenizer": "custom"}
    index = load_index(ROOT / "part2" / ".cache" / "index.pkl", source)
    started = time.perf_counter()
    runs = collect(index, texts)
    demos = clamp_demo(index, texts)
    agree = agreement(runs)
    md = RESULTS / "qualitative_reading.md"
    write_markdown(md, runs, demos, agree)
    print(f"耗时 {time.perf_counter() - started:.1f}s → {md}")
    print(f"top-1 相同 {agree['same_top1']}/{agree['n']}，"
          f"top-3 集合相同 {agree['same_top3_set']}/{agree['n']}")
    for demo in demos:
        print(f"  `{demo['text']}`: 钳制 {demo['clamped']['n_returned']:,} 条 / "
              f"不钳制 {demo['unclamped']['n_returned']:,} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
