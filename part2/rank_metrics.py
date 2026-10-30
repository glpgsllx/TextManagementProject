"""排名一致性指标（无相关性标注时用）：overlap@k 与 Kendall tau-b。

为什么需要这两个指标
--------------------
作业只给了语料、没有给相关性标注（qrels），所以 P@10、MAP、NDCG 这些**依赖标注**
的指标一个都算不出来。能算的只有"两个排序结果有多像"，分两层看：

    overlap@k  只看集合：两侧各自的前 k 条里有几条是同一篇文档。
                回答"检索到的文档是不是同一批"，完全不看内部顺序。
    Kendall τ  只看顺序：把两侧都排到的文档拿出来，比较两两之间的先后关系是否一致。
                回答"同一批文档的先后顺序是不是一致"。

两个指标必须一起报：overlap 高而 τ 低 = 文档选得差不多但谁先谁后分歧很大；
overlap 低而 τ 高 = 只有少数共同文档，但这些文档的相对次序高度一致。

τ 用 tau-b 而不是 tau-a：打分函数会产生**并列**（同分文档），tau-a 把并列算进分母
会把 τ 系统性压低，tau-b 把两侧的并列对数从分母里扣掉（Wikipedia "Kendall rank
correlation coefficient" 的定义，n1/n2 就是两侧各自的并列对数）。

显著性的口径：只有在两侧都没有并列（n1 = n2 = 0）时，τ 的抽样方差才有那个经典的
z_A = (nc − nd) / sqrt(n(n−1)(2n+5)/18) 近似（这是严格秩的结论），这时才给 p 值。
有并列就不给 p，避免拿一个不成立的近似去讲"显著"。
"""

import math
from itertools import combinations


def rank_map(ranked):
    """[(doc_id, score)] → {doc_id: 名次}，名次从 1 开始，覆盖所有候选文档。

    输入必须是**已排序且带确定性 tiebreak** 的结果：分数降序、同分按 doc_id 升序。
    有了确定性 tiebreak，同分文档的先后才可复现，否则同分块内的顺序是 dict 遍历
    顺序（Python 里稳定但无语义），会把噪声当成"排名差异"。
    """
    return {doc_id: position for position, (doc_id, _) in enumerate(ranked, 1)}


def overlap_at_k(ranked_a, ranked_b, k):
    """两侧前 k 条结果的交集大小（0 … k）。

    注意它算的是**两个集合的交集大小**，不是 Jaccard：分母固定是 k，所以
    "7/10" 读作"前 10 条里有 7 条两边都有"，直观且可直接跨查询比较。
    """
    set_a = {doc_id for doc_id, _ in ranked_a[:k]}
    set_b = {doc_id for doc_id, _ in ranked_b[:k]}
    return len(set_a & set_b)


def kendall_tau_b(rank_x, rank_y):
    """Kendall tau-b：分母扣掉两侧各自的并列对数。两侧字典必须覆盖同一批文档。

    逐符号（n = 参与比较的文档数）：
        n0 = n(n−1)/2              所有文档对，tau-a 的分母
        n1 = 只在 x 侧并列的文档对数
        n2 = 只在 y 侧并列的文档对数
        nc = 一致对：两侧都判 a 在 b 前面（或都判 a 在 b 后面）
        nd = 不一致对：一侧说 a 在前、另一侧说 b 在前
        τ  = (nc − nd) / sqrt((n0 − n1)(n0 − n2))

    τ = +1 完全一致，−1 完全相反，0 无关。返回 None 表示算不出来（文档少于 2 篇，
    或两侧各自内部全部并列导致分母为 0——钳制 idf 后大量同分文档时会出现）。
    """
    docs = list(rank_x)
    n = len(docs)
    if n < 2:
        return None
    n0 = n * (n - 1) // 2
    n1 = n2 = nc = nd = 0
    for a, b in combinations(docs, 2):
        dx = rank_x[a] - rank_x[b]
        dy = rank_y[a] - rank_y[b]
        if dx == 0:
            n1 += 1
        if dy == 0:
            n2 += 1
        if dx == 0 or dy == 0:
            continue  # 有一侧并列：这一对不提供方向信息，计进 n1/n2 后跳过
        if (dx > 0) == (dy > 0):
            nc += 1
        else:
            nd += 1
    denom = math.sqrt((n0 - n1) * (n0 - n2))
    if denom == 0:
        return None
    tau = (nc - nd) / denom

    z = None
    if n1 == 0 and n2 == 0:
        variance = n * (n - 1) * (2 * n + 5) / 18.0
        if variance > 0:
            z = (nc - nd) / math.sqrt(variance)
    p_two_sided = math.erfc(abs(z) / math.sqrt(2)) if z is not None else None
    return {"tau": tau, "n": n, "n0": n0, "n1": n1, "n2": n2, "nc": nc, "nd": nd,
            "z": z, "p_two_sided": p_two_sided}


def compare_pair(ranked_a, ranked_b, ks=(5, 10, 20, 50), pool=50):
    """比较两个排序结果，一次性给出 overlap@k、tau-b（并集 / 交集）、首条是否相同。

    pool（默认 50）是"深池"：两侧各取前 pool 条。

        并集 τ   在两侧前 pool 条的**并集**上算。样本最大，但包含了"这边排到很后面、
                 那边根本没召回"的文档，成员差异会混进次序差异里，容易出负值。
        交集 τ   只在两侧**都**进了前 pool 条的文档上算。成员固定，纯看次序，
                 但样本变小，且天然偏向"两侧都认可的文档"，读数偏高。

    两个都报：并集 τ 是"整体一致性"的保守估计，交集 τ 是"共同文档的排序一致性"。

    关于"一侧压根没召回"的文档：这不是异常，而是常态。BM25 的 Okapi idf 会把
    df > (N+1)/2 的词钳到 0，idf 为 0 的词项直接跳过，于是**只命中这个词的文档
    根本不会出现在 BM25 的结果里**，却可能出现在 TF-IDF 的 top-50 中（TF-IDF 的
    idf = ln(N/df) 只是很小，不是 0）。这类文档在缺席的那一侧没有名次，统一记为
    "该侧最末名之后"（即 len(该侧结果) + 1，彼此并列）——这是标准的截尾处理，
    含义是"没召回 = 比召回的所有文档都靠后"。n_missing_a / n_missing_b 会把这类
    文档的数量报出来，看到这两个数不为 0 就该意识到 τ 里混着成员差异。
    """
    rank_a, rank_b = rank_map(ranked_a), rank_map(ranked_b)
    top_a = [doc_id for doc_id, _ in ranked_a[:pool]]
    top_b = [doc_id for doc_id, _ in ranked_b[:pool]]

    union = set(top_a) | set(top_b)
    intersection = set(top_a) & set(top_b)
    # 一侧没召回的文档：记为该侧最末名之后（并列）
    missing_a = len(rank_a) + 1
    missing_b = len(rank_b) + 1

    tau_union = kendall_tau_b({d: rank_a.get(d, missing_a) for d in union},
                              {d: rank_b.get(d, missing_b) for d in union})
    tau_intersection = (kendall_tau_b({d: rank_a[d] for d in intersection},
                                      {d: rank_b[d] for d in intersection})
                        if len(intersection) >= 2 else None)

    return {
        "overlap": {f"@{k}": overlap_at_k(ranked_a, ranked_b, k) for k in ks},
        "tau_b_union": tau_union,
        "tau_b_intersection": tau_intersection,
        "n_union": len(union),
        "n_intersection": len(intersection),
        "n_missing_a": sum(1 for d in union if d not in rank_a),
        "n_missing_b": sum(1 for d in union if d not in rank_b),
        "top1_same": bool(top_a) and bool(top_b) and top_a[0] == top_b[0],
    }


def self_check():
    """自检 τ 的实现：同序 1、逆序 −1、手算 4/6。拿错系数会立刻暴露。"""
    docs = list(range(1, 9))
    same = kendall_tau_b({d: i for i, d in enumerate(docs, 1)},
                         {d: i for i, d in enumerate(docs, 1)})
    reversed_ = kendall_tau_b({d: i for i, d in enumerate(docs, 1)},
                              {d: 9 - i for i, d in enumerate(docs, 1)})
    x = {"a": 1, "b": 2, "c": 3, "d": 4}
    y = {"a": 1, "b": 3, "c": 2, "d": 4}   # 交换一对：nc = 5, nd = 1, n0 = 6
    hand = kendall_tau_b(x, y)
    assert abs(same["tau"] - 1.0) < 1e-12, same
    assert abs(reversed_["tau"] + 1.0) < 1e-12, reversed_
    assert abs(hand["tau"] - 4 / 6) < 1e-12, hand
    return {"identical": same["tau"], "reversed": reversed_["tau"],
            "hand_check": hand["tau"]}
