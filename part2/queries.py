"""查询集：50 条 diverse query，两个打分函数跑同一批。

为什么是 50 条
--------------
作业原文只要求 "a set of at least five diverse queries on both implementations"（3.3.2），
5 条是**及格线不是目标线**。Manning/Raghavan/Schütze《Introduction to Information
Retrieval》§8.1 给的是：

    "The test document collection and suite of information needs have to be of a
     reasonable size: you need to average performance over fairly large test sets,
     as results are highly variable over different documents and information needs.
     As a rule of thumb, 50 information needs has usually been found to be a
     sufficient minimum."

即 50 条信息需求是"经验上够用的下限"。这一条针对的是有相关性标注的评测（要在多条
查询上平均，因为单条查询的波动极大）；我们这份作业没有标注，只能算排名一致性，
但"单条查询波动大"这一点同样成立——之前 6 条查询上 TF-IDF 与 BM25 的 overlap@10
在 1/10 到 7/10 之间跳，正是这条经验法则要说的事。所以按 50 条来建。

50 条怎么分布
--------------
不能随机抓 50 个词。查询集要能支撑"差异从哪里来"的归因，所以按五个轴铺开，
每条查询只标一个主轴（axis），另有 purpose 说明"这条是为了看什么"：

    category     业态 / 主题检索（12 条）：Yelp 语料里真实存在的店类，1–3 个词
    aspect       方面 + 情感（12 条）：服务、等待、价格、态度这些评价维度
    attribute    属性约束（10 条）：gluten free、dog friendly 这类复合条件
    df_spectrum  单词项按 df 谱系铺开（10 条）：从 df=13,139 到 df=1，
                 专门用来看"查询词的稀有程度"是否就是两模型分歧的解释变量
    edge         边界与特殊输入（6 条）：全停用词、负 idf 词、重复词、缩写、
                 零命中词——这些不贡献统计效力，但验证实现没崩，且能演示钳制

df 分档（df_band）**不写死在表里**，而是在跑的时候按索引实测的 df 现算。写死的
分档会随语料变化而失效，实测的分档在换语料后仍然自洽。分档边界见 DF_BANDS。

边界查询的意义（edge 这 6 条重点看）：
    the and of            三个词 df 都接近 N，TF-IDF 的 idf→0、BM25 的 Okapi idf
                          钳到 0 → 两侧都应当"所有文档 0 分"，考察退不退得干净
    the food / pizza and pasta
                          全语料有 12 个词项的 Okapi 原始 idf 为负（df > (N+1)/2），
                          带上它们的查询就是钳制消融的现场
    pizza pizza           同一个词重复两次，考察 tf 权重与查询向量有没有被重复计数带偏
    don't waste your money 缩写展开成 do / not / waste / your / money，考察分词器
                          的缩写展开在查询侧是否和文档侧一致
    zzyzx                 df = 0，零命中，考察空结果是否优雅（不崩、指标标 n/a）
"""

from text_processing import get_tokenizer

# df 分档边界：按查询里**最高频**词项的 df 落档。df 越大词越常见、区分力越弱。
DF_BANDS = [
    ("df >= 10000 (超高频)", 10000, None),
    ("3000 <= df < 10000 (高频)", 3000, 10000),
    ("1000 <= df < 3000 (中频)", 1000, 3000),
    ("300 <= df < 1000 (低频)", 300, 1000),
    ("0 < df < 300 (稀有)", 1, 300),
    ("df = 0 (零命中)", 0, 1),
]

QUERIES = [
    # ---- category：业态 / 主题检索（12 条）----
    {"id": "Q01", "axis": "category", "text": "pizza",
     "purpose": "单一业态词，df=1,941，是最典型的主题检索基线"},
    {"id": "Q02", "axis": "category", "text": "sushi",
     "purpose": "第二个业态词，df=967，与 pizza 同档不同主题，检验结论不依赖某个具体词"},
    {"id": "Q03", "axis": "category", "text": "thai food",
     "purpose": "业态 + 泛指词 food（df=13,139），考察高频泛词是否被 idf 稀释"},
    {"id": "Q04", "axis": "category", "text": "taco tuesday",
     "purpose": "带时间限定（周二特价）的复合需求，两个词 df 都不高"},
    {"id": "Q05", "axis": "category", "text": "coffee shop",
     "purpose": "业态 + 场所词，shop 是高频词，考察高频泛词的干扰"},
    {"id": "Q06", "axis": "category", "text": "breakfast burrito",
     "purpose": "餐段 + 单品，两词 df 相差约 10 倍，考察 df 悬殊时的合并效应"},
    {"id": "Q07", "axis": "category", "text": "ice cream",
     "purpose": "甜品业态，ice 也是形容词/名词兼类，考察词性歧义"},
    {"id": "Q08", "axis": "category", "text": "bbq chicken",
     "purpose": "缩写型业态词（bbq），考察分词器对缩写的处理是否两侧一致"},
    {"id": "Q09", "axis": "category", "text": "barber shop",
     "purpose": "低频业态（barber df=41），长尾查询：候选少，排名应当高度一致"},
    {"id": "Q10", "axis": "category", "text": "dentist appointment",
     "purpose": "低频服务业（dentist df=142），与 barber 同属长尾，互为对照"},
    {"id": "Q11", "axis": "category", "text": "oil change",
     "purpose": "汽车服务，change 是多义高频词，考察多义词的拉扯"},
    {"id": "Q12", "axis": "category", "text": "brake repair",
     "purpose": "极低频（brake df=21），候选极少，是长尾端的极值样本"},

    # ---- aspect：方面 + 情感（12 条）----
    {"id": "Q13", "axis": "aspect", "text": "terrible service",
     "purpose": "负面情感 + 方面词，service df=11,347 是高频泛词，考察情感词能否压过泛词"},
    {"id": "Q14", "axis": "aspect", "text": "rude staff",
     "purpose": "负面情感 + 中频方面词（staff df=5,668）"},
    {"id": "Q15", "axis": "aspect", "text": "friendly staff",
     "purpose": "与 Q14 同方面词、情感极性相反，构成最小对：只有情感词不同"},
    {"id": "Q16", "axis": "aspect", "text": "slow service",
     "purpose": "速度维度 + 高频方面词，与 Q13 同方面词、不同维度词"},
    {"id": "Q17", "axis": "aspect", "text": "long wait",
     "purpose": "等待时长，wait df=3,506；wait 在 Yelp 里常出现在短评里，考察长度偏置"},
    {"id": "Q18", "axis": "aspect", "text": "overpriced",
     "purpose": "价格维度的单词项，df=339，低频且语义明确"},
    {"id": "Q19", "axis": "aspect", "text": "best pizza in town",
     "purpose": "已知分歧样本：TF-IDF 首条是没提 pizza 的短评（best+town 赢过 pizza），"
                "in 还是负 idf 词，一条顶两条用"},
    {"id": "Q20", "axis": "aspect", "text": "worst experience ever",
     "purpose": "三词全负面，experience/ever 都是高频泛词，考察泛词堆叠时谁主导"},
    {"id": "Q21", "axis": "aspect", "text": "cold food",
     "purpose": "温度/品质维度 + 高频泛词 food，与 Q03 的 food 对照"},
    {"id": "Q22", "axis": "aspect", "text": "wrong order",
     "purpose": "服务失误的具体类型，order 是多义词（点单/顺序）"},
    {"id": "Q23", "axis": "aspect", "text": "dirty bathroom",
     "purpose": "卫生维度，bathroom 低频、dirty 中低频，整体低频查询"},
    {"id": "Q24", "axis": "aspect", "text": "great atmosphere",
     "purpose": "正面评价 + 抽象名词，抽象词在评论里分布散，考察召回形态"},

    # ---- attribute：属性约束（10 条）----
    {"id": "Q25", "axis": "attribute", "text": "gluten free pancakes",
     "purpose": "三词复合约束（饮食限制 + 单品），是信息需求最接近真实检索的一条"},
    {"id": "Q26", "axis": "attribute", "text": "vegan options",
     "purpose": "饮食限制 + 泛指词，vegan df=182 很低频"},
    {"id": "Q27", "axis": "attribute", "text": "dog friendly patio",
     "purpose": "三词属性：宠物友好 + 户外座位，patio df=455"},
    {"id": "Q28", "axis": "attribute", "text": "outdoor seating",
     "purpose": "两词属性约束，与 Q27 共享语义（户外座位）但措辞不同，考察措辞敏感性"},
    {"id": "Q29", "axis": "attribute", "text": "drive thru",
     "purpose": "服务形式属性，thru 是 through 的口语写法，考察分词器是否在两侧一致处理"},
    {"id": "Q30", "axis": "attribute", "text": "free wifi",
     "purpose": "免费 + 设施，free df=1,478、wifi df=75，df 悬殊的两个属性词"},
    {"id": "Q31", "axis": "attribute", "text": "happy hour",
     "purpose": "促销属性，happy df=2,460 且是情感形容词，考察形容词的语义漂移"},
    {"id": "Q32", "axis": "attribute", "text": "kid friendly",
     "purpose": "人群属性 + 情感词，与 Q15 friendly staff 共享 friendly，构成对照"},
    {"id": "Q33", "axis": "attribute", "text": "wheelchair accessible",
     "purpose": "无障碍属性，两词都低频，长尾属性检索"},
    {"id": "Q34", "axis": "attribute", "text": "takeout order",
     "purpose": "服务形式属性，与 Q22 的 order 共享多义词，考察多义词在不同上下文的表现"},

    # ---- df_spectrum：单词项按 df 谱系铺开（10 条）----
    {"id": "Q35", "axis": "df_spectrum", "text": "food",
     "purpose": "df=13,139，谱系最高端：几乎每三篇就有一篇提到，区分力最弱"},
    {"id": "Q36", "axis": "df_spectrum", "text": "service",
     "purpose": "df=11,347，与 food 同为超高频泛词，互为重复验证"},
    {"id": "Q37", "axis": "df_spectrum", "text": "staff",
     "purpose": "df=5,668，高频档"},
    {"id": "Q38", "axis": "df_spectrum", "text": "wait",
     "purpose": "df=3,506，高频档下沿"},
    {"id": "Q39", "axis": "df_spectrum", "text": "salad",
     "purpose": "df=1,993，中频档"},
    {"id": "Q40", "axis": "df_spectrum", "text": "burger",
     "purpose": "df=1,552，中频档，与 salad 同档不同词"},
    {"id": "Q41", "axis": "df_spectrum", "text": "rude",
     "purpose": "df=917，低频档，情感词"},
    {"id": "Q42", "axis": "df_spectrum", "text": "patio",
     "purpose": "df=455，低频档，名词"},
    {"id": "Q43", "axis": "df_spectrum", "text": "espresso",
     "purpose": "df=76，稀有档：候选极少，两模型应当高度一致"},
    {"id": "Q44", "axis": "df_spectrum", "text": "xylophone",
     "purpose": "df=1，谱系最极端的一条（全语料只有 1 篇提到），考察极稀有的行为"},

    # ---- edge：边界与特殊输入（6 条）----
    {"id": "Q45", "axis": "edge", "text": "the and of",
     "purpose": "三个词 df 都 > 20,000：TF-IDF idf→0、BM25 的 Okapi idf 全被钳到 0，"
                "两侧都应当对所有文档给 0 分（结果集空或全同分）"},
    {"id": "Q46", "axis": "edge", "text": "the food",
     "purpose": "the 的 Okapi 原始 idf = −2.03，是钳制消融最狠的一条（实测 top-10 重合 1/10）"},
    {"id": "Q47", "axis": "edge", "text": "pizza and pasta",
     "purpose": "and 的原始 idf = −2.12，钳制前后 top-10 重合 6/10 且首条不同"},
    {"id": "Q48", "axis": "edge", "text": "pizza pizza",
     "purpose": "重复词项：TF-IDF 的查询向量不该被重复计数放大，BM25 按去重词项处理"},
    {"id": "Q49", "axis": "edge", "text": "don't waste your money",
     "purpose": "缩写展开成 do / not / waste / your / money（your 是高频词），"
                "考察展开后 not 是否被当成普通词参与打分"},
    {"id": "Q50", "axis": "edge", "text": "zzyzx",
     "purpose": "df = 0 的零命中查询：两侧都应返回空结果，指标标 n/a 而不是崩掉"},
]


def df_band(max_df):
    """把一个 df 落到 DF_BANDS 里的某一档，返回档位名。"""
    for name, low, high in DF_BANDS:
        if high is None:
            if max_df >= low:
                return name
        elif low <= max_df < high:
            return name
    return DF_BANDS[-1][0]


def profile_query(index, text):
    """对一条查询做实测画像：分词结果、每个词的 df、df 分档、候选文档数。

    df 全部现算，不写在 QUERIES 里——换语料/换分词器后画像自动跟着变，
    不会出现"表里写着 df=1941、实际已经变了"的失真。
    """
    terms = get_tokenizer(index.tokenizer).tokenize(text)
    dfs = [{"term": term, "df": index.df(term)} for term in sorted(set(terms))]
    hit_df = [item["df"] for item in dfs if item["df"]]
    candidates = set()
    for item in dfs:
        if item["df"]:
            candidates.update(index.postings[item["term"]])
    return {
        "text": text,
        "terms": terms,
        "n_terms": len(terms),
        "n_unique_terms": len({term: 1 for term in terms}),
        "term_df": dfs,
        "df_min": min(hit_df) if hit_df else 0,
        "df_max": max(hit_df) if hit_df else 0,
        "df_band": df_band(max(hit_df) if hit_df else 0),
        "n_candidates": len(candidates),
    }


def query_axes():
    """每个轴各有多少条查询，用来核对查询集没有偏到某一个轴上。"""
    counts = {}
    for query in QUERIES:
        counts[query["axis"]] = counts.get(query["axis"], 0) + 1
    return counts
