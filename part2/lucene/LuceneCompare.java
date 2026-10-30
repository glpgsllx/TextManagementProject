/*
 * LuceneCompare.java — 真·Apache Lucene 10.5.2 对照探针（作业 3.3.2 的
 *                      "compare your results with Lucene" 那一条）
 * ---------------------------------------------------------------------------
 * 为什么要自己写一个而不是复用 LuceneProbe.java
 *   老的 LuceneProbe 只跑 6 条查询、用 StandardAnalyzer 重新分词，因此
 *   "公式差异"和"分词差异"混在一起分不开。这里做两件事把它拆开：
 *
 *     (1) 用**我们自己分词器产出的 token**喂给 Lucene（WhitespaceAnalyzer 只是按
 *         空白切开，不再做任何处理）。这样两侧 token 序列逐条一致，
 *         剩下的差异就纯粹是打分实现（idf 公式、长度归一、norm 量化）。
 *     (2) 再建一份 StandardAnalyzer 的索引，用 Lucene 开箱即用的方式跑，
 *         这份读数 = "标准库默认配置下的结果"，与 (1) 的差就是分词器的贡献。
 *
 * 每个索引上跑两个相似度：
 *     BM25Similarity(1.2, 0.75)   Lucene 的 BM25（k1、b 与我们的默认一致）
 *     ClassicSimilarity           Lucene 内置的 TF-IDF（sqrt(tf)·idf/√dl）
 *
 * 输入（全部 UTF-8，由 lucene_experiment.py 导出到 workDir）：
 *     ids.txt        每行一个 review_id
 *     texts_tok.txt  每行一条评论，我们已经切好并归一化的 token，空格分隔
 *     texts_raw.txt  每行一条评论原文（\n \r \t 已替换为空格）
 *     queries.tsv    qid \t 原始查询文本 \t 我们切好的查询 token（空格分隔）
 *
 * 编译 / 运行：
 *     javac -encoding UTF-8 -cp "<luceneLib>/*" -d classes LuceneCompare.java
 *     java  -cp "<luceneLib>/*;classes" LuceneCompare <workDir> <out.json> <topK>
 *
 * 输出 JSON：索引统计（numDocs / sumTotalTermFreq / avgdl）、每条查询在两份索引
 *   上各两个相似度下的 totalHits 与 top-K（id + score）、top-1 的 explain 原文，
 *   以及 ours_tok 索引上每个查询词的 df（用来校验两侧分词是否逐条一致）。
 */
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.Terms;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.search.similarities.ClassicSimilarity;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.store.FSDirectory;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class LuceneCompare {

    // 四个配置：两份索引 × 两个相似度
    static final String[][] CONFIGS = {
        {"lucene_bm25_ours_tok",    "ours_tok", "bm25"},
        {"lucene_classic_ours_tok", "ours_tok", "classic"},
        {"lucene_bm25_std",         "std",      "bm25"},
        {"lucene_classic_std",      "std",      "classic"},
    };

    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.err.println("usage: LuceneCompare <workDir> <out.json> <topK> [explainDepth]");
            System.exit(1);
        }
        Path workDir = Paths.get(args[0]);
        String outPath = args[1];
        int topK = Integer.parseInt(args[2]);
        int explainDepth = args.length > 3 ? Integer.parseInt(args[3]) : 1;

        Path idsPath = workDir.resolve("ids.txt");
        Path tokPath = workDir.resolve("texts_tok.txt");
        Path rawPath = workDir.resolve("texts_raw.txt");
        Path qPath = workDir.resolve("queries.tsv");

        Analyzer whitespace = new WhitespaceAnalyzer();
        Analyzer standard = new StandardAnalyzer();

        System.out.println("building index ours_tok (WhitespaceAnalyzer over our tokens) ...");
        DirectoryReader rTok = build(workDir.resolve("idx_ours_tok"), idsPath, tokPath,
                                     whitespace, true);
        // 把 ours_tok 索引每一篇的 norm 原始值全量落盘：这是反解"Lucene 把文档长度
        // 量化成什么"的一手数据。explain 里那句 "length of field (approximate)"
        // 就是说 dl 不是真实长度，而是 norm 反解出来的近似值。
        dumpNorms(rTok, workDir.resolve("norms_ours_tok.tsv"));
        System.out.println("building index std (StandardAnalyzer over raw text) ...");
        DirectoryReader rStd = build(workDir.resolve("idx_std"), idsPath, rawPath,
                                     standard, false);

        Map<String, DirectoryReader> readers = new LinkedHashMap<>();
        readers.put("ours_tok", rTok);
        readers.put("std", rStd);

        Map<String, Object> stats = new LinkedHashMap<>();
        for (Map.Entry<String, DirectoryReader> e : readers.entrySet()) {
            Map<String, Object> s = new LinkedHashMap<>();
            s.put("numDocs", e.getValue().numDocs());
            long sumTTF = 0;
            for (LeafReaderContext ctx : e.getValue().leaves()) {
                Terms terms = ctx.reader().terms("text");
                if (terms != null) sumTTF += terms.getSumTotalTermFreq();
            }
            s.put("sumTotalTermFreq", sumTTF);
            s.put("avgdl", (double) sumTTF / e.getValue().numDocs());
            s.put("analyzer", e.getKey().equals("ours_tok")
                  ? "WhitespaceAnalyzer（不再分词，token 由我们的 custom 分词器给出）"
                  : "StandardAnalyzer（Lucene 默认，含英文停用词表）");
            stats.put(e.getKey(), s);
            System.out.println("  " + e.getKey() + " -> " + s);
        }

        // ---------- 查询 ----------
        List<String[]> queries = new ArrayList<>();
        try (BufferedReader br = reader(qPath.toString())) {
            String line;
            while ((line = br.readLine()) != null) {
                if (line.isEmpty()) continue;
                String[] parts = line.split("\t", -1);
                queries.add(new String[]{parts[0],
                                         parts.length > 1 ? parts[1] : "",
                                         parts.length > 2 ? parts[2] : ""});
            }
        }
        System.out.println("queries=" + queries.size() + " topK=" + topK);

        StringBuilder out = new StringBuilder();
        out.append("{\n");
        out.append("  \"lucene\": \"10.5.2\",\n");
        out.append("  \"java\": \"").append(esc(System.getProperty("java.version")))
           .append("\",\n");
        out.append("  \"topK\": ").append(topK).append(",\n");
        out.append("  \"indexes\": ").append(toJson(stats)).append(",\n");
        out.append("  \"queries\": [\n");

        boolean firstQ = true;
        for (String[] q : queries) {
            String qid = q[0], raw = q[1], toks = q[2];
            // ours_tok 索引：直接用我们的 token；std 索引：让 StandardAnalyzer 处理原文
            List<String> termsTok = toks.isEmpty()
                ? new ArrayList<>() : split(toks);
            List<String> termsStd = analyze(standard, raw);

            if (!firstQ) out.append(",\n");
            firstQ = false;
            out.append("    {\"id\": \"").append(esc(qid)).append("\", ")
               .append("\"text\": \"").append(esc(raw)).append("\", ")
               .append("\"terms_ours\": ").append(toJson(termsTok)).append(", ")
               .append("\"terms_std\": ").append(toJson(termsStd)).append(", ")
               .append("\"df_ours_tok_index\": ").append(toJson(dfOf(rTok, termsTok)))
               .append(", ")
               .append("\"df_std_index\": ").append(toJson(dfOf(rStd, termsStd)))
               .append(",\n     \"results\": {\n");

            boolean firstCfg = true;
            for (String[] cfg : CONFIGS) {
                String cfgName = cfg[0], idxName = cfg[1], simName = cfg[2];
                DirectoryReader reader = readers.get(idxName);
                List<String> terms = idxName.equals("ours_tok") ? termsTok : termsStd;
                IndexSearcher searcher = new IndexSearcher(reader);
                searcher.setSimilarity(makeSim(simName));

                out.append("       ");
                if (!firstCfg) out.append(", ");
                firstCfg = false;
                out.append("\"").append(cfgName).append("\": {");

                Query query = booleanShould(terms);
                if (query == null) {
                    out.append("\"n_terms\": 0, \"totalHits\": 0, \"top\": []}");
                    continue;
                }
                TopDocs td = searcher.search(query, topK);
                out.append("\"n_terms\": ").append(terms.size())
                   .append(", \"totalHits\": ").append(td.totalHits.value())
                   .append(", \"top\": [");
                for (int i = 0; i < td.scoreDocs.length; i++) {
                    ScoreDoc sd = td.scoreDocs[i];
                    String rid = reader.storedFields().document(sd.doc).get("id");
                    if (i > 0) out.append(", ");
                    out.append("{\"rank\": ").append(i + 1)
                       .append(", \"id\": \"").append(esc(rid)).append("\"")
                       .append(", \"score\": ").append(sd.score)
                       .append("}");
                }
                out.append("]");
                // explain 只在 ours_tok 索引上开：那份索引的 token 与我们逐条一致，
                // explain 里报的 idf / dl / avgdl 才能拿回来复算。std 索引开了也是
                // 几十 KB 的噪音。
                int depth = idxName.equals("ours_tok") ? explainDepth : 0;
                if (td.scoreDocs.length > 0 && depth > 0) {
                    // explain 原文里带 Lucene 实际使用的 idf / dl / avgdl，是核对公式的
                    // 一手证据。诊断时把它开到 10 篇，正常跑只留 1 篇。
                    out.append(", \"explain\": [");
                    int n = Math.min(explainDepth, td.scoreDocs.length);
                    for (int i = 0; i < n; i++) {
                        Explanation ex = searcher.explain(query, td.scoreDocs[i].doc);
                        if (i > 0) out.append(", ");
                        out.append("\"").append(esc(ex.toString())).append("\"");
                    }
                    out.append("]");
                }
                out.append("}\n");
            }
            out.append("     }}\n");
            System.out.println("  done " + qid);
        }
        out.append("  ]\n}\n");

        Files.write(Paths.get(outPath), out.toString().getBytes(StandardCharsets.UTF_8));
        rTok.close();
        rStd.close();
        System.out.println("written -> " + outPath);
    }

    // ---------------------------------------------------------------- 索引

    static DirectoryReader build(Path dir, Path idsPath, Path textPath, Analyzer analyzer,
                                 boolean mergeToOneSegment) throws Exception {
        FSDirectory directory = FSDirectory.open(dir);
        IndexWriterConfig cfg = new IndexWriterConfig(analyzer);
        cfg.setOpenMode(IndexWriterConfig.OpenMode.CREATE);
        cfg.setRAMBufferSizeMB(512.0);
        IndexWriter writer = new IndexWriter(directory, cfg);
        long n = 0;
        try (BufferedReader ids = reader(idsPath.toString());
             BufferedReader texts = reader(textPath.toString())) {
            String id, text;
            while ((id = ids.readLine()) != null && (text = texts.readLine()) != null) {
                Document doc = new Document();
                doc.add(new StringField("id", id, Field.Store.YES));
                doc.add(new TextField("text", text, Field.Store.NO));
                writer.addDocument(doc);
                n++;
            }
        }
        if (mergeToOneSegment) writer.forceMerge(1);   // 只有一个段才好按 docId 取 norm
        writer.commit();
        writer.close();
        return DirectoryReader.open(directory);
    }

    /** 全量落盘 {review_id \t norm 原始值}。norm 是 Lucene 打分时真正用的那个数。 */
    static void dumpNorms(DirectoryReader reader, Path path) throws Exception {
        StringBuilder sb = new StringBuilder();
        org.apache.lucene.index.NumericDocValues norms = null;
        for (LeafReaderContext ctx : reader.leaves()) {
            norms = ctx.reader().getNormValues("text");
            if (norms != null) break;
        }
        if (norms == null) {
            Files.write(path, "no norms\n".getBytes(StandardCharsets.UTF_8));
            return;
        }
        for (int i = 0; i < reader.maxDoc(); i++) {
            String rid = reader.storedFields().document(i).get("id");
            String value = norms.advanceExact(i) ? String.valueOf(norms.longValue()) : "-";
            sb.append(rid).append("\t").append(value).append("\n");
        }
        Files.write(path, sb.toString().getBytes(StandardCharsets.UTF_8));
        System.out.println("norms -> " + path);
    }

    static Similarity makeSim(String name) {
        if (name.equals("bm25")) {
            return new BM25Similarity(1.2f, 0.75f);   // 与我们 bm25.py 的默认一致
        }
        return new ClassicSimilarity();
    }

    /** SHOULD 布尔查询 = 命中任一查询词即召回，分数为各命中词项之和。
     *  这和 Lucene QueryParser 默认 OR 的语义一致；不走 QueryParser 是为了避免
     *  token 里带标点（emoji、缩写、连字符）时被当成查询语法解析出错。 */
    static Query booleanShould(List<String> terms) {
        if (terms.isEmpty()) return null;
        BooleanQuery.Builder b = new BooleanQuery.Builder();
        for (String t : terms) {
            b.add(new TermQuery(new Term("text", t)), BooleanClause.Occur.SHOULD);
        }
        return b.build();
    }

    /** 按空白切词，并**去重**。
     *
     *  去重是为了和我们对查询的处理对齐：教科书 BM25 写的是 Σ_{t∈Q}，Q 被当成词项
     *  的集合，"pizza pizza" 与 "pizza" 是同一个信息需求。Lucene 的 BooleanQuery
     *  不去重——同一个 TermQuery 加两次 SHOULD，分数就会正好翻倍（本语料上 Q48
     *  `pizza pizza` 实测如此，见 lucene_comparison.md 的"已知语义差异"）。
     *  这里显式去重，让两边的查询语义一致，差异才只反映打分实现。 */
    static List<String> split(String s) {
        java.util.LinkedHashSet<String> set = new java.util.LinkedHashSet<>();
        for (String part : s.split(" ")) {
            if (!part.isEmpty()) set.add(part);
        }
        return new ArrayList<>(set);
    }

    static List<String> analyze(Analyzer analyzer, String text) throws Exception {
        java.util.LinkedHashSet<String> set = new java.util.LinkedHashSet<>();
        try (TokenStream ts = analyzer.tokenStream("text", text)) {
            CharTermAttribute att = ts.addAttribute(CharTermAttribute.class);
            ts.reset();
            while (ts.incrementToken()) set.add(att.toString());
            ts.end();
        }
        return new ArrayList<>(set);
    }

    static Map<String, Long> dfOf(DirectoryReader reader, List<String> terms)
            throws Exception {
        Map<String, Long> out = new LinkedHashMap<>();
        for (String t : terms) out.put(t, (long) reader.docFreq(new Term("text", t)));
        return out;
    }

    // ---------------------------------------------------------------- 工具

    static String toJson(Object o) {
        if (o instanceof Map) {
            StringBuilder sb = new StringBuilder("{");
            boolean first = true;
            for (Map.Entry<?, ?> e : ((Map<?, ?>) o).entrySet()) {
                if (!first) sb.append(", ");
                first = false;
                sb.append("\"").append(esc(String.valueOf(e.getKey()))).append("\": ")
                  .append(toJson(e.getValue()));
            }
            return sb.append("}").toString();
        }
        if (o instanceof List) {
            StringBuilder sb = new StringBuilder("[");
            boolean first = true;
            for (Object item : (List<?>) o) {
                if (!first) sb.append(", ");
                first = false;
                sb.append(toJson(item));
            }
            return sb.append("]").toString();
        }
        if (o instanceof Double || o instanceof Float) return String.valueOf(o);
        if (o instanceof Number) return String.valueOf(o);
        return "\"" + esc(String.valueOf(o)) + "\"";
    }

    static BufferedReader reader(String path) throws Exception {
        return new BufferedReader(
            new InputStreamReader(new FileInputStream(path), StandardCharsets.UTF_8));
    }

    static String esc(String s) {
        return s.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }
}
