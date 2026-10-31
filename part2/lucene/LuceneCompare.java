/*
 * LuceneCompare.java — a real Apache Lucene 10.5.2 comparison probe (the
 *                      "compare your results with Lucene" item of assignment 3.3.2)
 * ---------------------------------------------------------------------------
 * Why write a new one instead of reusing LuceneProbe.java
 *   The old LuceneProbe only ran 6 queries and re-tokenized with StandardAnalyzer, so
 *   "formula differences" and "tokenization differences" were mixed up. Two things split them apart here:
 *
 *     (1) Feed Lucene **the tokens produced by our own tokenizer** (WhitespaceAnalyzer only
 *         splits on whitespace and does nothing else). Both sides then see identical token sequences,
 *         and what remains is purely the scoring implementation (idf formula, length normalization, norm quantization).
 *     (2) Also build a StandardAnalyzer index and run Lucene out of the box;
 *         that reading = "results under the standard library's default configuration", and its gap to (1) is the tokenizer's contribution.
 *
 * Two similarities are run on each index:
 *     BM25Similarity(1.2, 0.75)   Lucene's BM25 (k1 and b match our defaults)
 *     ClassicSimilarity           Lucene's built-in TF-IDF (sqrt(tf)·idf/√dl)
 *
 * Inputs (all UTF-8, exported to workDir by lucene_experiment.py):
 *     ids.txt        one review_id per line
 *     texts_tok.txt  one review per line, our already tokenized and normalized tokens, space-separated
 *     texts_raw.txt  one raw review text per line (\n \r \t replaced by spaces)
 *     queries.tsv    qid \t raw query text \t our tokenized query tokens (space-separated)
 *
 * Compile / run:
 *     javac -encoding UTF-8 -cp "<luceneLib>/*" -d classes LuceneCompare.java
 *     java  -cp "<luceneLib>/*;classes" LuceneCompare <workDir> <out.json> <topK>
 *
 * Output JSON: index statistics (numDocs / sumTotalTermFreq / avgdl), for every query on both indexes
 *   under each of the two similarities the totalHits and top-K (id + score), the raw explain of top-1,
 *   and the df of every query term on the ours_tok index (used to check that both sides tokenize identically).
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

    // four configurations: two indexes × two similarities
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
        // dump the raw norm of every document in the ours_tok index: this is first-hand data for reverse-engineering "what Lucene
        // quantizes the document length to". The phrase "length of field (approximate)" in explain
        // means dl is not the true length but an approximation decoded from the norm.
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
                  ? "WhitespaceAnalyzer (no further tokenization; tokens come from our custom tokenizer)"
                  : "StandardAnalyzer (Lucene default, with the English stop-word list)");
            stats.put(e.getKey(), s);
            System.out.println("  " + e.getKey() + " -> " + s);
        }

        // ---------- queries ----------
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
            // ours_tok index: use our tokens directly; std index: let StandardAnalyzer process the raw text
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
                // explain is enabled only on the ours_tok index: its tokens match ours one for one,
                // so the idf / dl / avgdl reported in explain can be taken back and recomputed. On the std index it
                // would just be tens of KB of noise.
                int depth = idxName.equals("ours_tok") ? explainDepth : 0;
                if (td.scoreDocs.length > 0 && depth > 0) {
                    // the raw explain carries the idf / dl / avgdl Lucene actually used, first-hand evidence
                    // for checking the formula. Raised to 10 docs when diagnosing; a normal run keeps only 1.
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

    // ---------------------------------------------------------------- indexing

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
        if (mergeToOneSegment) writer.forceMerge(1);   // only with a single segment can norms be read by docId
        writer.commit();
        writer.close();
        return DirectoryReader.open(directory);
    }

    /** Dump {review_id \t raw norm} for every document. The norm is the value Lucene actually uses when scoring. */
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
            return new BM25Similarity(1.2f, 0.75f);   // matches the defaults of our bm25.py
        }
        return new ClassicSimilarity();
    }

    /** SHOULD boolean query = a document is retrieved if it matches any query term, scored as the sum of matched terms.
     *  This matches the default OR semantics of Lucene's QueryParser; QueryParser is avoided so that
     *  tokens containing punctuation (emoji, abbreviations, hyphens) are not parsed as query syntax. */
    static Query booleanShould(List<String> terms) {
        if (terms.isEmpty()) return null;
        BooleanQuery.Builder b = new BooleanQuery.Builder();
        for (String t : terms) {
            b.add(new TermQuery(new Term("text", t)), BooleanClause.Occur.SHOULD);
        }
        return b.build();
    }

    /** Split on whitespace and **deduplicate**.
     *
     *  Deduplication aligns with how we treat queries: textbook BM25 is written as Σ_{t∈Q}, with Q treated as a set
     *  of terms, so "pizza pizza" and "pizza" are the same information need. Lucene's BooleanQuery
     *  does not deduplicate: adding the same TermQuery twice as SHOULD exactly doubles the score (measured on this corpus for Q48
     *  `pizza pizza`; see "Known semantic difference" in lucene_comparison.md).
     *  Deduplicating explicitly here keeps the query semantics identical on both sides, so differences reflect only the scoring implementation. */
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

    // ---------------------------------------------------------------- utilities

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
