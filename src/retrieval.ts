// Keyword search (BM25) over chunks, and fusing it with the embedding ranking.
//
// Embeddings are good at "what does this do" questions; exact identifiers, error messages
// and file names are where keyword search wins. Reciprocal rank fusion (RRF) combines the two
// rankings without having to calibrate cosine scores against BM25 scores.
import { EMBED_DIM } from './protocol';
import type { Hit, IndexedChunk } from './repoIndex';

const STOPWORDS = new Set(
  (
    'a an and are as at be by can do does for from get got has have how i if in into is it its of on or ' +
    'should so that the their then there these this to use used uses using was what when where which who why ' +
    'will with you your me my we our happen happens handled handle work works way'
  ).split(' '),
);

/** Light stemming: plurals and a few verb endings, so "retries" matches "retry". */
function stem(t: string): string {
  if (t.length > 4 && t.endsWith('ies')) return `${t.slice(0, -3)}y`;
  if (t.length > 5 && t.endsWith('ing')) return t.slice(0, -3);
  if (t.length > 4 && t.endsWith('ed') && !t.endsWith('eed')) return t.slice(0, -2);
  if (t.length > 3 && t.endsWith('es') && /(ss|x|ch|sh)es$/.test(t)) return t.slice(0, -2);
  if (t.length > 3 && t.endsWith('s') && !t.endsWith('ss')) return t.slice(0, -1);
  return t;
}

/**
 * Code-aware tokens: identifiers are kept whole and also split into their camelCase /
 * snake_case parts, so `retryOnTimeout` matches "retry", "timeout" and "retryOnTimeout".
 */
export function tokenize(text: string): string[] {
  const out: string[] = [];
  for (const word of text.match(/[A-Za-z0-9_]+/g) ?? []) {
    const parts = word
      .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
      .replace(/([A-Z]+)([A-Z][a-z])/g, '$1 $2')
      .split(/[\s_]+/)
      .filter(Boolean);
    const whole = word.toLowerCase().replace(/^_+|_+$/g, '');
    if (parts.length > 1 && whole.length > 2) out.push(whole);
    for (const p of parts) {
      const t = p.toLowerCase();
      if (t.length < 2 || STOPWORDS.has(t) || /^\d+$/.test(t)) continue;
      out.push(stem(t));
    }
  }
  return out;
}

export interface Ranked {
  index: number;
  score: number;
}

/** Okapi BM25 over chunk text plus its file path. */
export class Bm25 {
  private postings = new Map<string, { doc: number; tf: number }[]>();
  private lengths: Float32Array;
  private avgLength: number;

  constructor(
    docs: Pick<IndexedChunk, 'path' | 'text'>[],
    private k1 = 1.2,
    private b = 0.75,
  ) {
    this.lengths = new Float32Array(docs.length);
    docs.forEach((d, i) => {
      // The path counts like text: "middleware/throttle.go" is strong evidence for "throttle".
      const tokens = tokenize(`${d.path} ${d.text}`);
      this.lengths[i] = tokens.length;
      const tf = new Map<string, number>();
      for (const t of tokens) tf.set(t, (tf.get(t) ?? 0) + 1);
      for (const [t, n] of tf) {
        let list = this.postings.get(t);
        if (!list) this.postings.set(t, (list = []));
        list.push({ doc: i, tf: n });
      }
    });
    this.avgLength = docs.length ? this.lengths.reduce((a, b) => a + b, 0) / docs.length : 1;
  }

  search(query: string, k: number): Ranked[] {
    const n = this.lengths.length;
    const scores = new Float32Array(n);
    for (const term of new Set(tokenize(query))) {
      const list = this.postings.get(term);
      if (!list) continue;
      const idf = Math.log(1 + (n - list.length + 0.5) / (list.length + 0.5));
      for (const { doc, tf } of list) {
        const norm = tf + this.k1 * (1 - this.b + (this.b * this.lengths[doc]!) / this.avgLength);
        scores[doc]! += (idf * tf * (this.k1 + 1)) / norm;
      }
    }
    return topK(scores, k).filter((r) => r.score > 0);
  }
}

/** Exact cosine ranking over `dim` leading dimensions (vectors stored with `stride` floats per row). */
export function denseSearch(
  vectors: Float32Array,
  query: Float32Array,
  k: number,
  dim = EMBED_DIM,
  stride = EMBED_DIM,
): Ranked[] {
  const n = vectors.length / stride;
  const scores = new Float32Array(n);
  for (let r = 0; r < n; r++) {
    let s = 0;
    const off = r * stride;
    for (let d = 0; d < dim; d++) s += vectors[off + d]! * query[d]!;
    scores[r] = s;
  }
  return topK(scores, k);
}

function topK(scores: Float32Array, k: number): Ranked[] {
  return Array.from(scores.keys())
    .sort((a, b) => scores[b]! - scores[a]!)
    .slice(0, k)
    .map((index) => ({ index, score: scores[index]! }));
}

/** Reciprocal rank fusion: score = Σ 1 / (k + rank). k = 60 is the standard constant. */
export function rrf(rankings: Ranked[][], k = 60): Ranked[] {
  const fused = new Map<number, number>();
  for (const ranking of rankings) {
    ranking.forEach((r, rank) => fused.set(r.index, (fused.get(r.index) ?? 0) + 1 / (k + rank + 1)));
  }
  return [...fused].map(([index, score]) => ({ index, score })).sort((a, b) => b.score - a.score);
}

export interface HybridHit extends Hit {
  similarity: number; // cosine similarity to the question
  keywordRank: number | null; // 1-based rank in the BM25 list, if it was there
}

/** Hybrid retrieval as used by the app: embeddings + BM25, fused with RRF. */
export function hybridSearch(
  index: { chunks: IndexedChunk[]; vectors: Float32Array },
  bm25: Bm25,
  queryText: string,
  queryVector: Float32Array,
  k: number,
  pool = 50,
): HybridHit[] {
  const dense = denseSearch(index.vectors, queryVector, index.chunks.length);
  const similarity = new Map(dense.map((r) => [r.index, r.score]));
  const keyword = bm25.search(queryText, pool);
  const keywordRank = new Map(keyword.map((r, i) => [r.index, i + 1]));
  return rrf([dense.slice(0, pool), keyword])
    .slice(0, k)
    .map((r) => ({
      ...index.chunks[r.index]!,
      index: r.index,
      score: similarity.get(r.index) ?? 0,
      similarity: similarity.get(r.index) ?? 0,
      keywordRank: keywordRank.get(r.index) ?? null,
    }));
}

/**
 * Whether a query contains code-like tokens (an identifier with internal capitals or
 * underscores, a call, a quoted string, a dotted name or an error type), where exact
 * keyword matching helps. Plain-English questions are left to the embeddings.
 */
export function looksLikeCode(query: string): boolean {
  return (
    /[a-z][A-Z]|[A-Za-z]_[A-Za-z]|\w\(|[`'"]|\b[A-Z]{2,}[a-z]|\b\w+\.\w+\b|\b[A-Z][a-z]+[A-Z]\w*|^[A-Z_]{3,}$|\b\w*(Error|Exception)\b/.test(query) ||
    query.trim().split(/\s+/).length <= 2
  );
}
