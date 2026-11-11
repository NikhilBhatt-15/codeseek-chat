// Retrieval benchmark: how often does the right file come back for a question?
//
//   npm run eval                       # q4 (what the site ships), all retrievers
//   VARIANTS=fp32,fp16,q8,q4 npm run eval
//   RERANKER=<hf model id> npm run eval
//
// eval/benchmark.json pins each repo to a commit and labels every question with the files
// that answer it. Repos are indexed exactly like the site's Quick mode (same file selection
// and chunking), embedded with EmbeddingGemma 2 on the CPU, and cached in eval/.cache.
//
// Metrics (over chunks after overlap de-duplication, as the app does):
//   recall@k  share of a question's relevant files that appear in the top k chunks
//   hit@k     share of questions with at least one relevant file in the top k
//   MRR@10    1 / rank of the first chunk from a relevant file
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer, env } from '@huggingface/transformers';
import { formatDocument, formatQuery } from '../src/embedder';
import { fetchContents, fetchSnapshot, LIMITS } from '../src/github';
import { MAX_TOKENS, MODEL_ID } from '../src/protocol';
import { chunkRepo, type IndexedChunk } from '../src/repoIndex';
import { Bm25, denseSearch, looksLikeCode, rrf, type Ranked } from '../src/retrieval';

interface Benchmark {
  repos: { repo: string; commit: string }[];
  questions: { repo: string; question: string; files: string[]; kind?: string }[];
}

const root = process.env.CODESEEK_ROOT ?? join(dirname(fileURLToPath(import.meta.url)), '..');
const evalDir = join(root, 'eval');
const cacheDir = join(evalDir, '.cache');
if (process.env.HF_CACHE) env.cacheDir = process.env.HF_CACHE;
const bench = JSON.parse(readFileSync(join(evalDir, 'benchmark.json'), 'utf8')) as Benchmark;
const VARIANTS = (process.env.VARIANTS ?? 'q4').split(',');
const FULL_DIM = 768;
const KS = [1, 5, 10];

const slug = (repo: string) => repo.replace('/', '__');
const readJson = <T>(p: string): T | null => (existsSync(p) ? (JSON.parse(readFileSync(p, 'utf8')) as T) : null);
const writeJson = (p: string, v: unknown) => {
  mkdirSync(dirname(p), { recursive: true });
  writeFileSync(p, JSON.stringify(v));
};

// ---- corpora (Quick-mode file selection at the pinned commit) ----

interface Corpus {
  repo: string;
  paths: string[];
  chunks: IndexedChunk[];
}

async function loadCorpus(repo: string, commit: string): Promise<Corpus> {
  const p = join(cacheDir, slug(repo), `contents-${commit.slice(0, 12)}.json`);
  let contents = readJson<Record<string, string | null>>(p);
  if (!contents) {
    const [owner, name] = repo.split('/') as [string, string];
    const snap = await fetchSnapshot({ owner, repo: name, ref: commit }, LIMITS.quick, fetch, process.env.GITHUB_TOKEN);
    contents = Object.fromEntries(await fetchContents(snap, () => {}));
    writeJson(p, contents);
    // Plain copies, handy for reading the code when labelling questions.
    for (const [path, text] of Object.entries(contents)) {
      if (text === null) continue;
      const out = join(cacheDir, slug(repo), 'src', path);
      mkdirSync(dirname(out), { recursive: true });
      writeFileSync(out, text);
    }
  }
  return { repo, paths: Object.keys(contents).sort(), chunks: chunkRepo(new Map(Object.entries(contents))) };
}

// ---- embeddings (full 768 dims, normalised; 256 = Matryoshka prefix, re-normalised) ----

async function loadModel(dtype: string) {
  const config = await AutoConfig.from_pretrained(MODEL_ID);
  Object.assign(config, { vision_config: null, audio_config: null });
  const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
  const model = await AutoModel.from_pretrained(MODEL_ID, { config, device: 'cpu', dtype: dtype as 'q4' });
  return async (texts: string[]): Promise<Float32Array> => {
    const inputs = tokenizer(texts, { padding: true, truncation: true, max_length: MAX_TOKENS });
    const out = (await model(inputs)) as { sentence_embedding: { normalize(...a: unknown[]): { data: ArrayLike<number> } } };
    return Float32Array.from(out.sentence_embedding.normalize(2, -1).data);
  };
}

async function embedAll(embed: (t: string[]) => Promise<Float32Array>, texts: string[], label: string) {
  const out = new Float32Array(texts.length * FULL_DIM);
  const order = texts.map((_, i) => i).sort((a, b) => texts[a]!.length - texts[b]!.length);
  for (let s = 0; s < order.length; s += 8) {
    const idx = order.slice(s, s + 8);
    const v = await embed(idx.map((i) => texts[i]!));
    idx.forEach((row, j) => out.set(v.subarray(j * FULL_DIM, (j + 1) * FULL_DIM), row * FULL_DIM));
    process.stdout.write(`\r  ${label}: ${Math.min(s + 8, texts.length)}/${texts.length}`);
  }
  process.stdout.write('\n');
  return out;
}

function prefix(vectors: Float32Array, dim: number): Float32Array {
  const n = vectors.length / FULL_DIM;
  const out = new Float32Array(n * dim);
  for (let r = 0; r < n; r++) {
    let norm = 0;
    for (let d = 0; d < dim; d++) norm += vectors[r * FULL_DIM + d]! ** 2;
    norm = Math.sqrt(norm) || 1;
    for (let d = 0; d < dim; d++) out[r * dim + d] = vectors[r * FULL_DIM + d]! / norm;
  }
  return out;
}

interface Embedded {
  docs: Float32Array;
  queries: Float32Array;
  docSeconds: number;
}

async function embedCorpus(variant: string, corpus: Corpus, questions: string[], getModel: () => Promise<(t: string[]) => Promise<Float32Array>>): Promise<Embedded> {
  const base = join(cacheDir, 'embeddings', variant, slug(corpus.repo));
  const meta = readJson<{ chunks: number; docSeconds: number; questions: string[] }>(`${base}.json`);
  const sameQuestions = meta && JSON.stringify(meta.questions) === JSON.stringify(questions);
  if (meta && meta.chunks === corpus.chunks.length && sameQuestions && existsSync(`${base}.docs.bin`)) {
    return {
      docs: new Float32Array(readFileSync(`${base}.docs.bin`).buffer.slice(0)),
      queries: new Float32Array(readFileSync(`${base}.queries.bin`).buffer.slice(0)),
      docSeconds: meta.docSeconds,
    };
  }
  const embed = await getModel();
  let docs: Float32Array;
  let docSeconds: number;
  if (meta && meta.chunks === corpus.chunks.length && existsSync(`${base}.docs.bin`)) {
    docs = new Float32Array(readFileSync(`${base}.docs.bin`).buffer.slice(0));
    docSeconds = meta.docSeconds;
  } else {
    const t0 = performance.now();
    docs = await embedAll(embed, corpus.chunks.map((c) => formatDocument(c.path, c.text)), `${variant} ${corpus.repo}`);
    docSeconds = (performance.now() - t0) / 1000;
  }
  const queries = await embedAll(embed, questions.map(formatQuery), `${variant} ${corpus.repo} questions`);
  mkdirSync(dirname(base), { recursive: true });
  writeFileSync(`${base}.docs.bin`, new Uint8Array(docs.buffer));
  writeFileSync(`${base}.queries.bin`, new Uint8Array(queries.buffer));
  writeJson(`${base}.json`, { chunks: corpus.chunks.length, docSeconds, questions });
  return { docs, queries, docSeconds };
}

// ---- optional cross-encoder reranker ----

async function loadReranker(id: string) {
  const tokenizer = await AutoTokenizer.from_pretrained(id);
  const model = await AutoModelForSequenceClassification.from_pretrained(id, { device: 'cpu', dtype: 'q8' as 'q4' });
  return async (query: string, docs: string[]): Promise<number[]> => {
    const inputs = tokenizer(new Array(docs.length).fill(query), { text_pair: docs, padding: true, truncation: true, max_length: 512 });
    const { logits } = (await model(inputs)) as { logits: { data: ArrayLike<number>; dims: number[] } };
    const cols = logits.dims[1] ?? 1;
    return Array.from({ length: docs.length }, (_, i) => Number(logits.data[i * cols + cols - 1]));
  };
}

// ---- metrics ----

/** Drops chunks overlapping a better-ranked chunk of the same file, like the app's dedupeHits. */
function dedupe(ranked: Ranked[], chunks: IndexedChunk[]): Ranked[] {
  const kept: Ranked[] = [];
  for (const r of ranked) {
    const c = chunks[r.index]!;
    if (!kept.some((k) => { const o = chunks[k.index]!; return o.path === c.path && c.startLine <= o.endLine && o.startLine <= c.endLine; })) kept.push(r);
  }
  return kept;
}

interface Scores { [metric: string]: number }

function score(ranked: Ranked[], chunks: IndexedChunk[], relevant: string[]): Scores {
  const files = dedupe(ranked, chunks).slice(0, 10).map((r) => chunks[r.index]!.path);
  const s: Scores = {};
  for (const k of KS) {
    const top = new Set(files.slice(0, k));
    s[`recall@${k}`] = relevant.filter((f) => top.has(f)).length / relevant.length;
    s[`hit@${k}`] = relevant.some((f) => top.has(f)) ? 1 : 0;
  }
  const first = files.findIndex((f) => relevant.includes(f));
  s['MRR@10'] = first >= 0 ? 1 / (first + 1) : 0;
  return s;
}

const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / Math.max(1, xs.length);

// ---- run ----

const corpora = new Map<string, Corpus>();
for (const { repo, commit } of bench.repos) {
  const c = await loadCorpus(repo, commit);
  corpora.set(repo, c);
  console.log(`${repo}@${commit.slice(0, 7)}: ${c.paths.length} files, ${c.chunks.length} chunks`);
}
for (const q of bench.questions) {
  const c = corpora.get(q.repo);
  if (!c) throw new Error(`unknown repo ${q.repo}`);
  for (const f of q.files) if (!c.paths.includes(f)) throw new Error(`"${q.question}": ${f} is not in the ${q.repo} index`);
}
console.log(`${bench.questions.length} questions\n`);
if (process.env.FETCH_ONLY) process.exit(0);

const bm25 = new Map([...corpora].map(([repo, c]) => [repo, new Bm25(c.chunks)]));
const rerank = process.env.RERANKER ? await loadReranker(process.env.RERANKER) : null;

interface Row { config: string; variant: string; scores: Scores; perQuestion: Scores[]; byRepo: Record<string, Scores>; byKind: Record<string, Scores>; chunksPerSecond?: number }
const rows: Row[] = [];
const POOL = 50;

function addRow(config: string, variant: string, rank: (repo: string, q: Benchmark['questions'][number], qi: number) => Ranked[], extra: Partial<Row> = {}) {
  const perQuestion = bench.questions.map((q, i) => score(rank(q.repo, q, i), corpora.get(q.repo)!.chunks, q.files));
  const avg = (list: Scores[]) => Object.fromEntries(Object.keys(list[0]!).map((m) => [m, mean(list.map((s) => s[m]!))]));
  const byRepo = Object.fromEntries(bench.repos.map(({ repo }) => [repo, avg(perQuestion.filter((_, i) => bench.questions[i]!.repo === repo))]));
  const kinds = [...new Set(bench.questions.map((q) => q.kind ?? 'other'))];
  const byKind = Object.fromEntries(kinds.map((k) => [k, avg(perQuestion.filter((_, i) => (bench.questions[i]!.kind ?? 'other') === k))]));
  rows.push({ config, variant, scores: avg(perQuestion), perQuestion, byRepo, byKind, ...extra });
}

// BM25 alone doesn't depend on the model.
addRow('BM25 only', '-', (repo, q) => bm25.get(repo)!.search(q.question, POOL));

const rerankRanked = new Map<string, Ranked[]>();
const firstPlace: { label: string; counts: Record<string, number> }[] = [];
for (const variant of VARIANTS) {
  let model: Promise<(t: string[]) => Promise<Float32Array>> | null = null;
  const getModel = () => (model ??= loadModel(variant));
  const emb = new Map<string, Embedded>();
  let chunks = 0;
  let seconds = 0;
  for (const [repo, c] of corpora) {
    const qs = bench.questions.filter((q) => q.repo === repo).map((q) => q.question);
    const e = await embedCorpus(variant, c, qs, getModel);
    emb.set(repo, e);
    chunks += c.chunks.length;
    seconds += e.docSeconds;
  }
  const qIndex = (q: Benchmark['questions'][number]) => bench.questions.filter((x) => x.repo === q.repo).indexOf(q);
  for (const dim of [768, 256, 128]) {
    const docs = new Map([...emb].map(([repo, e]) => [repo, dim === FULL_DIM ? e.docs : prefix(e.docs, dim)]));
    const queries = new Map([...emb].map(([repo, e]) => [repo, dim === FULL_DIM ? e.queries : prefix(e.queries, dim)]));
    const dense = (repo: string, q: Benchmark['questions'][number]) => {
      const i = qIndex(q);
      return denseSearch(docs.get(repo)!, queries.get(repo)!.subarray(i * dim, (i + 1) * dim), POOL, dim, dim);
    };
    addRow(`Embeddings ${dim}d`, variant, (repo, q) => dense(repo, q), { chunksPerSecond: chunks / seconds });
    const hybrid = (repo: string, q: Benchmark['questions'][number]) => rrf([dense(repo, q), bm25.get(repo)!.search(q.question, POOL)]);
    addRow(`Hybrid ${dim}d (RRF)`, variant, hybrid);
    addRow(`Routed ${dim}d (hybrid if code-like)`, variant, (repo, q) => (looksLikeCode(q.question) ? hybrid(repo, q) : dense(repo, q)));
    // The reranker reorders the shipped ranking's own top 20, so the only difference from
    // "Embeddings 768d" is the reranker.
    if (rerank && dim === FULL_DIM && variant === VARIANTS[0]) {
      for (const q of bench.questions) {
        const c = corpora.get(q.repo)!;
        const pool = dense(q.repo, q).slice(0, 20);
        const s = await rerank(q.question, pool.map((r) => `${c.chunks[r.index]!.path}\n${c.chunks[r.index]!.text}`));
        rerankRanked.set(q.question, pool.map((r, i) => ({ index: r.index, score: s[i]! })).sort((a, b) => b.score - a.score));
      }
      addRow(`Embeddings ${dim}d + rerank top 20`, variant, (_repo, q) => rerankRanked.get(q.question)!);
      // What lands first: implementation code, tests, or prose (docs, README, changelog)?
      const kindOf = (path: string) =>
        /\.(md|mdx|rst|txt)$/i.test(path) || /(^|\/)docs?\//i.test(path)
          ? 'prose'
          : /(^|\/)(tests?|__tests__|spec|test-d)\/|[._-](test|spec)\./i.test(path)
            ? 'tests'
            : 'code';
      const firstKinds = (rank: (q: Benchmark['questions'][number]) => Ranked[]) => {
        const counts: Record<string, number> = { code: 0, tests: 0, prose: 0 };
        for (const q of bench.questions) counts[kindOf(corpora.get(q.repo)!.chunks[rank(q)[0]!.index]!.path)]!++;
        return counts;
      };
      firstPlace.push(
        { label: `Embeddings ${dim}d`, counts: firstKinds((q) => dense(q.repo, q)) },
        { label: `Embeddings ${dim}d + rerank top 20`, counts: firstKinds((q) => rerankRanked.get(q.question)!) },
      );
    }
  }
}

// ---- report ----

const pct = (x: number) => x.toFixed(2);
const header = `| Retriever | Model | recall@1 | recall@5 | recall@10 | hit@5 | MRR@10 |\n|---|---|---|---|---|---|---|`;
const lines = rows.map((r) => `| ${r.config} | ${r.variant} | ${pct(r.scores['recall@1']!)} | ${pct(r.scores['recall@5']!)} | ${pct(r.scores['recall@10']!)} | ${pct(r.scores['hit@5']!)} | ${pct(r.scores['MRR@10']!)} |`);
const repoHeader = `| Retriever | Model | ${bench.repos.map((r) => r.repo).join(' | ')} |\n|---|---|${bench.repos.map(() => '---').join('|')}|`;
const repoLines = rows.map((r) => `| ${r.config} | ${r.variant} | ${bench.repos.map(({ repo }) => pct(r.byRepo[repo]!['recall@5']!)).join(' | ')} |`);
const kinds = [...new Set(bench.questions.map((q) => q.kind ?? 'other'))];
const count = (k: string) => bench.questions.filter((q) => (q.kind ?? 'other') === k).length;
const kindHeader = `| Retriever | Model | ${kinds.map((k) => `${k} (${count(k)})`).join(' | ')} |
|---|---|${kinds.map(() => '---').join('|')}|`;
const kindLines = rows.map((r) => `| ${r.config} | ${r.variant} | ${kinds.map((k) => pct(r.byKind[k]!['recall@5']!)).join(' | ')} |`);
const speed = rows.filter((r) => r.chunksPerSecond).map((r) => `| ${r.variant} | ${r.chunksPerSecond!.toFixed(2)} |`);
const report = [
  `${bench.questions.length} questions over ${bench.repos.length} repos. Recall is over chunks (after overlap de-duplication).`,
  '',
  header,
  ...lines,
  '',
  'recall@5 by question kind:',
  '',
  kindHeader,
  ...kindLines,
  '',
  'recall@5 by repo:',
  '',
  repoHeader,
  ...repoLines,
  '',
  ...(firstPlace.length
    ? ['What each ranks first (questions whose top chunk is implementation code, tests, or prose):', '', '| Retriever | code | tests | prose |', '|---|---|---|---|', ...firstPlace.map((f) => `| ${f.label} | ${f.counts.code} | ${f.counts.tests} | ${f.counts.prose} |`), '']
    : []),
  ...(speed.length ? ['Embedding speed on this machine\'s CPU (chunks/s):', '', '| Model | chunks/s |', '|---|---|', ...[...new Set(speed)]] : []),
].join('\n');
console.log(`\n${report}`);
mkdirSync(join(evalDir, 'results'), { recursive: true });
const tag = process.env.TAG ?? VARIANTS.join('-');
writeFileSync(join(evalDir, 'results', `${tag}.md`), `${report}\n`);
writeJson(join(evalDir, 'results', `${tag}.json`), {
  date: new Date().toISOString().slice(0, 10),
  firstPlace,
  rows: rows.map((r) => ({ config: r.config, variant: r.variant, scores: r.scores, byRepo: r.byRepo, byKind: r.byKind, chunksPerSecond: r.chunksPerSecond, misses: bench.questions.filter((_, i) => r.perQuestion[i]!['hit@5'] === 0).map((q) => q.question) })),
});
