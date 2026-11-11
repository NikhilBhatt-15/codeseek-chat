// Pre-builds the indexes for the example repos, so the site can open them instantly.
//
//   npm run build:examples
//
// Writes public/examples/<owner>__<repo>.json (chunks + metadata + the retrieval results for
// each example question) and .bin (the vectors, float32). Embeddings come from the same
// EmbeddingGemma 2 model, settings and document format as the browser worker, run on the CPU.
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { AutoConfig, AutoModel, AutoTokenizer, env } from '@huggingface/transformers';
import { formatQuery } from '../src/embedder';
import { EXAMPLES } from '../src/examples';
import { fetchContents, fetchSnapshot, LIMITS, parseRepo } from '../src/github';
import { EMBED_DIM, MAX_TOKENS, MODEL_ID } from '../src/protocol';
import { buildIndex, chunkRepo, dedupeHits, embedChunks, search } from '../src/repoIndex';
import type { ExampleFile } from '../src/examples';

const root = process.env.CODESEEK_ROOT ?? join(dirname(fileURLToPath(import.meta.url)), '..');
const outDir = join(root, 'public', 'examples');
const bench = JSON.parse(readFileSync(join(root, 'eval', 'benchmark.json'), 'utf8')) as { repos: { repo: string; commit: string }[] };
if (process.env.HF_CACHE) env.cacheDir = process.env.HF_CACHE;

const config = await AutoConfig.from_pretrained(MODEL_ID);
Object.assign(config, { vision_config: null, audio_config: null });
const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
const model = await AutoModel.from_pretrained(MODEL_ID, { config, device: 'cpu', dtype: 'q4' });

async function embed(texts: string[]): Promise<Float32Array> {
  const inputs = tokenizer(texts, { padding: true, truncation: true, max_length: MAX_TOKENS });
  const out = (await model(inputs)) as { sentence_embedding: { slice(...a: unknown[]): any } };
  const vecs = out.sentence_embedding.slice(null, [0, EMBED_DIM]).normalize(2, -1);
  return Float32Array.from(vecs.data as ArrayLike<number>);
}

mkdirSync(outDir, { recursive: true });
for (const example of EXAMPLES) {
  // Pinned to the benchmark's commit when the repo is in it, so the eval's vectors can be reused.
  const pinned = bench.repos.find((r) => r.repo === example.repo)?.commit;
  const ref = { ...parseRepo(example.repo)!, ...(pinned ? { ref: pinned } : {}) };
  const snap = await fetchSnapshot(ref, LIMITS.quick, fetch, process.env.GITHUB_TOKEN);
  // The eval (npm run eval) caches the files and their full 768-dim q4 vectors for this commit.
  // Files download in parallel, so their order varies between fetches: reuse the eval's copy
  // of the files too, or the cached vectors would not line up with the chunks.
  const slug = example.repo.replace('/', '__');
  const evalContents = join(root, 'eval', '.cache', slug, `contents-${snap.commit.slice(0, 12)}.json`);
  const contents = existsSync(evalContents)
    ? new Map(Object.entries(JSON.parse(readFileSync(evalContents, 'utf8')) as Record<string, string | null>))
    : await fetchContents(snap, () => {});
  const chunks = chunkRepo(contents);
  const started = Date.now();
  const cached = join(root, 'eval', '.cache', 'embeddings', 'q4', slug);
  const meta = existsSync(`${cached}.json`) ? (JSON.parse(readFileSync(`${cached}.json`, 'utf8')) as { chunks: number }) : null;
  const reuse = EMBED_DIM === 768 && existsSync(evalContents) && meta?.chunks === chunks.length && existsSync(`${cached}.docs.bin`);
  const vectors = reuse
    ? new Float32Array(readFileSync(`${cached}.docs.bin`).buffer.slice(0))
    : await embedChunks(chunks, embed, (done, total) => {
        process.stdout.write(`\r${example.repo}: ${done}/${total} chunks`);
      });
  console.log(`${example.repo}: ${chunks.length} chunks in ${((Date.now() - started) / 1000).toFixed(0)} s${reuse ? ' (eval vectors reused)' : ''}`);
  const index = buildIndex(snap, 'quick', chunks, vectors, [...contents.keys()]);

  const questions = [];
  for (const q of example.questions) {
    const hits = dedupeHits(search(index, await embed([formatQuery(q.question)]), 16)).slice(0, 8);
    questions.push({ question: q.question, hits: hits.map((h) => ({ index: h.index, score: Number(h.score.toFixed(4)) })) });
    console.log(`  ${q.question}`);
    hits.forEach((h, i) => console.log(`    [${i + 1}] ${h.score.toFixed(3)} ${h.path}:${h.startLine}-${h.endLine}`));
  }

  const name = example.repo.replace('/', '__');
  const file: ExampleFile = {
    owner: index.owner,
    repo: index.repo,
    commit: index.commit,
    paths: index.paths,
    skipped: index.skipped,
    skippedBytes: index.skippedBytes,
    truncated: index.truncated,
    chunks: index.chunks,
    questions,
    builtAt: new Date().toISOString().slice(0, 10),
  };
  writeFileSync(join(outDir, `${name}.json`), JSON.stringify(file));
  writeFileSync(join(outDir, `${name}.bin`), new Uint8Array(index.vectors.buffer));
}
