// Answer-quality benchmark: the offline model (Qwen3.5 0.8B) vs Gemini on the same 64 questions,
// given the same retrieved excerpts (top 8, EmbeddingGemma 2 q4, 768 dims, as the app does).
//
//   npm run eval:answers                    # offline model only (CPU, ~30 min, cached)
//   $env:GEMINI_API_KEY = "<key>"; npm run eval:answers    # + Gemini answers + judge scores
//   JUDGE_BASE_URL=https://api.groq.com/openai/v1 JUDGE_API_KEY=<key> npm run eval:answers
//                                           # judge with gpt-oss-120b instead (OpenAI-compatible)
//
// Needs `npm run eval` first (it caches the corpora and embeddings this reuses).
//
// Scores:
//   cites right file  answer cites at least one excerpt from a labelled answer file
//                     (only over questions where retrieval put that file in the top 8)
//   valid citations   every [n] in the answer points at an excerpt that exists
//   judge (1-5)       Gemini grades correctness and grounding against the excerpts; needs a key
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { env, pipeline, type TextGenerationPipeline } from '@huggingface/transformers';
import { LOCAL_LLM_ID, LOCAL_LLM_LABEL } from '../src/llmProtocol';
import { systemPrompt, userPrompt } from '../src/prompt';
import { chunkRepo, dedupeHits, type Hit, type IndexedChunk } from '../src/repoIndex';
import { denseSearch } from '../src/retrieval';

interface Question { repo: string; question: string; files: string[]; kind: string }
const root = process.env.CODESEEK_ROOT ?? join(dirname(fileURLToPath(import.meta.url)), '..');
const evalDir = join(root, 'eval');
const cache = join(evalDir, '.cache');
if (process.env.HF_CACHE) env.cacheDir = process.env.HF_CACHE;
const bench = JSON.parse(readFileSync(join(evalDir, 'benchmark.json'), 'utf8')) as { repos: { repo: string; commit: string }[]; questions: Question[] };
const KEY = process.env.GEMINI_API_KEY ?? '';
const GEMINI_MODEL = process.env.GEMINI_MODEL ?? 'gemini-flash-lite-latest';
// The judge defaults to Gemini Flash. Any OpenAI-compatible API works instead (for example
// gpt-oss-120b on Groq): set JUDGE_BASE_URL, JUDGE_API_KEY and JUDGE_MODEL. A judge from another
// model family avoids Gemini grading Gemini's own answers.
const JUDGE_BASE_URL = (process.env.JUDGE_BASE_URL ?? '').replace(/\/+$/, '');
const JUDGE_API_KEY = process.env.JUDGE_API_KEY ?? '';
const JUDGE_MODEL = process.env.JUDGE_MODEL ?? (JUDGE_BASE_URL ? 'openai/gpt-oss-120b' : 'gemini-flash-latest');
const CAN_JUDGE = JUDGE_BASE_URL ? Boolean(JUDGE_API_KEY) : Boolean(KEY);
const RETRYABLE = new Set([429, 500, 502, 503, 504]);
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const DIM = 768;
const slug = (r: string) => r.replace('/', '__');
const readJson = <T>(p: string, fallback: T): T => (existsSync(p) ? (JSON.parse(readFileSync(p, 'utf8')) as T) : fallback);
const writeJson = (p: string, v: unknown) => {
  mkdirSync(dirname(p), { recursive: true });
  writeFileSync(p, JSON.stringify(v, null, 1));
};

// ---- retrieval, exactly as the app: top 16 by cosine, drop overlaps, keep 8 ----

const retrieved = new Map<string, Hit[]>();
for (const { repo, commit } of bench.repos) {
  const contents = readJson<Record<string, string | null> | null>(join(cache, slug(repo), `contents-${commit.slice(0, 12)}.json`), null);
  const base = join(cache, 'embeddings', 'q4', slug(repo));
  if (!contents || !existsSync(`${base}.docs.bin`)) throw new Error(`Run "npm run eval" first: no cached index for ${repo}.`);
  const chunks: IndexedChunk[] = chunkRepo(new Map(Object.entries(contents)));
  const docs = new Float32Array(readFileSync(`${base}.docs.bin`).buffer.slice(0));
  const queries = new Float32Array(readFileSync(`${base}.queries.bin`).buffer.slice(0));
  const qs = bench.questions.filter((q) => q.repo === repo);
  qs.forEach((q, i) => {
    const ranked = denseSearch(docs, queries.subarray(i * DIM, (i + 1) * DIM), 16, DIM, DIM);
    retrieved.set(q.question, dedupeHits(ranked.map((r) => ({ ...chunks[r.index]!, index: r.index, score: r.score }))).slice(0, 8));
  });
}

// ---- answering ----

type Answers = Record<string, { answer: string; seconds: number; tokens?: number }>;

async function answerLocal(): Promise<Answers> {
  const path = join(cache, 'answers', 'local.json');
  const done = readJson<Answers>(path, {});
  const todo = bench.questions.filter((q) => !done[q.question]);
  if (!todo.length) return done;
  const generator = (await pipeline('text-generation', LOCAL_LLM_ID, { device: 'cpu', dtype: 'q4' })) as TextGenerationPipeline;
  for (const [i, q] of todo.entries()) {
    const hits = retrieved.get(q.question)!;
    const t0 = performance.now();
    const out = (await generator(
      [
        { role: 'system', content: systemPrompt(q.repo) },
        { role: 'user', content: userPrompt(q.question, hits) },
      ],
      { max_new_tokens: 400, do_sample: false, repetition_penalty: 1.1 } as Record<string, unknown>,
    )) as { generated_text: { role: string; content: string }[] }[];
    const answer = out[0]!.generated_text.at(-1)!.content.trim();
    const tokens = generator.tokenizer.encode(answer).length;
    done[q.question] = { answer, seconds: (performance.now() - t0) / 1000, tokens };
    writeJson(path, done);
    console.log(`local ${i + 1}/${todo.length}: ${(tokens / done[q.question]!.seconds).toFixed(1)} tok/s  ${q.question.slice(0, 60)}`);
  }
  return done;
}

async function gemini(model: string, body: unknown): Promise<string> {
  for (let attempt = 0; ; attempt++) {
    const res = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'x-goog-api-key': KEY },
      body: JSON.stringify(body),
    });
    if (RETRYABLE.has(res.status) && attempt < 8) {
      // Rate limit or "high demand": back off and retry (15 s, 30 s, 45 s, ... up to 2 min).
      await sleep(Math.min(120000, 15000 * (attempt + 1)));
      continue;
    }
    if (!res.ok) throw new Error(`Gemini ${res.status}: ${(await res.text()).slice(0, 200)}`);
    const data = (await res.json()) as { candidates?: { content?: { parts?: { text?: string; thought?: boolean }[] } }[] };
    return (data.candidates?.[0]?.content?.parts ?? []).filter((p) => !p.thought).map((p) => p.text ?? '').join('').trim();
  }
}

async function answerGemini(): Promise<Answers> {
  const path = join(cache, 'answers', `${GEMINI_MODEL}.json`);
  const done = readJson<Answers>(path, {});
  for (const [i, q] of bench.questions.entries()) {
    if (done[q.question]) continue;
    const t0 = performance.now();
    const answer = await gemini(GEMINI_MODEL, {
      systemInstruction: { parts: [{ text: systemPrompt(q.repo) }] },
      contents: [{ role: 'user', parts: [{ text: userPrompt(q.question, retrieved.get(q.question)!) }] }],
      generationConfig: { temperature: 0.2 },
    });
    done[q.question] = { answer, seconds: (performance.now() - t0) / 1000 };
    writeJson(path, done);
    console.log(`${GEMINI_MODEL} ${i + 1}/${bench.questions.length}`);
  }
  return done;
}

// ---- scoring ----

const citations = (answer: string) =>
  [...answer.matchAll(/\[(\d{1,2}(?:\s*,\s*\d{1,2})*)\]/g)].flatMap((m) => m[1]!.split(',').map((n) => Number(n.trim())));

function autoScores(answers: Answers) {
  let eligible = 0;
  let citesRight = 0;
  let valid = 0;
  let cited = 0;
  for (const q of bench.questions) {
    const hits = retrieved.get(q.question)!;
    const a = answers[q.question]?.answer ?? '';
    const cites = citations(a);
    if (cites.length) cited++;
    if (cites.every((n) => n >= 1 && n <= hits.length)) valid++;
    if (hits.some((h) => q.files.includes(h.path))) {
      eligible++;
      if (cites.some((n) => hits[n - 1] && q.files.includes(hits[n - 1]!.path))) citesRight++;
    }
  }
  const n = bench.questions.length;
  return { citesRight: citesRight / eligible, eligible, validCitations: valid / n, anyCitation: cited / n };
}

/** OpenAI-compatible chat completion that returns JSON. */
async function openaiJson(prompt: string): Promise<string> {
  for (let attempt = 0; ; attempt++) {
    const res = await fetch(`${JUDGE_BASE_URL}/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${JUDGE_API_KEY}` },
      body: JSON.stringify({
        model: JUDGE_MODEL,
        temperature: 0,
        response_format: { type: 'json_object' },
        messages: [{ role: 'user', content: `${prompt}\n\nReply with JSON only: {"score": <1-5>, "reason": "<one sentence>"}` }],
      }),
    });
    if (RETRYABLE.has(res.status) && attempt < 8) {
      await sleep(Math.min(120000, 15000 * (attempt + 1)));
      continue;
    }
    if (!res.ok) throw new Error(`Judge ${res.status}: ${(await res.text()).slice(0, 200)}`);
    const data = (await res.json()) as { choices?: { message?: { content?: string } }[] };
    return data.choices?.[0]?.message?.content ?? '{}';
  }
}

async function judge(system: string, answers: Answers): Promise<Record<string, number>> {
  const path = join(cache, 'answers', `judge-${JUDGE_MODEL.replace(/[^\w.-]+/g, '_')}-${system}.json`);
  const done = readJson<Record<string, number>>(path, {});
  for (const q of bench.questions) {
    if (done[q.question] !== undefined) continue;
    const prompt = [
      'You grade answers to questions about a code repository. The answer was written from the numbered excerpts below.',
      'Score 1-5: 5 = correct, complete and grounded in the excerpts; 4 = correct with minor gaps; 3 = partly correct;',
      '2 = mostly wrong or ungrounded; 1 = wrong, invented, or no answer. Judge correctness against the excerpts, not style.',
      `The file(s) that truly answer it: ${q.files.join(', ')}.`,
      '',
      userPrompt(q.question, retrieved.get(q.question)!),
      '',
      `Answer to grade:\n${answers[q.question]?.answer ?? '(none)'}`,
    ].join('\n');
    const text = JUDGE_BASE_URL
      ? await openaiJson(prompt)
      : await gemini(JUDGE_MODEL, {
          contents: [{ role: 'user', parts: [{ text: prompt }] }],
          generationConfig: {
            temperature: 0,
            responseMimeType: 'application/json',
            responseSchema: { type: 'OBJECT', properties: { score: { type: 'INTEGER' }, reason: { type: 'STRING' } }, required: ['score'] },
          },
        });
    const match = text.match(/\{[\s\S]*\}/);
    done[q.question] = Math.max(1, Math.min(5, Number((JSON.parse(match ? match[0] : text) as { score: number }).score) || 1));
    writeJson(path, done);
  }
  return done;
}

// ---- run ----

const systems: { name: string; answers: Answers }[] = [{ name: `${LOCAL_LLM_LABEL} (offline, q4)`, answers: await answerLocal() }];
const geminiCache = readJson<Answers>(join(cache, 'answers', `${GEMINI_MODEL}.json`), {});
if (KEY || bench.questions.every((q) => geminiCache[q.question])) {
  systems.push({ name: `${GEMINI_MODEL} (API)`, answers: KEY ? await answerGemini() : geminiCache });
} else console.log('\nNo GEMINI_API_KEY: skipping Gemini answers.');
if (!CAN_JUDGE) console.log('No judge key: skipping judge scores.');

const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / Math.max(1, xs.length);
const rows: string[] = [];
const json: Record<string, unknown>[] = [];
for (const s of systems) {
  const auto = autoScores(s.answers);
  const list = Object.values(s.answers);
  const tps = list.filter((a) => a.tokens).map((a) => a.tokens! / a.seconds);
  const scores = CAN_JUDGE ? Object.values(await judge(s.name.split(' ')[0]!, s.answers)) : [];
  const judged = scores.length ? `${mean(scores).toFixed(2)} | ${Math.round((scores.filter((x) => x >= 4).length / scores.length) * 100)}%` : '- | -';
  rows.push(`| ${s.name} | ${Math.round(auto.citesRight * 100)}% | ${Math.round(auto.validCitations * 100)}% | ${judged} | ${tps.length ? mean(tps).toFixed(1) : '-'} |`);
  json.push({ system: s.name, ...auto, judgeMean: scores.length ? mean(scores) : null, judgeAtLeast4: scores.length ? scores.filter((x) => x >= 4).length / scores.length : null, cpuTokensPerSecond: tps.length ? mean(tps) : null });
}
const report = [
  `${bench.questions.length} questions; every system gets the same top-8 excerpts. Judge: ${CAN_JUDGE ? JUDGE_MODEL : 'not run (no key)'}.`,
  '',
  '| System | Cites the right file | Valid citations | Judge mean (1-5) | Judged ≥4 | CPU tokens/s |',
  '|---|---|---|---|---|---|',
  ...rows,
].join('\n');
console.log(`\n${report}`);
mkdirSync(join(evalDir, 'results'), { recursive: true });
writeFileSync(join(evalDir, 'results', 'answers.md'), `${report}\n`);
writeJson(join(evalDir, 'results', 'answers.json'), { date: new Date().toISOString().slice(0, 10), judge: CAN_JUDGE ? JUDGE_MODEL : null, systems: json });
