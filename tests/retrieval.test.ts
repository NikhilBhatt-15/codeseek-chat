import { describe, expect, it } from 'vitest';
import { AgentTools, needsAgent, runAgent } from '../src/agent';
import { EMBED_DIM } from '../src/protocol';
import type { Hit, IndexedChunk } from '../src/repoIndex';
import { Bm25, denseSearch, hybridSearch, rrf, tokenize } from '../src/retrieval';

const chunk = (path: string, text: string, startLine = 1): IndexedChunk => ({
  path,
  text,
  startLine,
  endLine: startLine + text.split('\n').length - 1,
});

describe('tokenize', () => {
  it('splits identifiers but also keeps them whole', () => {
    expect(tokenize('retryOnTimeout')).toEqual(['retryontimeout', 'retry', 'timeout']); // 'on' is a stopword
    expect(tokenize('verify_signature(sig)')).toEqual(['verify_signature', 'verify', 'signature', 'sig']);
    expect(tokenize('HTTPError')).toEqual(['httperror', 'http', 'error']);
  });

  it('drops stopwords and stems plurals', () => {
    expect(tokenize('Where do retries happen?')).toEqual(['retry']);
    expect(tokenize('the headers')).toEqual(['header']);
  });
});

describe('BM25', () => {
  const docs = [
    chunk('src/retry.ts', 'export function calculateDelay(retryCount) { return 2 ** retryCount }'),
    chunk('src/timeout.ts', 'export function timeout(request, ms) { setTimeout(abort, ms) }'),
    chunk('README.md', 'A tiny HTTP client'),
  ];
  const bm25 = new Bm25(docs);

  it('ranks the chunk with the query terms first', () => {
    expect(bm25.search('how is the retry delay calculated', 3)[0]!.index).toBe(0);
    expect(bm25.search('timeout', 3).map((r) => r.index)).toEqual([1]);
  });

  it('matches file paths', () => {
    expect(bm25.search('retry.ts', 3)[0]!.index).toBe(0);
  });

  it('returns nothing for unknown words', () => {
    expect(bm25.search('kubernetes', 3)).toEqual([]);
  });
});

describe('fusion', () => {
  it('rewards documents ranked well by both lists', () => {
    const fused = rrf([
      [{ index: 1, score: 0.9 }, { index: 2, score: 0.8 }, { index: 3, score: 0.7 }],
      [{ index: 2, score: 12 }, { index: 3, score: 9 }],
    ]);
    // 2 is 2nd and 1st, 3 is 3rd and 2nd, 1 is only in one list.
    expect(fused.map((r) => r.index)).toEqual([2, 3, 1]);
  });

  it('dense search uses only the leading dimensions when asked', () => {
    // Row 0 matches in dim 0, row 1 matches in dim 1 (outside a 1-dim prefix).
    const vectors = new Float32Array([1, 0, 0, 1]);
    expect(denseSearch(vectors, new Float32Array([0, 1]), 2, 2, 2)[0]!.index).toBe(1);
    expect(denseSearch(vectors, new Float32Array([0.2, 1]), 2, 1, 2)[0]!.index).toBe(0);
  });

  it('hybrid search keeps cosine similarity for display', () => {
    const chunks = [chunk('a.ts', 'alpha'), chunk('b.ts', 'calculateDelay beta')];
    const vectors = new Float32Array(2 * EMBED_DIM);
    vectors[0] = 1; // a.ts is the embedding match
    vectors[EMBED_DIM + 1] = 1;
    const query = new Float32Array(EMBED_DIM);
    query[0] = 1;
    const hits = hybridSearch({ chunks, vectors }, new Bm25(chunks), 'calculateDelay', query, 2);
    expect(hits.map((h) => h.path).sort()).toEqual(['a.ts', 'b.ts']);
    expect(hits.find((h) => h.path === 'a.ts')!.similarity).toBe(1);
    expect(hits.find((h) => h.path === 'b.ts')!.keywordRank).toBe(1);
  });
});

describe('agent', () => {
  const chunks = [
    chunk('src/retry.ts', 'export function calculateDelay(n) {\n  return 2 ** n;\n}', 1),
    chunk('src/ky.ts', 'import {calculateDelay} from "./retry";\nconst wait = calculateDelay(count);', 10),
  ];
  const index = { owner: 'o', repo: 'r', chunks };
  const search = (_q: string, k: number): Hit[] => [{ ...chunks[0]!, index: 0, score: 0.9 }].slice(0, k);

  it('routes reference-style questions to the agent', () => {
    expect(needsAgent('Where is calculateDelay called?')).toBe(true);
    expect(needsAgent('How are timeouts implemented?')).toBe(false);
  });

  it('finds whole-word references and numbers their chunks', () => {
    const tools = new AgentTools(index, search);
    const out = tools.findReferences('calculateDelay()');
    expect(out.count).toBe(3);
    expect(out.text).toContain('[1] src/retry.ts:1:');
    expect(out.text).toContain('[2] src/ky.ts:11: const wait = calculateDelay(count);');
    expect(tools.findReferences('calculate').count).toBe(0); // whole words only
  });

  it('reads line ranges back from the chunks', () => {
    const tools = new AgentTools(index, search);
    const out = tools.readLines('src/ky.ts', 11, 50);
    expect(out.text).toBe('[1] src/ky.ts (lines 11-11)\nconst wait = calculateDelay(count);');
  });

  it('runs tools until the model answers, keeping citations stable', async () => {
    const bodies: { contents: { role: string; parts: Record<string, unknown>[] }[] }[] = [];
    const replies = [
      { parts: [{ functionCall: { name: 'search_code', args: { query: 'retry delay' } }, thoughtSignature: 'sig' }] },
      { parts: [{ functionCall: { name: 'find_references', args: { symbol: 'calculateDelay' } } }] },
      { parts: [{ text: 'It is defined in retry.ts [1] and called from ky.ts [2].' }] },
    ];
    const fake = (async (_url: string, init: RequestInit) => {
      bodies.push(JSON.parse(String(init.body)));
      return new Response(JSON.stringify({ candidates: [{ content: { role: 'model', ...replies[bodies.length - 1] } }] }));
    }) as typeof fetch;
    const steps: string[] = [];
    const result = await runAgent({ apiKey: 'k', model: 'm', index, search, question: 'Where is calculateDelay called?', fetchImpl: fake, onStep: (s) => steps.push(s.label) });
    expect(result.answer).toContain('[2]');
    expect(result.sources.map((s) => s.path)).toEqual(['src/retry.ts', 'src/ky.ts']);
    expect(steps).toEqual(['Searched “retry delay”', 'Found references to calculateDelay']);
    // The model's turn (with its thought signature) and the tool result go back verbatim.
    const second = bodies[1]!.contents;
    expect(second.at(-2)!.parts[0]!.thoughtSignature).toBe('sig');
    expect(JSON.stringify(second.at(-1))).toContain('functionResponse');
  });

  it('forces an answer when the step budget runs out', async () => {
    let calls = 0;
    const fake = (async (_url: string, init: RequestInit) => {
      calls++;
      const mode = JSON.parse(String(init.body)).toolConfig.functionCallingConfig.mode;
      const parts = mode === 'NONE' ? [{ text: 'done' }] : [{ functionCall: { name: 'search_code', args: { query: 'x' } } }];
      return new Response(JSON.stringify({ candidates: [{ content: { parts } }] }));
    }) as typeof fetch;
    const result = await runAgent({ apiKey: 'k', model: 'm', index, search, question: 'q', maxSteps: 2, fetchImpl: fake });
    expect(result.answer).toBe('done');
    expect(calls).toBe(3);
    expect(result.steps).toHaveLength(2);
  });
});
