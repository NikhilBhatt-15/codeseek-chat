import { describe, expect, it } from 'vitest';
import { parseSseEvents, streamChat, textOfChunk } from '../src/gemini';
import { linkCitations, retrievalQuery, userPrompt } from '../src/prompt';
import { EMBED_DIM } from '../src/protocol';
import { buildIndex, chunkRepo, dedupeHits, embedChunks, estimateChunks, missingFiles, search, type Hit } from '../src/repoIndex';

const unit = (dim: number) => {
  const v = new Float32Array(EMBED_DIM);
  v[dim] = 1;
  return v;
};

describe('index', () => {
  it('embeds in length-sorted batches but keeps rows aligned with chunks', async () => {
    const chunks = chunkRepo(new Map([['a.py', 'x = 1'], ['b.py', 'y = 22222222222'], ['c.bin', null]]));
    expect(chunks.map((c) => c.path)).toEqual(['a.py', 'b.py']);
    // Fake embedder: dimension chosen by the text, so we can check where each row landed.
    const embed = async (texts: string[]) => {
      const out = new Float32Array(texts.length * EMBED_DIM);
      texts.forEach((t, i) => out.set(unit(t.includes('a.py') ? 0 : 1), i * EMBED_DIM));
      return out;
    };
    const progress: number[] = [];
    const vectors = await embedChunks(chunks, embed, (d) => progress.push(d), undefined, 1);
    expect(vectors[0]).toBe(1); // row 0 (a.py) -> dim 0
    expect(vectors[EMBED_DIM + 1]).toBe(1); // row 1 (b.py) -> dim 1
    expect(progress).toEqual([1, 2]);

    const hits = search({ chunks, vectors }, unit(1), 1);
    expect(hits[0]).toMatchObject({ path: 'b.py', score: 1 });
  });

  it('drops hits overlapping a better one in the same file', () => {
    const h = (path: string, s: number, e: number, score: number): Hit => ({ path, startLine: s, endLine: e, score, text: '', index: 0 });
    const kept = dedupeHits([h('a', 1, 20, 0.9), h('a', 16, 35, 0.8), h('b', 1, 20, 0.7), h('a', 40, 50, 0.6)]);
    expect(kept.map((k) => [k.path, k.startLine])).toEqual([['a', 1], ['b', 1], ['a', 40]]);
  });
});

describe('prompt', () => {
  it('numbers sources and fences code safely', () => {
    const hit: Hit = { path: 'x.md', startLine: 1, endLine: 3, text: 'a\n```\nb', score: 1, index: 0 };
    const p = userPrompt('why?', [hit]);
    expect(p).toContain('[1] x.md (lines 1-3)');
    expect(p).toContain('````markdown\na\n```\nb\n````');
    expect(p.endsWith('Question: why?')).toBe(true);
  });

  it('adds the previous question only for short follow-ups', () => {
    expect(retrievalQuery('and where is it called?', 'how is auth done')).toBe('how is auth done\nand where is it called?');
    expect(retrievalQuery('how does the chunker decide where to split files', 'x')).toBe(
      'how does the chunker decide where to split files',
    );
  });

  it('links citations but not code, indexes or unknown numbers', () => {
    const url = (n: number) => (n <= 2 ? `https://g/${n}` : null);
    const md = 'Done in `arr[1]` here [1][2], see x[1] and [3].\n```\nlist[2]\n```\n[link](u) [1](u)';
    expect(linkCitations(md, url)).toBe(
      'Done in `arr[1]` here [[1]](https://g/1)[[2]](https://g/2), see x[1] and [3].\n```\nlist[2]\n```\n[link](u) [1](u)',
    );
  });
});

describe('gemini streaming', () => {
  it('parses SSE events across chunk boundaries', () => {
    const first = parseSseEvents('data: {"a":1}\n\ndata: {"b"');
    expect(first).toEqual({ events: ['{"a":1}'], rest: 'data: {"b"' });
    expect(parseSseEvents(`${first.rest}:2}\r\n\r\n`).events).toEqual(['{"b":2}']);
  });

  it('skips thought parts', () => {
    const payload = { candidates: [{ content: { parts: [{ text: 'hmm', thought: true }, { text: 'Hi' }] } }] };
    expect(textOfChunk(payload)).toBe('Hi');
  });

  it('streams text and sends the key in a header, not the URL', async () => {
    let seen: { url: string; init: RequestInit } | null = null;
    const sse = ['Hel', 'lo'].map((t) => `data: ${JSON.stringify({ candidates: [{ content: { parts: [{ text: t }] } }] })}\n\n`).join('');
    const fake = (async (url: string, init: RequestInit) => {
      seen = { url, init };
      return new Response(sse);
    }) as unknown as typeof fetch;
    const deltas: string[] = [];
    const full = await streamChat({ apiKey: 'k', model: 'm', system: 's', turns: [{ role: 'user', text: 'q' }], onText: (d) => deltas.push(d), fetchImpl: fake });
    expect(full).toBe('Hello');
    expect(deltas).toEqual(['Hel', 'lo']);
    expect(seen!.url).not.toContain('k');
    expect((seen!.init.headers as Record<string, string>)['x-goog-api-key']).toBe('k');
  });

  it('turns API errors into readable messages', async () => {
    const fake = (async () =>
      new Response(JSON.stringify({ error: { message: 'API key not valid.' } }), { status: 400 })) as unknown as typeof fetch;
    await expect(
      streamChat({ apiKey: 'bad', model: 'm', system: 's', turns: [], onText: () => {}, fetchImpl: fake }),
    ).rejects.toThrow(/key was rejected/);
  });
});

describe('extending a quick index', () => {
  const snap = (files: string[], skipped = 0) => ({
    owner: 'o', repo: 'r', branch: null, commit: 'c', truncated: false, skipped, skippedBytes: skipped * 1000,
    files: files.map((path) => ({ path, size: 10 })),
  });
  const vec = (n: number, fill: number) => new Float32Array(n * EMBED_DIM).fill(fill);
  const chunk = (path: string) => ({ path, startLine: 1, endLine: 1, text: path });

  it('adds only the missing files and keeps rows aligned', () => {
    const quick = buildIndex(snap(['a.py', 'b.py'], 2), 'quick', [chunk('a.py'), chunk('b.py')], vec(2, 1), ['a.py', 'b.py']);
    expect(quick).toMatchObject({ size: 'quick', skipped: 2, skippedBytes: 2000 });

    const full = snap(['a.py', 'b.py', 'c.py', 'empty.py']);
    expect(missingFiles(full, quick).map((f) => f.path)).toEqual(['c.py', 'empty.py']);

    const extended = buildIndex(full, 'full', [chunk('c.py')], vec(1, 2), ['c.py', 'empty.py'], quick);
    expect(extended).toMatchObject({ size: 'full', skipped: 0, key: quick.key });
    expect(extended.paths).toEqual(['a.py', 'b.py', 'c.py', 'empty.py']);
    expect(extended.chunks.map((c) => c.path)).toEqual(['a.py', 'b.py', 'c.py']);
    expect(extended.vectors.length).toBe(3 * EMBED_DIM);
    expect(extended.vectors[2 * EMBED_DIM]).toBe(2); // c.py's row follows the original two
    expect(missingFiles(full, extended)).toEqual([]);
  });

  it('estimates about one chunk per KB', () => {
    expect(estimateChunks(1_161_709)).toBe(1162); // ky: 1,106 chunks measured
    expect(estimateChunks(10)).toBe(1);
  });
});
