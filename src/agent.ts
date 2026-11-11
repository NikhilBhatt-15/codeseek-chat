// Multi-step retrieval: Gemini drives three tools over the in-browser index, so it can search,
// read the result, then follow a symbol to where it is defined or called before answering.
//
// One search is enough for "how does X work". Questions like "where is this called?" or
// "what happens after Y?" need a second hop that a single similarity lookup can't make.
import { GeminiError } from './gemini';
import type { Hit, IndexedChunk } from './repoIndex';

export interface AgentIndex {
  owner: string;
  repo: string;
  chunks: IndexedChunk[];
}

export interface AgentStep {
  tool: 'search_code' | 'find_references' | 'read_lines';
  label: string; // human-readable, e.g. Searched "retry delay"
  results: number; // how many things came back
}

export interface AgentResult {
  answer: string;
  sources: Hit[]; // [n] in the answer cites sources[n - 1]
  steps: AgentStep[];
}

type Search = (query: string, k: number) => Hit[] | Promise<Hit[]>;

interface Part {
  text?: string;
  thought?: boolean;
  functionCall?: { name: string; args?: Record<string, unknown>; id?: string };
  [key: string]: unknown;
}
interface Content {
  role: 'user' | 'model';
  parts: Part[];
}

const TOOLS = [
  {
    functionDeclarations: [
      {
        name: 'search_code',
        description:
          'Semantic + keyword search over the repository. Returns the best-matching code excerpts, each with a source number.',
        parameters: {
          type: 'OBJECT',
          properties: { query: { type: 'STRING', description: 'What to look for, in words or identifiers.' } },
          required: ['query'],
        },
      },
      {
        name: 'find_references',
        description:
          'Exact, whole-word search for an identifier (function, class, variable, option name). Lists every line that mentions it, so you can find where it is defined and where it is called or used.',
        parameters: {
          type: 'OBJECT',
          properties: { symbol: { type: 'STRING', description: 'The identifier, e.g. calculateDelay or verify_signature.' } },
          required: ['symbol'],
        },
      },
      {
        name: 'read_lines',
        description: 'Reads a line range of an indexed file, to see more context around a result. At most 120 lines.',
        parameters: {
          type: 'OBJECT',
          properties: {
            path: { type: 'STRING' },
            start_line: { type: 'INTEGER' },
            end_line: { type: 'INTEGER' },
          },
          required: ['path', 'start_line', 'end_line'],
        },
      },
    ],
  },
];

export function agentSystemPrompt(repo: string, maxSteps: number): string {
  return [
    `You are a senior engineer answering questions about the GitHub repository ${repo}.`,
    'You can only see the code through your tools. Every result is numbered like [3].',
    'Strategy: start with search_code. When the question is about where something is called, used, defined or what happens next,',
    'take the identifier from the results and use find_references to follow it, then read_lines if you need more context.',
    `Use at most ${maxSteps} tool calls, then answer.`,
    'Answer from the tool results only. Cite them inline as [1] or [1][3] right after the claims they support, never [1, 3].',
    'If the results do not contain the answer, say so plainly. Do not invent files, functions or behaviour.',
    'Be concise. Use Markdown, and fenced code blocks with a language tag for code.',
  ].join('\n');
}

/** Whether a question likely needs more than one retrieval hop. */
export const needsAgent = (question: string) =>
  /\b(call(s|ed|er|ers|ing)?|used|uses|usage|references?|invok\w*|trace|flow|end[- ]to[- ]end|after|before|follow|who|chain|path from|step by step)\b/i.test(question);

const escapeRegExp = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

export class AgentTools {
  readonly sources: Hit[] = [];
  private byKey = new Map<string, number>();

  constructor(
    private index: AgentIndex,
    private search: Search,
  ) {}

  /** Registers a source and returns its 1-based citation number. */
  private cite(hit: Hit): number {
    const key = `${hit.path}:${hit.startLine}-${hit.endLine}`;
    const known = this.byKey.get(key);
    if (known) return known;
    this.sources.push(hit);
    this.byKey.set(key, this.sources.length);
    return this.sources.length;
  }

  async searchCode(query: string): Promise<{ text: string; count: number }> {
    const hits = await this.search(query, 6);
    const text = hits
      .map((h) => `[${this.cite(h)}] ${h.path} (lines ${h.startLine}-${h.endLine})\n${h.text}`)
      .join('\n\n');
    return { text: text || 'No results.', count: hits.length };
  }

  findReferences(symbol: string): { text: string; count: number } {
    const name = symbol.trim().replace(/[()]/g, '').split(/[.\s]/).filter(Boolean).pop() ?? '';
    if (!name) return { text: 'Give an identifier.', count: 0 };
    const re = new RegExp(`(^|[^A-Za-z0-9_$])${escapeRegExp(name)}(?![A-Za-z0-9_$])`);
    const seen = new Set<string>();
    const lines: string[] = [];
    let count = 0;
    for (let i = 0; i < this.index.chunks.length; i++) {
      const c = this.index.chunks[i]!;
      const rows = c.text.split('\n');
      let n = 0;
      rows.forEach((row, j) => {
        const line = c.startLine + j;
        const key = `${c.path}:${line}`;
        if (!re.test(row) || seen.has(key)) return;
        seen.add(key);
        count++;
        if (lines.length >= 40) return;
        if (n++ === 0) this.cite({ ...c, index: i, score: 0 });
        lines.push(`[${this.byKey.get(`${c.path}:${c.startLine}-${c.endLine}`)}] ${c.path}:${line}: ${row.trim().slice(0, 160)}`);
      });
    }
    if (!count) return { text: `No lines mention ${name}.`, count };
    const more = count > lines.length ? `\n(${count - lines.length} more not shown)` : '';
    return { text: `${count} line(s) mention ${name}:\n${lines.join('\n')}${more}`, count };
  }

  readLines(path: string, start: number, end: number): { text: string; count: number } {
    const chunks = this.index.chunks.map((c, i) => ({ c, i })).filter(({ c }) => c.path === path);
    if (!chunks.length) return { text: `${path} is not in the index.`, count: 0 };
    const lines = new Map<number, string>();
    for (const { c } of chunks) c.text.split('\n').forEach((row, j) => lines.set(c.startLine + j, row));
    const from = Math.max(1, Math.floor(start) || 1);
    const to = Math.min(Math.max(from, Math.floor(end) || from), from + 119);
    const rows: string[] = [];
    for (let l = from; l <= to; l++) if (lines.has(l)) rows.push(lines.get(l)!);
    if (!rows.length) return { text: `No indexed lines in ${path} between ${from} and ${to}.`, count: 0 };
    const last = from + rows.length - 1;
    const n = this.cite({ path, startLine: from, endLine: last, text: rows.join('\n'), index: chunks[0]!.i, score: 0 });
    return { text: `[${n}] ${path} (lines ${from}-${last})\n${rows.join('\n')}`, count: rows.length };
  }
}

function friendly(status: number, body: string): string {
  let message = body;
  try {
    message = (JSON.parse(body) as { error?: { message?: string } }).error?.message ?? body;
  } catch {
    // not JSON
  }
  if (status === 400 && /API key/i.test(message)) return 'That Gemini API key was rejected. Check it in Settings.';
  if (status === 429) return 'Gemini rate limit reached for this key. Wait a minute and try again.';
  return `Gemini error ${status}: ${message}`;
}

export async function runAgent(opts: {
  apiKey: string;
  model: string;
  index: AgentIndex;
  search: Search;
  question: string;
  history?: { role: 'user' | 'model'; text: string }[];
  maxSteps?: number;
  onStep?: (step: AgentStep) => void;
  signal?: AbortSignal;
  fetchImpl?: typeof fetch;
}): Promise<AgentResult> {
  const { apiKey, model, index, question, maxSteps = 5, onStep, signal, fetchImpl = fetch } = opts;
  const tools = new AgentTools(index, opts.search);
  const steps: AgentStep[] = [];
  const contents: Content[] = [
    ...(opts.history ?? []).map((t) => ({ role: t.role, parts: [{ text: t.text }] })),
    { role: 'user', parts: [{ text: question }] },
  ];
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent`;

  for (let round = 0; ; round++) {
    const final = round >= maxSteps; // out of budget: force a written answer
    const res = await fetchImpl(url, {
      method: 'POST',
      signal,
      headers: { 'Content-Type': 'application/json', 'x-goog-api-key': apiKey },
      body: JSON.stringify({
        systemInstruction: { parts: [{ text: agentSystemPrompt(`${index.owner}/${index.repo}`, maxSteps) }] },
        contents,
        tools: TOOLS,
        toolConfig: { functionCallingConfig: { mode: final ? 'NONE' : 'AUTO' } },
        generationConfig: { temperature: 0.2 },
      }),
    });
    if (!res.ok) throw new GeminiError(friendly(res.status, await res.text()), res.status);
    const data = (await res.json()) as { candidates?: { content?: Content }[] };
    const parts = data.candidates?.[0]?.content?.parts ?? [];
    const calls = parts.filter((p) => p.functionCall);
    if (!calls.length || final) {
      const answer = parts.filter((p) => !p.thought && p.text).map((p) => p.text).join('');
      return { answer: answer || 'The model returned no answer.', sources: tools.sources, steps };
    }
    // Send the model's turn back unchanged (it may carry thought signatures), then the results.
    contents.push({ role: 'model', parts });
    const responses: Part[] = [];
    for (const { functionCall } of calls) {
      const { name, args = {}, id } = functionCall!;
      let out: { text: string; count: number };
      let step: AgentStep;
      if (name === 'search_code') {
        const q = String(args.query ?? '');
        out = await tools.searchCode(q);
        step = { tool: 'search_code', label: `Searched “${q}”`, results: out.count };
      } else if (name === 'find_references') {
        const s = String(args.symbol ?? '');
        out = tools.findReferences(s);
        step = { tool: 'find_references', label: `Found references to ${s}`, results: out.count };
      } else if (name === 'read_lines') {
        const p = String(args.path ?? '');
        out = tools.readLines(p, Number(args.start_line), Number(args.end_line));
        step = { tool: 'read_lines', label: `Read ${p}:${Number(args.start_line)}-${Number(args.end_line)}`, results: out.count };
      } else {
        out = { text: `Unknown tool ${name}.`, count: 0 };
        step = { tool: 'search_code', label: `Unknown tool ${name}`, results: 0 };
      }
      steps.push(step);
      onStep?.(step);
      responses.push({ functionResponse: { name, ...(id ? { id } : {}), response: { result: out.text } } });
    }
    contents.push({ role: 'user', parts: responses });
  }
}
