// Split source files into chunks that follow code structure.
// Port of codeseek's chunking.py so the CLI and the web app index code the same way.

export interface Chunk {
  startLine: number; // 1-based, inclusive
  endLine: number; // 1-based, inclusive
  text: string;
}

export interface ChunkOptions {
  maxLines: number;
  maxChars: number;
  overlapLines: number;
}

export const DEFAULT_CHUNKING: ChunkOptions = { maxLines: 50, maxChars: 1500, overlapLines: 5 };

const DECL_RE = new RegExp(
  '^(?:export\\s+|pub(?:\\([^)]*\\))?\\s+|public\\s+|private\\s+|protected\\s+|static\\s+|' +
    'async\\s+|default\\s+|abstract\\s+|final\\s+|internal\\s+|override\\s+)*' +
    '(?:def|class|func|fn|function|interface|struct|enum|trait|impl|type|module|' +
    'namespace|object|record|macro_rules!|const|let|var|val|@\\w+|#\\[|##?\\s)',
);
const CLOSING_RE = /^[\])}][\])};,]*\s*$/;
// A declaration directly under one of these (decorator, attribute, comment) stays attached.
const GLUE_PREFIXES = ['@', '#', '//', '/*', '--'];

type Range = [start: number, end: number]; // half-open, 0-based

const isBlank = (s: string) => s.trim() === '';
const startsIndented = (s: string) => /^\s/.test(s);

function isBoundary(lines: string[], i: number): boolean {
  const line = lines[i]!;
  if (isBlank(line) || startsIndented(line) || CLOSING_RE.test(line)) return false;
  const prev = i > 0 ? lines[i - 1]! : '';
  if (DECL_RE.test(line)) {
    return !(prev && !startsIndented(prev) && GLUE_PREFIXES.some((p) => prev.startsWith(p)));
  }
  return i > 0 && isBlank(prev);
}

function segments(lines: string[]): Range[] {
  const starts = [0];
  for (let i = 1; i < lines.length; i++) if (isBoundary(lines, i)) starts.push(i);
  return starts.map((s, k) => [s, starts[k + 1] ?? lines.length]);
}

function windows(start: number, end: number, maxLines: number, overlap: number): Range[] {
  const out: Range[] = [];
  for (let s = start; s < end; s += maxLines - overlap) {
    const e = Math.min(s + maxLines, end);
    out.push([s, e]);
    if (e === end) break;
  }
  return out;
}

function trimBlank(lines: string[], start: number, end: number): Range {
  while (start < end && isBlank(lines[start]!)) start++;
  while (end > start && isBlank(lines[end - 1]!)) end--;
  return [start, end];
}

export function splitLines(text: string): string[] {
  const lines = text.split(/\r\n|\r|\n/);
  if (lines.at(-1) === '') lines.pop(); // match Python's str.splitlines()
  return lines;
}

export function chunkText(text: string, opts: ChunkOptions = DEFAULT_CHUNKING): Chunk[] {
  const { maxLines, maxChars, overlapLines } = opts;
  const lines = splitLines(text);
  if (lines.length === 0) return [];

  // Prefix sums of line lengths (+1 for the newline) make size checks O(1).
  const prefix = [0];
  for (const line of lines) prefix.push(prefix.at(-1)! + line.length + 1);
  const size = (s: number, e: number) => prefix[e]! - prefix[s]!;

  // 1) Split oversized segments into windows so every piece fits the budget.
  const pieces: Range[] = [];
  for (const seg of segments(lines)) {
    const [s, e] = trimBlank(lines, seg[0], seg[1]);
    if (s >= e) continue;
    if (e - s <= maxLines && size(s, e) <= maxChars) {
      pieces.push([s, e]);
      continue;
    }
    for (const [ws, w] of windows(s, e, maxLines, overlapLines)) {
      // A window can still exceed maxChars with very long lines; shrink it.
      let we = w;
      while (we - ws > 1 && size(ws, we) > maxChars) we -= Math.max(1, Math.floor((we - ws) / 4));
      pieces.push([ws, we]);
    }
  }

  // 2) Greedily merge neighbouring small pieces up to the budget. Overlapping windows never merge.
  const merged: Range[] = [];
  for (const [s, e] of pieces) {
    const last = merged.at(-1);
    if (last && s >= last[1] && e - last[0] <= maxLines && size(last[0], e) <= maxChars) {
      last[1] = e;
      continue;
    }
    merged.push([s, e]);
  }

  const chunks: Chunk[] = [];
  const seen = new Set<string>();
  for (const range of merged) {
    const [s, e] = trimBlank(lines, range[0], range[1]);
    const key = `${s}:${e}`;
    if (s >= e || seen.has(key)) continue;
    seen.add(key);
    let body = lines.slice(s, e).join('\n');
    if (body.length > maxChars) body = body.slice(0, maxChars); // one enormous line (minified code)
    chunks.push({ startLine: s + 1, endLine: e, text: body });
  }
  return chunks;
}
