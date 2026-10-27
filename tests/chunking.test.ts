import { describe, expect, it } from 'vitest';
import { chunkText, splitLines } from '../src/chunking';

const opts = (maxLines: number, maxChars = 2000, overlapLines = 2) => ({ maxLines, maxChars, overlapLines });

// Mirrors codeseek's tests/test_chunking.py so both chunkers stay in agreement.
describe('chunkText', () => {
  it('returns nothing for empty or blank text', () => {
    expect(chunkText('', opts(20))).toEqual([]);
    expect(chunkText('\n\n  \n', opts(20))).toEqual([]);
  });

  it('keeps a small file as one chunk', () => {
    const chunks = chunkText('def a():\n    return 1\n\n\ndef b():\n    return 2\n', opts(20));
    expect(chunks.map((c) => [c.startLine, c.endLine])).toEqual([[1, 6]]);
  });

  it('splits at top-level definitions', () => {
    const body = Array.from({ length: 8 }, (_, i) => `    x${i} = ${i}`).join('\n');
    const chunks = chunkText(`def first():\n${body}\n\n\ndef second():\n${body}\n`, opts(10));
    expect(chunks.map((c) => c.text.split('\n')[0])).toEqual(['def first():', 'def second():']);
  });

  it('keeps a decorator with its function', () => {
    const body = Array.from({ length: 8 }, (_, i) => `    y${i} = ${i}`).join('\n');
    const chunks = chunkText(`x = 1\n\n@cache\ndef f():\n${body}\n`, opts(10));
    expect(chunks.some((c) => c.text.startsWith('@cache\ndef f():'))).toBe(true);
  });

  it('uses overlapping windows for an oversized block and covers every line', () => {
    const text = 'def big():\n' + Array.from({ length: 50 }, (_, i) => `    v${i} = ${i}`).join('\n');
    const chunks = chunkText(text, opts(20, 10_000, 5));
    expect(chunks.length).toBeGreaterThanOrEqual(3);
    expect(chunks.every((c) => c.endLine - c.startLine + 1 <= 20)).toBe(true);
    expect(chunks[1]!.startLine).toBe(chunks[0]!.endLine - 5 + 1);
    const covered = new Set(chunks.flatMap((c) => Array.from({ length: c.endLine - c.startLine + 1 }, (_, k) => c.startLine + k)));
    expect(covered.size).toBe(51);
  });

  it('respects the character budget', () => {
    const text = Array.from({ length: 30 }, () => 'z'.repeat(500)).join('\n');
    expect(chunkText(text, opts(20)).every((c) => c.text.length <= 2000)).toBe(true);
  });

  it('reports line numbers that match the text', () => {
    const lines = Array.from({ length: 119 }, (_, k) => ((k + 1) % 7 ? `line_${k + 1}` : ''));
    for (const c of chunkText(lines.join('\n'), opts(15, 5000, 3))) {
      expect(c.text.split('\n')).toEqual(lines.slice(c.startLine - 1, c.endLine));
    }
  });

  it('handles CRLF like Python splitlines', () => {
    expect(splitLines('a\r\nb\r\n')).toEqual(['a', 'b']);
    expect(chunkText('a\r\nb\r\n', opts(10))[0]!.text).toBe('a\nb');
  });
});
