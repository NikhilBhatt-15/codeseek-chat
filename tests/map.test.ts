import { describe, expect, it } from 'vitest';
import { clusterCount, describeClusters, kmeans, prefixRows, project, rng } from '../src/mapLayout';

// Two well-separated groups on the unit circle.
const rows = [
  ...Array.from({ length: 12 }, (_, i) => [1, 0.01 * i]),
  ...Array.from({ length: 8 }, (_, i) => [0.01 * i, 1]),
].map((r) => {
  const n = Math.hypot(r[0]!, r[1]!);
  return [r[0]! / n, r[1]! / n];
});

describe('map layout', () => {
  it('keeps the number of topics small enough for distinct colors', () => {
    expect(clusterCount(3)).toBe(3);
    expect(clusterCount(72)).toBe(4);
    expect(clusterCount(349)).toBe(8);
    expect(clusterCount(5000)).toBe(8);
  });

  it('finds separated groups and numbers the biggest first', () => {
    const assign = kmeans(rows, 2, rng(1));
    expect(new Set(assign.slice(0, 12))).toEqual(new Set([0]));
    expect(new Set(assign.slice(12))).toEqual(new Set([1]));
  });

  it('is deterministic for a seed', () => {
    expect(kmeans(rows, 3, rng(5))).toEqual(kmeans(rows, 3, rng(5)));
  });

  it('re-normalises the Matryoshka prefix', () => {
    const [row] = prefixRows(new Float32Array([3, 4, 9, 9]), 4, 2);
    expect(row![0]).toBeCloseTo(0.6);
    expect(row![1]).toBeCloseTo(0.8);
  });

  it('projects into the unit square', async () => {
    const pts = await project(rows, rng(42));
    expect(pts).toHaveLength(rows.length);
    for (const [x, y] of pts) {
      expect(x).toBeGreaterThanOrEqual(0);
      expect(y).toBeLessThanOrEqual(1);
    }
  });

  it('names clusters by distinctive identifiers', () => {
    const chunks = [
      { path: 'retry.ts', startLine: 1, endLine: 2, text: 'function retryDelay() { backoff() }' },
      { path: 'retry2.ts', startLine: 1, endLine: 2, text: 'const backoff = retryDelay * 2' },
      { path: 'router.go', startLine: 1, endLine: 2, text: 'func (mx *Mux) Route(pattern string)' },
      { path: 'tree.go', startLine: 1, endLine: 2, text: 'func findRoute(pattern string) *node' },
    ];
    const clusters = describeClusters(chunks, [0, 0, 1, 1]);
    expect(clusters[0]!.keywords).toContain('backoff');
    expect(clusters[1]!.keywords).toContain('pattern');
    expect(clusters[0]!.files).toEqual(['retry.ts', 'retry2.ts']);
  });
});
