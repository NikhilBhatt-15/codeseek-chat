// The math behind the code map: reduce, cluster, project to 2D and name the clusters.
// Everything is deterministic (seeded), so the same index always draws the same map.
import { UMAP } from 'umap-js';
import { tokenize } from './retrieval';
import type { IndexedChunk } from './repoIndex';

/** Small seeded PRNG (mulberry32). */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * The first `dim` dimensions of each row, re-normalised. EmbeddingGemma is trained with
 * Matryoshka loss, so a short prefix keeps the coarse structure: plenty for a map, and
 * clustering and UMAP's neighbour search get 6x cheaper than on all 768 dimensions.
 */
export function prefixRows(vectors: Float32Array, stride: number, dim: number): number[][] {
  const n = vectors.length / stride;
  const rows: number[][] = [];
  for (let r = 0; r < n; r++) {
    let norm = 0;
    for (let d = 0; d < dim; d++) norm += vectors[r * stride + d]! ** 2;
    norm = Math.sqrt(norm) || 1;
    const row = new Array<number>(dim);
    for (let d = 0; d < dim; d++) row[d] = vectors[r * stride + d]! / norm;
    rows.push(row);
  }
  return rows;
}

const dot = (a: number[], b: number[]) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!;
  return s;
};

/** How many topics to show: grows with the repo, at most 8 so each keeps a distinct color. */
export const clusterCount = (n: number) => Math.max(Math.min(4, n), Math.min(8, Math.round(Math.sqrt(n / 6))));

/** Spherical k-means (cosine) with k-means++ seeding. Returns a cluster id per row. */
export function kmeans(rows: number[][], k: number, random: () => number, iterations = 20): number[] {
  const n = rows.length;
  if (n === 0) return [];
  k = Math.min(k, n);
  const centers: number[][] = [rows[Math.floor(random() * n)]!.slice()];
  while (centers.length < k) {
    // Pick the next center with probability proportional to its distance from the nearest one.
    const dist = rows.map((r) => Math.max(0, 1 - Math.max(...centers.map((c) => dot(r, c)))));
    const total = dist.reduce((a, b) => a + b, 0);
    let pick = random() * total;
    let i = 0;
    while (i < n - 1 && (pick -= dist[i]!) > 0) i++;
    centers.push(rows[i]!.slice());
  }
  const assign = new Array<number>(n).fill(0);
  for (let it = 0; it < iterations; it++) {
    let moved = false;
    for (let i = 0; i < n; i++) {
      let best = 0;
      let bestSim = -Infinity;
      for (let c = 0; c < k; c++) {
        const s = dot(rows[i]!, centers[c]!);
        if (s > bestSim) [bestSim, best] = [s, c];
      }
      if (assign[i] !== best) moved = true;
      assign[i] = best;
    }
    for (let c = 0; c < k; c++) {
      const sum = new Array<number>(rows[0]!.length).fill(0);
      let count = 0;
      for (let i = 0; i < n; i++) {
        if (assign[i] !== c) continue;
        count++;
        rows[i]!.forEach((v, d) => (sum[d]! += v));
      }
      if (!count) continue;
      const norm = Math.sqrt(sum.reduce((a, v) => a + v * v, 0)) || 1;
      centers[c] = sum.map((v) => v / norm);
    }
    if (!moved && it > 0) break;
  }
  // Number clusters by size, biggest first, so colors follow a stable order.
  const sizes = new Map<number, number>();
  assign.forEach((c) => sizes.set(c, (sizes.get(c) ?? 0) + 1));
  const order = [...sizes.keys()].sort((a, b) => sizes.get(b)! - sizes.get(a)! || a - b);
  const renumber = new Map(order.map((c, i) => [c, i]));
  return assign.map((c) => renumber.get(c)!);
}

/** 2D layout with UMAP, run in slices so the page stays responsive. Coordinates are 0..1. */
export async function project(
  rows: number[][],
  random: () => number,
  onProgress?: (fraction: number) => void,
): Promise<[number, number][]> {
  const n = rows.length;
  let points: number[][];
  if (n < 8) {
    points = rows.map((r) => [r[0]!, r[1]!]); // too few points for UMAP's neighbour graph
  } else {
    // Fewer epochs than UMAP's default 500 for small maps: the layout settles well before then.
    const nEpochs = n <= 1000 ? 200 : 300;
    const umap = new UMAP({ nComponents: 2, nNeighbors: Math.min(15, n - 1), minDist: 0.12, random, nEpochs });
    const epochs = umap.initializeFit(rows);
    // Work in ~30 ms slices, yielding between them. Counting time rather than epochs matters
    // when the tab is in the background, where browsers slow every timer down to ~1 s.
    for (let e = 0; e < epochs; ) {
      const sliceStart = performance.now();
      while (e < epochs && performance.now() - sliceStart < 30) {
        umap.step();
        e++;
      }
      onProgress?.(e / epochs);
      await new Promise((r) => setTimeout(r, 0));
    }
    points = umap.getEmbedding();
  }
  const xs = points.map((p) => p[0]!);
  const ys = points.map((p) => p[1]!);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  return points.map((p) => [(p[0]! - x0) / (x1 - x0 || 1), (p[1]! - y0) / (y1 - y0 || 1)]);
}

// Words that say nothing about what code does.
const GENERIC = new Set(
  (
    'const let var function return import export from def self this true false null none undefined type string number ' +
    'boolean new class async await err nil func package int str bool void public private static value values data ' +
    'test tests expect assert result results option options item items obj object array list name names index key ' +
    'keys args kwargs param params arg get set has call value_ len len_ ok error t context src source lib code file ' +
    'md ts js py go json yaml rst txt main end the com org www example https http core util utils helper helpers ' +
    'internal common base default make create new'
  ).split(' '),
);

export interface ClusterInfo {
  id: number;
  size: number;
  name: string; // short label, from distinctive identifiers (or Gemini)
  keywords: string[];
  files: string[]; // most common files, most chunks first
}

/** Names each cluster by its most distinctive identifiers (TF-IDF over chunks). */
export function describeClusters(chunks: IndexedChunk[], assign: number[]): ClusterInfo[] {
  const n = chunks.length;
  const docTokens = chunks.map((c) => new Set(tokenize(`${c.path} ${c.text}`).filter((t) => t.length > 2 && !GENERIC.has(t) && !/^\d/.test(t))));
  const df = new Map<string, number>();
  for (const set of docTokens) for (const t of set) df.set(t, (df.get(t) ?? 0) + 1);
  const ids = [...new Set(assign)].sort((a, b) => a - b);
  return ids.map((id) => {
    const members = assign.flatMap((c, i) => (c === id ? [i] : []));
    const tf = new Map<string, number>();
    for (const i of members) for (const t of docTokens[i]!) tf.set(t, (tf.get(t) ?? 0) + 1);
    const scored = [...tf]
      .filter(([, c]) => c >= Math.max(2, members.length * 0.15) || members.length < 4)
      .map(([t, c]) => [t, (c / members.length) * Math.log(n / df.get(t)!)] as const)
      .sort((a, b) => b[1] - a[1]);
    // Skip near-duplicates ("server" after "createhttptestserver").
    const keywords: string[] = [];
    for (const [t] of scored) {
      if (keywords.length >= 6) break;
      if (keywords.some((k) => k.includes(t) || t.includes(k))) continue;
      keywords.push(t);
    }
    const fileCount = new Map<string, number>();
    for (const i of members) fileCount.set(chunks[i]!.path, (fileCount.get(chunks[i]!.path) ?? 0) + 1);
    const files = [...fileCount].sort((a, b) => b[1] - a[1]).map(([f]) => f);
    const tests = members.filter((i) => /(^|\/)(tests?|__tests__|spec|test-d)\/|[._-](test|spec)\./i.test(chunks[i]!.path)).length;
    const base = keywords.slice(0, 2).join(' · ') || files[0]!.split('/').pop()!;
    const name = tests >= members.length * 0.6 ? `tests: ${keywords[0] ?? base}` : base;
    return { id, size: members.length, name, keywords, files };
  });
}
