// Build, search and cache a repository's embedding index.
import { chunkText, DEFAULT_CHUNKING } from './chunking';
import { formatDocument } from './embedder';
import type { RepoSnapshot } from './github';
import { EMBED_DIM, INDEX_VERSION, MODEL_ID } from './protocol';

export interface IndexedChunk {
  path: string;
  startLine: number;
  endLine: number;
  text: string;
}

export interface RepoIndex {
  key: string;
  owner: string;
  repo: string;
  branch: string | null; // null = default branch
  commit: string;
  chunks: IndexedChunk[];
  vectors: Float32Array; // chunks.length × EMBED_DIM, row-major, L2-normalised
  files: number;
  createdAt: number;
}

export interface Hit extends IndexedChunk {
  index: number;
  score: number;
}

export const indexKey = (owner: string, repo: string, commit: string) =>
  `${owner}/${repo}@${commit}#v${INDEX_VERSION}:${MODEL_ID}:${EMBED_DIM}`.toLowerCase();

export function chunkRepo(contents: Map<string, string | null>): IndexedChunk[] {
  const out: IndexedChunk[] = [];
  for (const [path, text] of contents) {
    if (!text) continue;
    for (const c of chunkText(text, DEFAULT_CHUNKING)) out.push({ path, ...c });
  }
  return out;
}

export interface EmbedFn {
  (texts: string[]): Promise<Float32Array>;
}

/** Embeds chunks in length-sorted batches (less padding), reporting progress after each. */
export async function embedChunks(
  chunks: IndexedChunk[],
  embed: EmbedFn,
  onProgress: (done: number, total: number) => void,
  signal?: AbortSignal,
  batchSize = 8,
): Promise<Float32Array> {
  const texts = chunks.map((c) => formatDocument(c.path, c.text));
  const order = texts.map((_, i) => i).sort((a, b) => texts[a]!.length - texts[b]!.length);
  const vectors = new Float32Array(chunks.length * EMBED_DIM);
  for (let start = 0; start < order.length; start += batchSize) {
    signal?.throwIfAborted();
    const idx = order.slice(start, start + batchSize);
    const batch = await embed(idx.map((i) => texts[i]!));
    idx.forEach((row, k) => vectors.set(batch.subarray(k * EMBED_DIM, (k + 1) * EMBED_DIM), row * EMBED_DIM));
    onProgress(Math.min(start + batchSize, order.length), order.length);
  }
  return vectors;
}

/** Exact cosine search (vectors are normalised, so a dot product). */
export function search(index: Pick<RepoIndex, 'chunks' | 'vectors'>, query: Float32Array, k: number): Hit[] {
  const n = index.chunks.length;
  const scores = new Float32Array(n);
  for (let r = 0; r < n; r++) {
    let s = 0;
    const off = r * EMBED_DIM;
    for (let d = 0; d < EMBED_DIM; d++) s += index.vectors[off + d]! * query[d]!;
    scores[r] = s;
  }
  return Array.from(scores.keys())
    .sort((a, b) => scores[b]! - scores[a]!)
    .slice(0, k)
    .map((i) => ({ ...index.chunks[i]!, index: i, score: scores[i]! }));
}

/** Drops hits that overlap a better hit in the same file (overlapping windows). */
export function dedupeHits(hits: Hit[]): Hit[] {
  const kept: Hit[] = [];
  for (const h of hits) {
    const overlaps = kept.some((k) => k.path === h.path && h.startLine <= k.endLine && k.startLine <= h.endLine);
    if (!overlaps) kept.push(h);
  }
  return kept;
}

export function indexFromSnapshot(
  snap: RepoSnapshot,
  chunks: IndexedChunk[],
  vectors: Float32Array,
  files: number,
): RepoIndex {
  return {
    key: indexKey(snap.owner, snap.repo, snap.commit),
    owner: snap.owner,
    repo: snap.repo,
    branch: snap.branch,
    commit: snap.commit,
    chunks,
    vectors,
    files,
    createdAt: Date.now(),
  };
}

// ---- IndexedDB cache (best effort: the app works without it) ----

const DB_NAME = 'codeseek-chat';
const STORE = 'indexes';
const KEEP = 6;

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE, { keyPath: 'key' });
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function tx<T>(mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  return openDb().then(
    (db) =>
      new Promise<T>((resolve, reject) => {
        const req = fn(db.transaction(STORE, mode).objectStore(STORE));
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
      }),
  );
}

export async function loadCached(key: string): Promise<RepoIndex | null> {
  try {
    return ((await tx('readonly', (s) => s.get(key))) as RepoIndex | undefined) ?? null;
  } catch {
    return null;
  }
}

/** Newest cached index for a repo at any commit: a fallback when GitHub can't be reached. */
export async function loadLatestCached(owner: string, repo: string): Promise<RepoIndex | null> {
  try {
    const all = (await tx('readonly', (s) => s.getAll())) as RepoIndex[];
    const prefix = `${owner}/${repo}@`.toLowerCase();
    const suffix = indexKey('', '', '').slice(2); // "#v<version>:<model>:<dim>"
    return (
      all
        .filter((i) => i.key.startsWith(prefix) && i.key.endsWith(suffix))
        .sort((a, b) => b.createdAt - a.createdAt)[0] ?? null
    );
  } catch {
    return null;
  }
}

export async function saveCached(index: RepoIndex): Promise<void> {
  try {
    await tx('readwrite', (s) => s.put(index));
    const all = (await tx('readonly', (s) => s.getAll())) as RepoIndex[];
    const stale = all.sort((a, b) => b.createdAt - a.createdAt).slice(KEEP);
    for (const old of stale) await tx('readwrite', (s) => s.delete(old.key));
  } catch {
    // Private browsing or quota: carry on without a cache.
  }
}
