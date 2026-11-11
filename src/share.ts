// Small pure helpers for shareable links and the "how it found this" view.
import type { Hit } from './repoIndex';

/** A link that reopens `repo` and asks `question`: <base>repo?repo=owner/name&q=question. */
export function shareUrl(base: string, repo: string, question?: string): string {
  const url = new URL('repo', base.endsWith('/') ? base : `${base}/`);
  url.searchParams.set('repo', repo);
  if (question) url.searchParams.set('q', question);
  return url.toString();
}

export interface FileMatch {
  path: string;
  score: number; // best chunk in the file
  chunks: number; // retrieved chunks from this file
}

/** Groups hits by file, best file first. */
export function fileMatches(hits: Pick<Hit, 'path' | 'score'>[]): FileMatch[] {
  const byPath = new Map<string, FileMatch>();
  for (const h of hits) {
    const f = byPath.get(h.path);
    if (f) {
      f.score = Math.max(f.score, h.score);
      f.chunks++;
    } else byPath.set(h.path, { path: h.path, score: h.score, chunks: 1 });
  }
  return [...byPath.values()].sort((a, b) => b.score - a.score);
}
