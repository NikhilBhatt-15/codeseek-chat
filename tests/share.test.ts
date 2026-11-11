import { describe, expect, it } from 'vitest';
import { normaliseQuestion } from '../src/examples';
import { fileMatches, shareUrl } from '../src/share';

describe('share links', () => {
  it('builds /repo?repo=&q= links under the site base', () => {
    const url = shareUrl('https://x.netlify.app/', 'sindresorhus/ky', 'Where do retries happen?');
    expect(url).toBe('https://x.netlify.app/repo?repo=sindresorhus%2Fky&q=Where+do+retries+happen%3F');
    expect(shareUrl('https://x.github.io/codeseek-chat', 'a/b')).toBe('https://x.github.io/codeseek-chat/repo?repo=a%2Fb');
    const back = new URL(url).searchParams;
    expect(back.get('repo')).toBe('sindresorhus/ky');
    expect(back.get('q')).toBe('Where do retries happen?');
  });

  it('matches saved questions loosely', () => {
    expect(normaliseQuestion('  Where do  retries happen?? ')).toBe(normaliseQuestion('where do retries happen'));
  });
});

describe('fileMatches', () => {
  it('groups hits by file with the best score', () => {
    const files = fileMatches([
      { path: 'a.ts', score: 0.5 },
      { path: 'b.ts', score: 0.7 },
      { path: 'a.ts', score: 0.6 },
    ]);
    expect(files).toEqual([
      { path: 'b.ts', score: 0.7, chunks: 1 },
      { path: 'a.ts', score: 0.6, chunks: 2 },
    ]);
  });
});
