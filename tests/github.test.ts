import { describe, expect, it } from 'vitest';
import { blobUrl, fetchContents, fetchSnapshot, GitHubError, isIndexable, parseRepo, rawUrl, selectFiles } from '../src/github';

describe('parseRepo', () => {
  it.each([
    ['owner/repo', { owner: 'owner', repo: 'repo', ref: undefined }],
    ['https://github.com/owner/repo', { owner: 'owner', repo: 'repo', ref: undefined }],
    ['github.com/owner/re.po.git', { owner: 'owner', repo: 're.po', ref: undefined }],
    ['https://github.com/owner/repo/tree/dev/x', { owner: 'owner', repo: 'repo', ref: 'dev/x' }],
    ['  https://www.github.com/o-1/r_2/  ', { owner: 'o-1', repo: 'r_2', ref: undefined }],
  ])('parses %s', (input, expected) => expect(parseRepo(input)).toEqual(expected));

  it.each(['', 'just-a-name', 'https://gitlab.com/o/r', 'a/b/c'])('rejects %s', (input) =>
    expect(parseRepo(input)).toBeNull(),
  );
});

describe('file selection', () => {
  it('applies the same rules as the CLI', () => {
    expect(isIndexable('src/app.ts')).toBe(true);
    expect(isIndexable('Dockerfile')).toBe(true);
    expect(isIndexable('node_modules/x/index.js')).toBe(false);
    expect(isIndexable('.github/workflows/ci.yml')).toBe(false);
    expect(isIndexable('package-lock.json')).toBe(false);
    expect(isIndexable('dist/app.min.js')).toBe(false);
    expect(isIndexable('logo.png')).toBe(false);
  });

  it('prefers source over docs and tests, within limits', () => {
    const entries = [
      { path: 'README.md', size: 100 },
      { path: 'tests/test_a.py', size: 100 },
      { path: 'src/a.py', size: 100 },
      { path: 'src/huge.py', size: 999_999 },
      { path: 'src/b.py', size: 100 },
    ];
    const { files, skipped } = selectFiles(entries, { maxFiles: 2, maxFileBytes: 1000, maxTotalBytes: 10_000 });
    expect(files.map((f) => f.path)).toEqual(['src/a.py', 'src/b.py']);
    expect(skipped).toBe(2);
  });
});

describe('urls', () => {
  const snap = { owner: 'o', repo: 'r', commit: 'abc' };
  it('encodes paths and line ranges', () => {
    expect(rawUrl(snap, 'a b/c#.py')).toBe('https://raw.githubusercontent.com/o/r/abc/a%20b/c%23.py');
    expect(blobUrl(snap, 'x.py', 3, 9)).toBe('https://github.com/o/r/blob/abc/x.py#L3-L9');
    expect(blobUrl(snap, 'x.py', 3, 3)).toBe('https://github.com/o/r/blob/abc/x.py#L3');
  });
});

const json = (body: unknown, status = 200, headers: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), { status, headers });

describe('fetchSnapshot', () => {
  it('resolves the default branch commit and filters the tree in two API calls', async () => {
    const calls: string[] = [];
    const fake = (async (url: string) => {
      calls.push(url);
      if (url.endsWith('/repos/o/r/commits/HEAD')) {
        return json({ sha: 'c0ffee', html_url: 'https://github.com/Owner/Repo/commit/c0ffee' });
      }
      if (url.includes('/git/trees/c0ffee')) {
        return json({
          truncated: false,
          tree: [
            { path: 'src', type: 'tree' },
            { path: 'src/a.py', type: 'blob', size: 10 },
            { path: 'img.png', type: 'blob', size: 10 },
          ],
        });
      }
      throw new Error(`unexpected ${url}`);
    }) as unknown as typeof fetch;
    const snap = await fetchSnapshot({ owner: 'o', repo: 'r' }, undefined, fake);
    expect(snap).toMatchObject({ owner: 'Owner', repo: 'Repo', branch: null, commit: 'c0ffee', skipped: 0 });
    expect(snap.files.map((f) => f.path)).toEqual(['src/a.py']);
    expect(calls).toHaveLength(2);
  });

  it('uses the ref from the URL when given', async () => {
    const calls: string[] = [];
    const fake = (async (url: string) => {
      calls.push(url);
      return url.includes('/commits/')
        ? json({ sha: 's', html_url: 'https://github.com/o/r/commit/s' })
        : json({ truncated: false, tree: [] });
    }) as unknown as typeof fetch;
    const snap = await fetchSnapshot({ owner: 'o', repo: 'r', ref: 'feature/x' }, undefined, fake);
    expect(calls[0]).toContain('/commits/feature%2Fx');
    expect(snap.branch).toBe('feature/x');
  });

  it('explains rate limiting and missing repos', async () => {
    const limited = (async () => json({}, 403, { 'x-ratelimit-reset': '0' })) as unknown as typeof fetch;
    await expect(fetchSnapshot({ owner: 'o', repo: 'r' }, undefined, limited)).rejects.toThrow(/rate limit/);
    const missing = (async () => json({}, 404)) as unknown as typeof fetch;
    await expect(fetchSnapshot({ owner: 'o', repo: 'r' }, undefined, missing)).rejects.toBeInstanceOf(GitHubError);
  });
});

describe('fetchContents', () => {
  it('downloads with bounded concurrency and drops binary files', async () => {
    let inFlight = 0;
    let peak = 0;
    const fake = (async (url: string) => {
      inFlight++;
      peak = Math.max(peak, inFlight);
      await new Promise((r) => setTimeout(r, 5));
      inFlight--;
      return new Response(url.endsWith('bin.py') ? 'a\u0000b' : `text of ${url.split('/').pop()}`);
    }) as unknown as typeof fetch;
    const files = Array.from({ length: 10 }, (_, i) => ({ path: `f${i}.py`, size: 1 }));
    files.push({ path: 'bin.py', size: 1 });
    const snap = { owner: 'o', repo: 'r', branch: 'main', commit: 'c', files, skipped: 0, truncated: false };
    const progress: number[] = [];
    const out = await fetchContents(snap, (d) => progress.push(d), undefined, fake, 3);
    expect(peak).toBeLessThanOrEqual(3);
    expect(out.get('f0.py')).toBe('text of f0.py');
    expect(out.get('bin.py')).toBeNull();
    expect(progress.at(-1)).toBe(11);
  });
});
