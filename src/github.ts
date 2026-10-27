// Fetch a public GitHub repository's source files without a backend.
// Two GitHub API calls per repo (commit, tree); file contents come from
// raw.githubusercontent.com, which does not count against the 60/hour API limit.

export interface RepoRef {
  owner: string;
  repo: string;
  ref?: string; // branch, tag or commit from the URL, if any
}

export interface RepoSnapshot {
  owner: string;
  repo: string;
  branch: string | null; // null = default branch
  commit: string;
  files: RepoFile[];
  skipped: number; // indexable-looking files left out because of the size limits
  skippedBytes: number;
  truncated: boolean; // GitHub's tree listing was cut short (very large repos)
}

export interface RepoFile {
  path: string;
  size: number;
}

export interface Limits {
  maxFiles: number;
  maxFileBytes: number;
  maxTotalBytes: number;
}

export type IndexSize = 'quick' | 'full';

// Code embeds at roughly 1 chunk per KB; a laptop iGPU manages ~2 chunks/s. Quick keeps a
// first index to a few minutes; Full takes everything up to the old browser-friendly cap.
// Either way source is kept first, so it is big test suites and docs that get dropped.
export const LIMITS: Record<IndexSize, Limits> = {
  // Same per-file cap in both, so files Quick leaves out are all counted in `skipped`.
  quick: { maxFiles: 300, maxFileBytes: 200_000, maxTotalBytes: 400_000 },
  full: { maxFiles: 400, maxFileBytes: 200_000, maxTotalBytes: 2_500_000 },
};
export const DEFAULT_LIMITS = LIMITS.quick;

// Same rules as codeseek's discovery.py.
const SKIP_DIRS = new Set([
  '.git', '.hg', '.svn', '.venv', 'venv', 'env', 'node_modules', '__pycache__', '.mypy_cache',
  '.pytest_cache', '.ruff_cache', '.tox', '.idea', '.vscode', 'dist', 'build', 'target', 'out',
  '.next', '.nuxt', '.gradle', 'vendor', '.terraform', 'coverage', '.codeseek',
]);
const TEXT_EXTENSIONS = new Set([
  '.py', '.pyi', '.js', '.jsx', '.mjs', '.cjs', '.ts', '.tsx', '.go', '.rs', '.java', '.kt', '.kts',
  '.scala', '.c', '.h', '.cc', '.cpp', '.cxx', '.hpp', '.hh', '.cs', '.fs', '.swift', '.m', '.mm',
  '.rb', '.php', '.lua', '.dart', '.ex', '.exs', '.erl', '.hs', '.ml', '.clj', '.r', '.jl', '.zig',
  '.nim', '.sh', '.bash', '.zsh', '.ps1', '.psm1', '.bat', '.sql', '.graphql', '.proto', '.html',
  '.css', '.scss', '.sass', '.less', '.vue', '.svelte', '.astro', '.md', '.mdx', '.rst', '.txt',
  '.toml', '.yaml', '.yml', '.json', '.ini', '.cfg', '.xml', '.gradle', '.cmake', '.tf', '.nix',
]);
const TEXT_FILENAMES = new Set(['Dockerfile', 'Makefile', 'CMakeLists.txt', 'Justfile', 'Rakefile', 'Gemfile']);
const SKIP_FILENAMES = new Set([
  'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml', 'poetry.lock', 'uv.lock', 'Cargo.lock',
  'composer.lock', 'Gemfile.lock', 'go.sum',
]);

export class GitHubError extends Error {
  constructor(
    message: string,
    readonly rateLimited = false,
  ) {
    super(message);
  }
}

/** Accepts "owner/repo", github.com URLs (optionally with /tree/<ref>), and .git URLs. */
export function parseRepo(input: string): RepoRef | null {
  const s = input.trim().replace(/\.git$/, '').replace(/\/+$/, '');
  const m =
    s.match(/^(?:https?:\/\/)?(?:www\.)?github\.com\/([\w.-]+)\/([\w.-]+)(?:\/tree\/(.+))?$/i) ??
    s.match(/^([\w.-]+)\/([\w.-]+)$/);
  if (!m) return null;
  return { owner: m[1]!, repo: m[2]!, ref: m[3] };
}

export function isIndexable(path: string): boolean {
  const parts = path.split('/');
  const name = parts.at(-1)!;
  if (parts.slice(0, -1).some((d) => SKIP_DIRS.has(d) || d.startsWith('.'))) return false;
  if (SKIP_FILENAMES.has(name) || name.endsWith('.min.js') || name.endsWith('.min.css')) return false;
  if (TEXT_FILENAMES.has(name)) return true;
  const dot = name.lastIndexOf('.');
  return dot > 0 && TEXT_EXTENSIONS.has(name.slice(dot).toLowerCase());
}

/** Keep the most useful files within the limits: source before docs/config, small before big. */
export function selectFiles(
  entries: RepoFile[],
  limits: Limits = DEFAULT_LIMITS,
): { files: RepoFile[]; skipped: number; skippedBytes: number } {
  const candidates = entries.filter((f) => f.size > 0 && f.size <= limits.maxFileBytes && isIndexable(f.path));
  const rank = (p: string) => (/\.(md|mdx|rst|txt|json|ya?ml|toml|xml|ini|cfg)$/i.test(p) ? 1 : 0);
  const isTest = (p: string) => /(^|\/)(tests?|__tests__|spec)\/|\.(test|spec)\./i.test(p);
  candidates.sort(
    (a, b) =>
      rank(a.path) - rank(b.path) ||
      Number(isTest(a.path)) - Number(isTest(b.path)) ||
      a.path.split('/').length - b.path.split('/').length ||
      a.size - b.size,
  );
  const files: RepoFile[] = [];
  let total = 0;
  for (const f of candidates) {
    if (files.length >= limits.maxFiles || total + f.size > limits.maxTotalBytes) continue;
    files.push(f);
    total += f.size;
  }
  files.sort((a, b) => a.path.localeCompare(b.path));
  const chosen = new Set(files);
  const skippedBytes = candidates.reduce((sum, f) => sum + (chosen.has(f) ? 0 : f.size), 0);
  return { files, skipped: candidates.length - files.length, skippedBytes };
}

async function api<T>(path: string, fetchImpl: typeof fetch, token?: string): Promise<T> {
  const headers: Record<string, string> = { Accept: 'application/vnd.github+json' };
  // Optional: authenticated requests get 5,000/hour instead of 60/hour per IP.
  if (token) headers.Authorization = `Bearer ${token}`;
  const res = await fetchImpl(`https://api.github.com${path}`, { headers });
  if (res.status === 401) throw new GitHubError('GitHub rejected the token. Check or remove it in Settings.');
  if (res.status === 404) throw new GitHubError('Repository not found. Is it public?');
  if (res.status === 403 || res.status === 429) {
    const reset = Number(res.headers.get('x-ratelimit-reset'));
    const when = reset ? ` Try again after ${new Date(reset * 1000).toLocaleTimeString()}.` : '';
    throw new GitHubError(
      token
        ? `GitHub's rate limit for your token was reached.${when}`
        : `GitHub allows 60 requests/hour per network without a login, and that's used up.${when} Add a GitHub token in Settings to raise it to 5,000/hour.`,
      true,
    );
  }
  if (!res.ok) throw new GitHubError(`GitHub API error ${res.status}`);
  return (await res.json()) as T;
}

export async function fetchSnapshot(
  ref: RepoRef,
  limits: Limits = DEFAULT_LIMITS,
  fetchImpl: typeof fetch = fetch,
  token?: string,
): Promise<RepoSnapshot> {
  const base = `/repos/${ref.owner}/${ref.repo}`;
  // HEAD resolves to the default branch's latest commit, saving a call to /repos/{o}/{r}.
  const commit = await api<{ sha: string; html_url: string }>(
    `${base}/commits/${encodeURIComponent(ref.ref ?? 'HEAD')}`,
    fetchImpl,
    token,
  );
  // Canonical owner/repo casing, from https://github.com/<owner>/<repo>/commit/<sha>.
  const [, owner = ref.owner, repo = ref.repo] = new URL(commit.html_url).pathname.split('/');
  const branch = ref.ref ?? null;
  const tree = await api<{ tree: { path: string; type: string; size?: number }[]; truncated: boolean }>(
    `${base}/git/trees/${commit.sha}?recursive=1`,
    fetchImpl,
    token,
  );
  const blobs = tree.tree.filter((e) => e.type === 'blob').map((e) => ({ path: e.path, size: e.size ?? 0 }));
  const { files, skipped, skippedBytes } = selectFiles(blobs, limits);
  return { owner, repo, branch, commit: commit.sha, files, skipped, skippedBytes, truncated: tree.truncated };
}

export function rawUrl(snap: Pick<RepoSnapshot, 'owner' | 'repo' | 'commit'>, path: string): string {
  const encoded = path.split('/').map(encodeURIComponent).join('/');
  return `https://raw.githubusercontent.com/${snap.owner}/${snap.repo}/${snap.commit}/${encoded}`;
}

export function blobUrl(
  snap: Pick<RepoSnapshot, 'owner' | 'repo' | 'commit'>,
  path: string,
  start?: number,
  end?: number,
): string {
  const encoded = path.split('/').map(encodeURIComponent).join('/');
  const lines = start ? `#L${start}${end && end !== start ? `-L${end}` : ''}` : '';
  return `https://github.com/${snap.owner}/${snap.repo}/blob/${snap.commit}/${encoded}${lines}`;
}

/** Download file contents with bounded concurrency. Binary-looking files come back as null. */
export async function fetchContents(
  snap: RepoSnapshot,
  onProgress: (done: number, total: number) => void,
  signal?: AbortSignal,
  fetchImpl: typeof fetch = fetch,
  concurrency = 8,
): Promise<Map<string, string | null>> {
  const out = new Map<string, string | null>();
  let next = 0;
  let done = 0;
  const worker = async () => {
    while (next < snap.files.length) {
      const file = snap.files[next++]!;
      const res = await fetchImpl(rawUrl(snap, file.path), { signal });
      const text = res.ok ? await res.text() : null;
      out.set(file.path, text === null || text.slice(0, 8192).includes('\u0000') ? null : text);
      onProgress(++done, snap.files.length);
    }
  };
  await Promise.all(Array.from({ length: Math.min(concurrency, snap.files.length) }, worker));
  return out;
}
