// Example repos ship with pre-built indexes (public/examples/, made by scripts/build-examples.ts)
// so they open instantly, before the embedding model has downloaded. Each has a few example
// questions whose retrieval results were computed at build time, with a saved answer.
import { EMBED_DIM } from './protocol';
import type { Hit, IndexedChunk, RepoIndex } from './repoIndex';
import { indexKey } from './repoIndex';

export interface ExampleQuestion {
  question: string;
  /** Saved answer in Markdown. [n] cites the n-th retrieved excerpt. */
  answer: string;
}

export interface Example {
  repo: string;
  blurb: string;
  questions: ExampleQuestion[];
  /**
   * Names for the code map's topics, by cluster id (biggest cluster first). Written from each
   * cluster's files and identifiers; the clustering is seeded, so ids are stable for this index.
   */
  topics: string[];
}

/** Shape of public/examples/<owner>__<repo>.json. */
export interface ExampleFile {
  owner: string;
  repo: string;
  commit: string;
  paths: string[];
  skipped: number;
  skippedBytes: number;
  truncated: boolean;
  chunks: IndexedChunk[];
  questions: { question: string; hits: { index: number; score: number }[] }[];
  builtAt: string;
}

export const EXAMPLES: Example[] = [
  {
    repo: 'sindresorhus/ky',
    blurb: 'HTTP client for fetch',
    topics: [
      'HTTP behaviour tests',
      'error & option tests',
      'options & context',
      'bodies, streams & errors',
      'retry logic',
      'schema validation',
      'retry timing tests',
      'option merging',
    ],
    questions: [
      {
        question: 'Where does ky decide whether to retry a failed request?',
        answer: "The decision is made in `Ky.ts`, in the error path that runs after a failed attempt [7]:\n\n1. The retry **limit** and the request **method** must allow it first. `POST` isn't retried by default, and `shouldRetry` isn't even called for it [2][4].\n2. A `retry.shouldRetry({error, retryCount})` callback runs next. Only an exact `false` stops the retry and `true` forces one, while anything else (even `0`) falls through to the defaults [7][2].\n3. **Timeouts** are retried only with `retry.retryOnTimeout`, which defaults to `false` [7][4].\n4. **HTTP errors** are retried only for statuses in `retry.statusCodes`. For `afterStatusCodes`, a `Retry-After`-style header sets the delay, and a malformed one falls back to the computed backoff [7].\n\nThe wait itself goes through `retry.delay`, `jitter` and `backoffLimit` [6]. You can also force a retry with `ky.retry()`, optionally passing a `delay`, `code`, `cause` or replacement `request` [5]. A `ForceRetryError` is recognised by its brand, so this works across copies of ky [1].",
      },
      {
        question: 'How do hooks run before a request is sent?',
        answer: "Two hook types run before anything goes over the network:\n\n- **`init`** hooks run first, on the mutable options, before the `Request` is built. They're **synchronous**, and each gets its own copies of `headers`, `context`, `json`, `searchParams` and `retry`, so edits only affect that request [2][4]. A typical use is adding a search parameter such as an API key [3].\n- **`beforeRequest`** hooks then receive `{request, options, retryCount}` with `retryCount` always `0`, because they run once before retry handling starts [3]. A hook can mutate the request (for example, set an `Authorization` header), return a **new `Request`** (for example, with a changed URL), or return a **`Response`** to skip the network entirely [1][3].\n\nOn retries, `beforeRetry` hooks run instead, and can change the options for that retry and the ones after it [3].",
      },
      {
        question: 'How are timeouts implemented?',
        answer: "Per-attempt timeouts live in `source/utils/timeout.ts` [1]. `timeout()` races the `fetch` call against a `setTimeout` (a manual `Promise` instead of `Promise.race`, see #91):\n\n- When the timer fires first, it marks `timedOut`, **aborts** the request's `AbortController` and rejects with a `TimeoutError` [1][6].\n- If a response still arrives after that, its body is cancelled and the response is dropped [1].\n- A `finally` block always clears the timer, so it can't abort a later retry [1].\n\nThere's also an overall budget: `Ky` keeps a `#deadline` computed from `totalTimeout` (or `Infinity` without one) [2]. Both `timeout` and `totalTimeout` must be a non-negative number or `false`, and anything else throws a `TypeError` [5]. Retry waits use an abortable `delay()` helper, so cancelling also interrupts the wait between attempts [8].",
      },
    ],
  },
  {
    repo: 'pallets/itsdangerous',
    blurb: 'signed data for Python',
    topics: ['serializers', 'build & docs setup', 'signing & keys', 'timestamps & expiry'],
    questions: [
      {
        question: 'How is a signature verified?',
        answer: "`Signer.unsign()` splits the value at the separator and checks the signature, raising `BadSignature(\"Signature ... does not match\")` on failure [2][6]. `validate()` wraps that and returns `True`/`False` [2].\n\nThe check itself is `verify_signature()` [6]:\n\n1. Base64-decode the signature, and return `False` if that fails.\n2. For each secret key, **newest first** (`reversed(self.secret_keys)`), derive the signing key and ask the algorithm to verify the value [6].\n3. Return `True` on the first match.\n\nTrying every key is what makes key rotation work. You pass keys oldest to newest, the newest one signs, and older ones still verify [7][1]. The default algorithm is `HMACAlgorithm`, an HMAC with SHA-1 unless you set `digest_method` [4]. That's still safe, because the hash's collision weakness doesn't carry over to HMAC [1].",
      },
      {
        question: 'When does a timestamped signature count as expired?',
        answer: "`TimestampSigner` adds the time to the signed value. `sign()` appends `sep + base64(int(time.time()))`, then signs the whole thing [2]. So the timestamp is covered by the signature and can't be edited.\n\nOn `unsign(value, max_age=...)` [7]:\n\n1. The signature is checked first. If it's bad, a `BadTimeSignature` is raised (with `date_signed` when the timestamp is readable).\n2. A missing or malformed timestamp also raises `BadTimeSignature` [7][3].\n3. If `max_age` is given, `age = now - timestamp`. When `age > max_age`, it raises **`SignatureExpired`** (`\"Signature age 15 > 5 seconds\"`) with `date_signed` set, so you can tell users when a link was created [7][4][1].\n\n`validate(value, max_age)` returns `False` instead of raising [5], and the tests freeze time to check the exact boundary [3].",
      },
    ],
  },
  {
    repo: 'NikhilBhatt-15/codeseek',
    blurb: 'semantic code search CLI',
    topics: ['embedding backends', 'tests', 'index storage', 'CLI & MCP server'],
    questions: [
      {
        question: 'How are files split into chunks?',
        answer: "`chunk_text()` in `chunking.py` works in three steps, without parsing any particular language [2][8]:\n\n1. **Split at top-level boundaries.** A line is a boundary if it's unindented and either starts a declaration (`def`, `class`, `func`, `fn`, `function`, `struct`, `export`, …, matched by `_DECL_RE`) or follows a blank line [2][3]. Decorators, attributes and comments stay attached to the declaration below them (`_GLUE_PREFIXES`), and closing brackets never start a chunk [3].\n2. **Fit the budget.** Each segment is trimmed of blank lines. If it's over `max_lines`/`max_chars`, it's cut into **overlapping line windows** (`_windows`), which shrink further if long lines still break the character budget [8][3].\n3. **Merge small neighbours.** Adjacent small pieces are merged greedily up to the budget, but overlapping windows never are [8].\n\nThe indexer then embeds chunks in groups of about 256 and commits as it goes, so an interrupted run keeps its progress [1][6].",
      },
      {
        question: 'How are search results ranked?',
        answer: "Ranking is a plain vector similarity search in `IndexStore.search()` [1][7]:\n\n1. `_load_matrix()` reads every chunk's vector from SQLite into one NumPy matrix and caches it until the index's generation changes, so repeated queries don't reload anything [1].\n2. Every chunk is scored against the query vector. Then `np.argpartition` picks the top *n* without a full sort, and `np.argsort` orders just those by descending score [7].\n3. Each winner is joined back to its file path, line range and text, and returned as a `Hit` with its score [7].\n\nSo the order is purely by score, which the tests check (`hits[0].score >= hits[-1].score`) [3]. The CLI's `search` command passes `--limit` and `--glob` through and prints the hits, or JSON with `--json` [6].",
      },
    ],
  },
];

export const findExample = (repo: string) =>
  EXAMPLES.find((e) => e.repo.toLowerCase() === repo.trim().toLowerCase());

export interface LoadedExample {
  example: Example;
  index: RepoIndex;
  builtAt: string;
  /** Pre-computed retrieval results, keyed by the question's normalised text. */
  hits: Map<string, Hit[]>;
}

export const normaliseQuestion = (q: string) => q.trim().toLowerCase().replace(/[?.!\s]+$/, '').replace(/\s+/g, ' ');

const loaded = new Map<string, Promise<LoadedExample>>();

export function loadExample(example: Example, base = import.meta.env.BASE_URL): Promise<LoadedExample> {
  let p = loaded.get(example.repo);
  if (!p) {
    p = fetchExample(example, base);
    p.catch(() => loaded.delete(example.repo)); // allow a retry
    loaded.set(example.repo, p);
  }
  return p;
}

async function fetchExample(example: Example, base: string): Promise<LoadedExample> {
  const name = `${base}examples/${example.repo.replace('/', '__')}`;
  const [meta, bin] = await Promise.all([
    fetch(`${name}.json`).then((r) => (r.ok ? (r.json() as Promise<ExampleFile>) : Promise.reject(new Error(`${r.status}`)))),
    fetch(`${name}.bin`).then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(new Error(`${r.status}`)))),
  ]);
  const vectors = new Float32Array(bin);
  if (vectors.length !== meta.chunks.length * EMBED_DIM) throw new Error('example index is corrupt');
  const index: RepoIndex = {
    key: indexKey(meta.owner, meta.repo, meta.commit),
    owner: meta.owner,
    repo: meta.repo,
    branch: null,
    commit: meta.commit,
    chunks: meta.chunks,
    vectors,
    paths: meta.paths,
    size: 'quick',
    skipped: meta.skipped,
    skippedBytes: meta.skippedBytes,
    truncated: meta.truncated,
    createdAt: 0,
  };
  const hits = new Map<string, Hit[]>();
  for (const q of meta.questions) {
    hits.set(
      normaliseQuestion(q.question),
      q.hits.map((h) => ({ ...meta.chunks[h.index]!, index: h.index, score: h.score })),
    );
  }
  return { example, index, builtAt: meta.builtAt, hits };
}
