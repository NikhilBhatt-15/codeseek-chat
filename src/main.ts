import { runAgent, needsAgent, type AgentStep } from './agent';
import { Embedder } from './embedder';
import { LOCAL_LLM_LABEL, LOCAL_LLM_MB } from './llmProtocol';
import { LocalLlm } from './localLlm';
import { EXAMPLES, findExample, loadExample, normaliseQuestion, type Example, type LoadedExample } from './examples';
import { DEFAULT_MODEL, GeminiError, streamChat, SUGGESTED_MODELS, type Turn } from './gemini';
import { blobUrl, fetchContents, fetchSnapshot, GitHubError, LIMITS, parseRepo, type IndexSize, type RepoSnapshot } from './github';
import { linkCitations, retrievalQuery, systemPrompt, userPrompt } from './prompt';
import { renderMarkdown } from './render';
import {
  buildIndex,
  chunkRepo,
  dedupeHits,
  embedChunks,
  estimateChunks,
  indexKey,
  loadCached,
  loadLatestCached,
  missingFiles,
  saveCached,
  search,
  type Hit,
  type RepoIndex,
} from './repoIndex';
import { EMBED_DIM } from './protocol';
import { go, interceptLinks, showView, viewOf, type View } from './router';
import { fileMatches, shareUrl } from './share';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const el = <K extends keyof HTMLElementTagNameMap>(tag: K, cls?: string, text?: string) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};
const link = (href: string, text: string, cls = '') => {
  const a = el('a', cls, text);
  a.href = href;
  a.target = '_blank';
  a.rel = 'noopener';
  return a;
};

// ---- settings (per-device conveniences; storage may be unavailable) ----

const KEY_STORAGE = 'codeseek-chat:gemini-key';
const GH_TOKEN_STORAGE = 'codeseek-chat:github-token';
const MODEL_STORAGE = 'codeseek-chat:model';
const AGENT_STORAGE = 'codeseek-chat:agent';
const ENGINE_STORAGE = 'codeseek-chat:engine';
/** Who writes answers: Gemini with the visitor's key, a small LLM in the browser, or nobody. */
type Engine = 'cloud' | 'local' | 'off';
const engineOf = (e: string | null): Engine => (e === 'local' || e === 'off' ? e : 'cloud');
type AgentMode = 'auto' | 'always' | 'off';
const agentMode = (m: string | null): AgentMode => (m === 'always' || m === 'off' ? m : 'auto');
const store = {
  get(key: string): string | null {
    try {
      return localStorage.getItem(key) ?? sessionStorage.getItem(key);
    } catch {
      return null;
    }
  },
  set(key: string, value: string | null, persist: boolean): void {
    try {
      localStorage.removeItem(key);
      sessionStorage.removeItem(key);
      if (value) (persist ? localStorage : sessionStorage).setItem(key, value);
    } catch {
      // storage blocked: the value lives in memory for this page only
    }
  },
  persisted(key: string): boolean {
    try {
      return localStorage.getItem(key) !== null;
    } catch {
      return false;
    }
  },
};

const settings = {
  apiKey: store.get(KEY_STORAGE) ?? '',
  githubToken: store.get(GH_TOKEN_STORAGE) ?? '',
  model: store.get(MODEL_STORAGE) ?? DEFAULT_MODEL,
  agent: agentMode(store.get(AGENT_STORAGE)),
  engine: engineOf(store.get(ENGINE_STORAGE)),
};

/** What actually happens for the next question. */
const answerMode = (): 'cloud' | 'local' | 'search' =>
  settings.engine === 'local' ? 'local' : settings.engine === 'cloud' && settings.apiKey ? 'cloud' : 'search';

// ---- offline LLM (created on first use) ----

let localLlm: LocalLlm | null = null;
const llmProgress = { loaded: 0, total: 0 };
function getLocalLlm(): LocalLlm {
  if (!localLlm) {
    localLlm = new LocalLlm();
    localLlm.onProgress = (loaded, total) => Object.assign(llmProgress, { loaded, total });
  }
  return localLlm;
}

function useOffline(): void {
  settings.engine = 'local';
  store.set(ENGINE_STORAGE, 'local', true);
  updateModeNote();
  void getLocalLlm().load().catch(() => {});
}

// ---- device ----

const hasWebGPU = 'gpu' in navigator;
const isMobile =
  /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent) ||
  (matchMedia('(pointer: coarse)').matches && Math.min(screen.width, screen.height) < 820);
/** On phones the 175 MB model download only starts once the visitor asks for it. */
let heavyAllowed = !isMobile;
const MODEL_MB = 175;
const GET_KEY_URL = 'https://aistudio.google.com/apikey';

// ---- state ----

let embedder: Embedder | null = null;
let index: RepoIndex | null = null;
let example: LoadedExample | null = null; // set while a pre-built example index is open
let history: Turn[] = [];
let lastQuestion: string | undefined;
let indexing: AbortController | null = null;
let answering: AbortController | null = null;

const model = { state: 'idle' as 'idle' | 'loading' | 'ready' | 'error', loaded: 0, total: 0, device: '' };

function getEmbedder(): Embedder {
  if (!embedder) {
    embedder = new Embedder();
    embedder.onLoadProgress = (loaded, total) => {
      model.loaded = loaded;
      model.total = total;
      renderModelStatus();
    };
  }
  return embedder;
}

/** Starts (or joins) the model download. */
function loadModel(): Promise<string> {
  if (model.state !== 'ready') model.state = 'loading';
  renderModelStatus();
  const ready = getEmbedder().load();
  ready.then(
    (device) => {
      model.state = 'ready';
      model.device = device;
      renderModelStatus();
    },
    () => {
      model.state = 'error';
      renderModelStatus();
    },
  );
  return ready;
}

const modelProgress = () => (model.total ? `${mb(model.loaded)} / ${mb(model.total)}` : 'starting');

function renderModelStatus(): void {
  const line = $('model-status');
  if (!example || model.state === 'idle') {
    line.hidden = true;
    return;
  }
  line.hidden = false;
  line.textContent =
    model.state === 'ready'
      ? `EmbeddingGemma 2 is ready on your ${model.device === 'webgpu' ? 'GPU' : 'CPU'}. Ask anything about this repo.`
      : model.state === 'error'
        ? 'The embedding model failed to load. Saved answers still work.'
        : `Saved answers are instant. For your own questions, EmbeddingGemma 2 is downloading in the background (${modelProgress()}, once).`;
}

/** Whether the browser already has the model files (Transformers.js caches them in Cache Storage). */
async function modelCached(): Promise<boolean> {
  try {
    const cache = await caches.open('transformers-cache');
    const keys = await cache.keys();
    return keys.some((r) => r.url.includes('embeddinggemma') && r.url.includes('.onnx'));
  } catch {
    return false;
  }
}

// ---- UI helpers ----

function showError(message: string | null): void {
  const box = $('error');
  box.hidden = !message;
  box.textContent = message ?? '';
}

const STEPS = ['Read the repository from GitHub', 'Download files', 'Load EmbeddingGemma 2', 'Embed code'];

function setStep(active: number, note = '', fraction?: number): void {
  $('steps').replaceChildren(...STEPS.map((label, i) => el('li', i < active ? 'done' : i === active ? 'active' : '', label)));
  $('progress-note').textContent = note;
  if (fraction !== undefined) $('bar-fill').style.width = `${Math.round(fraction * 100)}%`;
}

const mb = (bytes: number) => `${(bytes / 1e6).toFixed(0)} MB`;
const plural = (n: number, word: string) => `${n.toLocaleString()} ${word}${n === 1 ? '' : 's'}`;

const modelLabel = (m: string) =>
  m.replace(/^gemini-/, '').replace(/-latest$/, '').split('-').map((w) => w[0]!.toUpperCase() + w.slice(1)).join(' ');

function updateModeNote(): void {
  const mode = answerMode();
  const badge = $('mode-badge');
  badge.classList.toggle('on', mode !== 'search');
  badge.textContent =
    mode === 'cloud'
      ? `AI answers on · ${modelLabel(settings.model)}`
      : mode === 'local'
        ? `Offline AI · ${LOCAL_LLM_LABEL}`
        : 'Search only · turn on AI answers';
  badge.title = mode === 'search' ? 'Add a free Gemini key, or use the offline AI' : 'Change how answers are written';
  $('mode-note').textContent =
    mode === 'cloud'
      ? 'Your question and the matching code are sent to Gemini.'
      : mode === 'local'
        ? 'Answers are written on your device. Nothing leaves your browser.'
        : 'You get the matching code. Answers need a free Gemini key or the offline AI.';
}

const repoParams = (repo: string, question?: string | null) => {
  const params = new URLSearchParams({ repo });
  if (question) params.set('q', question);
  return params;
};

/** Keeps the address bar on /repo?repo=…&q=… (shareable) without adding history entries. */
function setUrl(repo: string | null, question: string | null): void {
  try {
    if (repo) go('repo', repoParams(repo, question), true);
  } catch {
    // ignore
  }
}

/** Opens the workspace view for a repo: a new history entry from home, a replacement inside it. */
function enterWorkspace(repo: string, question?: string | null): void {
  go('repo', repoParams(repo, question), viewOf(new URL(location.href)) === 'repo');
}

// ---- landing ----

function renderExamples(): void {
  const box = $('examples');
  box.replaceChildren(el('p', 'examples-head', 'Instant answers, no download: pick a question'));
  for (const ex of EXAMPLES) {
    const card = el('div', 'example');
    const head = el('button', 'example-repo', ex.repo);
    head.type = 'button';
    head.title = `Open ${ex.repo} (pre-built index)`;
    head.onclick = () => void openExample(ex);
    const qs = el('div', 'example-qs');
    for (const q of ex.questions) {
      const b = el('button', 'chip', q.question);
      b.type = 'button';
      b.onclick = () => void openExample(ex, q.question);
      qs.append(b);
    }
    card.append(head, el('span', 'muted small', ` · ${ex.blurb}`), qs);
    box.append(card);
  }
  if (isMobile) {
    const note = $('device-note');
    note.hidden = false;
    note.textContent =
      'On a phone? The examples below answer instantly. Indexing your own repo downloads a 175 MB model and works best on a desktop browser with WebGPU.';
  }
}

// ---- indexing ----

const RATE_STORAGE = 'codeseek-chat:chunks-per-second';
const sizeSelect = $<HTMLSelectElement>('size-select');
sizeSelect.value = 'quick';

/** This device's embedding speed from its last run, for time estimates. */
function measuredRate(): number | null {
  const rate = Number(store.get(RATE_STORAGE));
  return rate > 0 ? rate : null;
}

function duration(seconds: number): string {
  if (seconds < 60) return `${Math.max(5, Math.round(seconds / 5) * 5)} s`;
  return `${Math.round(seconds / 60)} min`;
}

/** A guess for devices that haven't indexed anything yet: laptop GPUs do ~2 chunks/s, CPUs far less. */
const guessedRate = () => (hasWebGPU && !isMobile ? 2 : 0.4);

function estimate(chunks: number): string {
  const rate = measuredRate();
  return `about ${duration(chunks / (rate ?? guessedRate()))}`;
}

async function estimateLine(snap: RepoSnapshot, files = snap.files): Promise<string> {
  const bytes = files.reduce((sum, f) => sum + f.size, 0);
  const chunks = estimateChunks(bytes);
  const measured = measuredRate() !== null;
  const parts = [`${plural(files.length, 'file')}, ~${plural(chunks, 'chunk')}: ${estimate(chunks)} on ${measured ? 'your device' : 'a typical laptop'}`];
  if (!(await modelCached())) parts.push(`plus a one-time ${MODEL_MB} MB model download`);
  if (!hasWebGPU) parts.push('slower here: this browser has no WebGPU');
  return `${parts.join(', ')}.`;
}

/** Shows the estimate with a Start button; resolves when clicked, rejects when cancelled. */
function waitForStart(signal: AbortSignal, label: string): Promise<void> {
  const box = $('confirm');
  const btn = $<HTMLButtonElement>('start-btn');
  btn.textContent = label;
  box.hidden = false;
  $('progress').scrollIntoView({ block: 'center', behavior: 'smooth' });
  btn.focus({ preventScroll: true });
  return new Promise<void>((resolve, reject) => {
    const done = () => {
      box.hidden = true;
      signal.removeEventListener('abort', abort);
    };
    const abort = () => {
      done();
      reject(signal.reason);
    };
    signal.addEventListener('abort', abort);
    btn.onclick = () => {
      done();
      resolve();
    };
  });
}

interface IndexOptions {
  size?: IndexSize;
  extend?: RepoIndex; // index the files this one skipped, at the same commit
  confirm?: boolean; // show the estimate and wait for Start (shared links, phones)
}

/** Indexes a repo and opens the chat. Resolves to true once the chat is open. */
async function indexRepo(input: string, opts: IndexOptions = {}): Promise<boolean> {
  const ref = parseRepo(input);
  if (!ref) {
    showError('Enter a repository as owner/repo or a github.com URL.');
    return false;
  }
  const size: IndexSize = opts.size ?? (sizeSelect.value === 'full' ? 'full' : 'quick');
  showError(null);
  indexing?.abort();
  const controller = new AbortController();
  indexing = controller;
  const { signal } = controller;

  $<HTMLButtonElement>('index-btn').disabled = true;
  if (!opts.extend) enterWorkspace(`${ref.owner}/${ref.repo}`, new URL(location.href).searchParams.get('q'));
  $('progress').hidden = false;
  $('chat').hidden = true;
  $('estimate').hidden = true;
  $('confirm').hidden = true;
  $('progress-title').textContent = `${opts.extend ? 'Indexing the rest of' : 'Indexing'} ${ref.owner}/${ref.repo}…`;
  setStep(0, 'Listing files…', 0);

  try {
    // Extending pins the commit, so new chunks line up with the ones already indexed.
    const snapRef = opts.extend ? { ...ref, ref: opts.extend.commit } : ref;
    const snap = await fetchSnapshot(snapRef, LIMITS[size], undefined, settings.githubToken || undefined);
    signal.throwIfAborted();
    const cached = opts.extend ?? (await loadCached(indexKey(snap.owner, snap.repo, snap.commit)));
    if (cached && (size === 'quick' || cached.size === 'full' || cached.skipped === 0)) {
      openChat(cached, 'loaded from this browser’s cache');
      if (heavyAllowed) void loadModel(); // warm up for the first question
      return true;
    }
    // A cached quick index of this commit is extended rather than rebuilt.
    const base = cached ?? undefined;
    const todo = missingFiles(snap, base ?? null);
    if (!base && todo.length === 0) throw new GitHubError('No source files found in that repository.');

    const est = $('estimate');
    est.textContent = await estimateLine(snap, todo);
    est.hidden = false;
    if (opts.confirm || !heavyAllowed) {
      setStep(1, isMobile ? 'This may be slow or fail on a phone. A desktop browser works best.' : '', 0);
      await waitForStart(signal, isMobile ? 'Try anyway' : 'Start indexing');
      heavyAllowed = true;
    }

    // Start the model download while files are fetched.
    const modelReady = loadModel();
    modelReady.catch(() => {}); // surfaced below

    setStep(1, `${plural(todo.length, 'file')} to index`, 0.05);
    const contents = await fetchContents(
      { ...snap, files: todo },
      (done, total) => setStep(1, `${done} / ${total} files`, 0.05 + 0.15 * (done / total)),
      signal,
    );
    signal.throwIfAborted();

    const chunks = chunkRepo(contents);
    const timer = setInterval(() => {
      const pct = model.total ? model.loaded / model.total : 0;
      setStep(2, model.total ? `${modelProgress()}. Downloaded once, then cached by your browser.` : 'Starting…', 0.2 + 0.2 * pct);
    }, 250);
    let device: string;
    try {
      device = await modelReady;
    } finally {
      clearInterval(timer);
    }
    signal.throwIfAborted();

    const where = device === 'webgpu' ? 'your GPU (WebGPU)' : 'your CPU (WASM)';
    setStep(3, `${plural(chunks.length, 'chunk')} on ${where}, ${estimate(chunks.length)}`, 0.4);
    const started = performance.now();
    const vectors = await embedChunks(
      chunks,
      (texts) => getEmbedder().embedBatch(texts),
      (done, total) => {
        const secs = (performance.now() - started) / 1000;
        const left = done > 0 ? (secs / done) * (total - done) : 0;
        setStep(3, `${done} / ${plural(total, 'chunk')} on ${where}${left > 3 ? `, about ${duration(left)} left` : ''}`, 0.4 + 0.6 * (done / total));
      },
      signal,
    );
    const secs = (performance.now() - started) / 1000;
    if (chunks.length >= 16 && secs > 0) store.set(RATE_STORAGE, (chunks.length / secs).toFixed(2), true);

    const built = buildIndex(snap, size, chunks, vectors, [...contents.keys()], base);
    await saveCached(built);
    openChat(built, base ? `added ${plural(todo.length, 'file')}` : '');
    return true;
  } catch (err) {
    if (signal.aborted) {
      $('progress').hidden = true;
      if (opts.extend) {
        openChat(opts.extend, ''); // back to the index we had
        return true;
      }
      if (index) $('chat').hidden = false;
      else go('home', '', true);
      return false;
    }
    if (err instanceof GitHubError && err.rateLimited) {
      const fallback = opts.extend ?? (await loadLatestCached(ref.owner, ref.repo));
      if (fallback) {
        openChat(fallback, 'cached copy: GitHub rate limit reached, so newer commits were not checked');
        showError(err.message);
        return true;
      }
    }
    $('progress').hidden = true;
    if (index && opts.extend) $('chat').hidden = false;
    else go('home', '', true);
    showError(err instanceof Error ? err.message : String(err));
    return false;
  } finally {
    if (indexing === controller) indexing = null;
    $<HTMLButtonElement>('index-btn').disabled = false;
  }
}

/** Opens an example from its pre-built index (falling back to indexing it live), then asks `question`. */
async function openExample(ex: Example, question?: string): Promise<void> {
  showError(null);
  indexing?.abort();
  $<HTMLInputElement>('repo-input').value = ex.repo;
  enterWorkspace(ex.repo, question);
  let loaded: LoadedExample;
  try {
    loaded = await loadExample(ex);
  } catch {
    if ((await indexRepo(ex.repo)) && question) await ask(question);
    return;
  }
  openChat(loaded.index, `pre-built index from ${loaded.builtAt}`, loaded);
  const intro = el('div', 'msg bot');
  intro.append(el('p', 'answer', 'This repo is pre-indexed, so these answer instantly. You can also ask your own question.'));
  const chips = el('div', 'example-qs');
  for (const q of ex.questions) {
    const b = el('button', 'chip', q.question);
    b.type = 'button';
    b.onclick = () => void ask(q.question);
    chips.append(b);
  }
  intro.append(chips);
  $('messages').append(intro);
  if (heavyAllowed) void loadModel();
  if (question) await ask(question);
}

function coverageNote(built: RepoIndex): string {
  const notes: string[] = [];
  if (built.skipped) notes.push(`${plural(built.skipped, 'file')} not indexed (size limit)`);
  if (built.truncated) notes.push('GitHub truncated the file list (very large repo)');
  return notes.join('; ');
}

function openChat(built: RepoIndex, note: string, fromExample: LoadedExample | null = null): void {
  index = built;
  example = fromExample;
  history = [];
  lastQuestion = undefined;
  $('progress').hidden = true;
  $('chat').hidden = false;
  $('messages').replaceChildren();
  showTab('ask');
  mapState = null;
  const repoLink = link(`https://github.com/${built.owner}/${built.repo}/tree/${built.commit}`, `${built.owner}/${built.repo}`);
  $('repo-label').replaceChildren(repoLink, ` @ ${built.branch ?? built.commit.slice(0, 7)}${built.branch ? ` (${built.commit.slice(0, 7)})` : ''}`);
  $('repo-stats').textContent = [`${plural(built.paths.length, 'file')}, ${plural(built.chunks.length, 'chunk')}`, note, coverageNote(built)]
    .filter(Boolean)
    .join(' · ');

  const more = $<HTMLButtonElement>('more-btn');
  const canExtend = built.size === 'quick' && built.skipped > 0;
  more.hidden = !canExtend;
  if (canExtend) {
    more.textContent = `Index the rest (${plural(built.skipped, 'file')}, ${estimate(estimateChunks(built.skippedBytes))})`;
    more.onclick = () => void indexRepo(`${built.owner}/${built.repo}`, { size: 'full', extend: built });
  }
  updateModeNote();
  renderModelStatus();
  setUrl(`${built.owner}/${built.repo}`, null);
  if (!isMobile) $('ask-input').focus();
}

// ---- rendering answers ----

/** Code with line numbers taken from the file. */
function codeBlock(text: string, firstLine: number): HTMLPreElement {
  const pre = el('pre', 'numbered');
  const code = el('code');
  text.split('\n').forEach((line, i) => {
    const row = el('span', 'line', line);
    row.dataset.n = String(firstLine + i);
    code.append(row);
  });
  pre.append(code);
  return pre;
}

function hitCard(current: RepoIndex, h: Hit, n: number): HTMLElement {
  const box = el('div', 'source');
  const head = el('div', 'source-head');
  const name = link(blobUrl(current, h.path, h.startLine, h.endLine), '', 'source-link');
  const slash = h.path.lastIndexOf('/');
  name.append(
    el('span', 'num', `[${n}]`),
    el('span', 'dir', slash >= 0 ? h.path.slice(0, slash + 1) : ''),
    el('strong', '', h.path.slice(slash + 1)),
    el('span', 'lines', ` L${h.startLine}–${h.endLine}`),
  );
  head.append(name, el('span', 'score', h.score.toFixed(3)));
  box.append(head, codeBlock(h.text, h.startLine));
  return box;
}

function renderSources(current: RepoIndex, hits: Hit[], open: boolean): HTMLDetailsElement {
  const details = el('details', 'sources');
  details.open = open;
  details.append(el('summary', '', `${plural(hits.length, 'source')} from the repo`));
  hits.forEach((h, i) => details.append(hitCard(current, h, i + 1)));
  return details;
}

// ---- retrieval ----

/**
 * Embedding retrieval. Hybrid BM25 + rank fusion and a cross-encoder reranker were both
 * measured on the benchmark in eval/ and did not beat embeddings alone, so exact matching is
 * left to the agent's find_references tool, which follows identifiers across files.
 */
function retrieve(current: RepoIndex, vector: Float32Array, k = 8): Hit[] {
  return dedupeHits(search(current, vector, 2 * k)).slice(0, k);
}

/** Cosine similarity of a chunk to the question (vectors are normalised). */
function similarity(current: RepoIndex, row: number, query: Float32Array): number {
  let s = 0;
  for (let d = 0; d < EMBED_DIM; d++) s += current.vectors[row * EMBED_DIM + d]! * query[d]!;
  return s;
}

const howTiming = (ms: number) => `in ${ms < 1 ? '<1' : Math.round(ms)} ms`;

const HOW_NOTES = {
  saved: 'Your question was embedded with EmbeddingGemma 2 and compared with every chunk by cosine similarity.',
  search: 'Your question was embedded with EmbeddingGemma 2 and compared with every chunk by cosine similarity.',
  agent: 'Gemini ran the searches listed above, following identifiers across files, and read the results before answering.',
};

/**
 * Plain-language match strength. Cosine scores from this model cluster in a narrow band
 * (about 0.70 to 0.85), so the label is relative to the best match for the question.
 */
function strength(score: number, best: number): { label: string; cls: string } {
  const gap = best - score;
  if (gap <= 0.01) return { label: 'best match', cls: 'best' };
  if (gap <= 0.03) return { label: 'strong', cls: 'strong' };
  if (gap <= 0.06) return { label: 'related', cls: 'related' };
  return { label: 'weak', cls: 'weak' };
}

/** The retrieval at a glance: best-matching files with their similarity scores. */
function renderHowFound(current: RepoIndex, hits: Hit[], timing: string, method: keyof typeof HOW_NOTES = 'search'): HTMLDetailsElement {
  const details = el('details', 'how');
  details.append(el('summary', '', `How it found this · ${plural(current.chunks.length, 'chunk')} searched ${timing}`));
  const files = fileMatches(hits).slice(0, 6);
  const top = files[0]?.score ?? 1;
  // Scale between a bit below the weakest file and the best one, so close scores stay readable.
  const last = files.at(-1)?.score ?? 0;
  const floor = Math.max(0, last - Math.max(0.01, (top - last) * 0.5));
  const list = el('div', 'how-list');
  for (const f of files) {
    const row = el('div', 'how-row');
    const name = link(blobUrl(current, f.path), f.path, 'how-path');
    const bar = el('div', 'how-bar');
    const fill = el('div');
    fill.style.width = `${Math.max(6, ((f.score - floor) / Math.max(1e-6, top - floor)) * 100)}%`;
    bar.append(fill);
    const s = strength(f.score, top);
    const score = el('span', 'how-score');
    score.append(el('span', `strength ${s.cls}`, s.label), ` ${f.score.toFixed(3)}${f.chunks > 1 ? ` ×${f.chunks}` : ''}`);
    row.append(name, bar, score);
    list.append(row);
  }
  details.append(
    list,
    el(
      'p',
      'muted small',
      `${HOW_NOTES[method]} The number is the cosine similarity of the file's best piece (×n = pieces used); the label compares it with the best match. The excerpts are cited below.`,
    ),
  );
  return details;
}

function keyCta(): HTMLElement {
  const box = el('div', 'cta');
  box.append(el('span', '', 'Want a written answer with citations? '));
  const get = link(GET_KEY_URL, 'Get a free Gemini key (30 sec)', 'cta-link');
  const paste = el('button', 'ghost small', 'Paste key');
  paste.type = 'button';
  paste.onclick = openSettings;
  box.append(get, paste);
  if (!isMobile) {
    const offline = el('button', 'ghost small', `Or use offline AI (${LOCAL_LLM_MB} MB, no key)`);
    offline.type = 'button';
    offline.title = `Downloads ${LOCAL_LLM_LABEL} once and writes answers on your device`;
    offline.onclick = () => {
      useOffline();
      offline.textContent = 'Offline AI on. Ask again to get an answer.';
      offline.disabled = true;
    };
    box.append(offline);
  }
  return box;
}

function shareButton(current: RepoIndex, question: string): HTMLButtonElement {
  const btn = el('button', 'ghost small share', 'Copy link to this answer');
  btn.type = 'button';
  btn.onclick = async () => {
    const url = shareUrl(`${location.origin}${import.meta.env.BASE_URL}`, `${current.owner}/${current.repo}`, question);
    try {
      await navigator.clipboard.writeText(url);
      btn.textContent = 'Link copied ✓';
    } catch {
      window.prompt('Copy this link:', url);
    }
    setTimeout(() => (btn.textContent = 'Copy link to this answer'), 2000);
  };
  return btn;
}

function savedAnswer(question: string): { answer: string; hits: Hit[] } | null {
  if (!example || example.index !== index) return null;
  const key = normaliseQuestion(question);
  const q = example.example.questions.find((x) => normaliseQuestion(x.question) === key);
  const hits = example.hits.get(key);
  return q && hits && q.answer ? { answer: q.answer, hits } : null;
}

// ---- asking ----

async function ask(question: string): Promise<void> {
  if (!index || !question.trim()) return;
  showError(null);
  answering?.abort();
  const controller = new AbortController();
  answering = controller;
  const current = index;
  setUrl(`${current.owner}/${current.repo}`, question);

  const messages = $('messages');
  const asked = el('div', 'msg user', question);
  messages.append(asked);
  const bot = el('div', 'msg bot pending');
  const answer = el('div', 'answer');
  bot.append(answer);
  messages.append(bot);
  bot.scrollIntoView({ block: 'nearest' });
  $<HTMLButtonElement>('ask-btn').disabled = true;
  const urlFor = (hits: Hit[]) => (n: number) => {
    const h = hits[n - 1];
    return h ? blobUrl(current, h.path, h.startLine, h.endLine) : null;
  };
  const footer = (shown: Hit[]) => {
    const row = el('div', 'msg-actions');
    const onMap = el('button', 'ghost small', 'Show on map');
    onMap.type = 'button';
    onMap.title = 'See where these matches sit in the map of the code';
    onMap.onclick = () => {
      lastHits = shown;
      showTab('map');
    };
    row.append(shareButton(current, question), onMap);
    return row;
  };

  let status: ReturnType<typeof setInterval> | undefined;
  try {
    const saved = savedAnswer(question);
    if (saved) {
      lastQuestion = question;
      lastHits = saved.hits;
      answer.innerHTML = renderMarkdown(linkCitations(saved.answer, urlFor(saved.hits)));
      bot.append(
        el('p', 'muted small saved', 'Saved example answer. The excerpts were retrieved by EmbeddingGemma 2 from this index.'),
        renderHowFound(current, saved.hits, 'at build time', 'saved'),
        renderSources(current, saved.hits, false),
        footer(saved.hits),
      );
      asked.scrollIntoView({ block: 'start', behavior: 'smooth' });
      history = [...history, { role: 'user' as const, text: question }, { role: 'model' as const, text: saved.answer }].slice(-6);
      return;
    }

    if (model.state !== 'ready' && !heavyAllowed) {
      // Phones: don't start a 175 MB download without asking.
      answer.append(el('p', '', `Your own questions need the ${MODEL_MB} MB embedding model, which is slow or unsupported on most phones. Open this page on a desktop to ask anything, or try the example questions above.`));
      const go = el('button', 'ghost small', 'Download it anyway');
      go.type = 'button';
      go.onclick = () => {
        heavyAllowed = true;
        bot.previousElementSibling?.remove();
        bot.remove();
        void ask(question);
      };
      bot.append(go);
      return;
    }

    if (model.state !== 'ready') {
      const paintStatus = () => (answer.textContent = `Loading EmbeddingGemma 2 (${modelProgress()})… your question runs as soon as it's ready.`);
      paintStatus();
      status = setInterval(paintStatus, 250);
    }
    const query = await getEmbedder().embedQuery(retrievalQuery(question, lastQuestion));
    clearInterval(status);
    answer.textContent = '';
    if (controller.signal.aborted) return;
    const t0 = performance.now();
    const hits = retrieve(current, query);
    const searchMs = performance.now() - t0;
    lastHits = hits;
    lastQuestion = question;

    const mode = answerMode();
    if (mode === 'cloud' && (settings.agent === 'always' || (settings.agent === 'auto' && needsAgent(question)))) {
      const trace = el('ol', 'agent-steps');
      bot.insertBefore(trace, answer);
      const onStep = (s: AgentStep) =>
        trace.append(el('li', s.tool, `${s.label} · ${plural(s.results, s.tool === 'read_lines' ? 'line' : 'result')}`));
      const result = await runAgent({
        apiKey: settings.apiKey,
        model: settings.model,
        index: current,
        question,
        history,
        signal: controller.signal,
        onStep,
        search: async (q, k) => retrieve(current, await getEmbedder().embedQuery(q), k),
      });
      const sources = result.sources.map((h) => ({ ...h, score: similarity(current, h.index, query) }));
      lastHits = sources;
      answer.innerHTML = renderMarkdown(linkCitations(result.answer, urlFor(sources)));
      bot.append(
        renderHowFound(current, sources, `in ${plural(result.steps.length, 'step')}`, 'agent'),
        renderSources(current, sources, false),
        footer(sources),
      );
      history = [...history, { role: 'user' as const, text: question }, { role: 'model' as const, text: result.answer }].slice(-6);
      return;
    }

    if (mode === 'local') {
      const llm = getLocalLlm();
      if (!llm.loaded) {
        const paint = () =>
          (answer.textContent = `Loading the offline AI, ${LOCAL_LLM_LABEL} (${llmProgress.total ? `${mb(llmProgress.loaded)} / ${mb(llmProgress.total)}` : 'starting'}). Downloaded once, then cached.`);
        paint();
        status = setInterval(paint, 250);
        await llm.load();
        clearInterval(status);
        answer.textContent = '';
      }
      let text = '';
      let frame = 0;
      const paint = () => {
        frame = 0;
        answer.innerHTML = renderMarkdown(linkCitations(text, urlFor(hits)));
      };
      const tokensPerSecond = await llm.generate(
        [
          { role: 'system', content: systemPrompt(`${current.owner}/${current.repo}`) },
          { role: 'user', content: userPrompt(question, hits) },
        ],
        (t) => {
          text += t;
          frame ||= requestAnimationFrame(paint);
        },
        controller.signal,
      );
      cancelAnimationFrame(frame);
      paint();
      const where = llm.device === 'webgpu' ? 'your GPU' : 'your CPU';
      bot.append(
        el('p', 'muted small saved', `Written offline by ${LOCAL_LLM_LABEL} on ${where} at ${tokensPerSecond.toFixed(0)} tokens/s. Small models can be wrong; check the sources.`),
        renderHowFound(current, hits, howTiming(searchMs)),
        renderSources(current, hits, false),
        footer(hits),
      );
      return;
    }

    if (mode === 'search') {
      answer.innerHTML = renderMarkdown(hits.length ? 'Here’s the code that best matches your question:' : 'No matching code found.');
      const results = el('div', 'results');
      hits.slice(0, 5).forEach((h, i) => results.append(hitCard(current, h, i + 1)));
      bot.append(results, keyCta(), renderHowFound(current, hits, howTiming(searchMs)), footer(hits));
      return;
    }

    let text = '';
    let frame = 0;
    const paint = () => {
      frame = 0;
      answer.innerHTML = renderMarkdown(linkCitations(text, urlFor(hits)));
    };
    const turns: Turn[] = [...history, { role: 'user', text: userPrompt(question, hits) }];
    text = await streamChat({
      apiKey: settings.apiKey,
      model: settings.model,
      system: systemPrompt(`${current.owner}/${current.repo}`),
      turns,
      signal: controller.signal,
      onText: (delta) => {
        text += delta;
        frame ||= requestAnimationFrame(paint);
      },
    });
    cancelAnimationFrame(frame);
    paint();
    bot.append(renderHowFound(current, hits, howTiming(searchMs)), renderSources(current, hits, false), footer(hits));
    // Keep a short plain-text history (without the bulky excerpts) for follow-ups.
    history = [...history, { role: 'user' as const, text: question }, { role: 'model' as const, text }].slice(-6);
  } catch (err) {
    if (controller.signal.aborted) return;
    const message = err instanceof GeminiError || err instanceof Error ? err.message : String(err);
    answer.textContent = '';
    answer.append(el('p', 'error', message));
  } finally {
    clearInterval(status);
    bot.classList.remove('pending');
    $<HTMLButtonElement>('ask-btn').disabled = false;
    if (answering === controller) answering = null;
  }
}

// ---- wiring ----

$('repo-form').addEventListener('submit', (e) => {
  e.preventDefault();
  const value = $<HTMLInputElement>('repo-input').value;
  const ex = findExample(value);
  if (ex && sizeSelect.value === 'quick') void openExample(ex);
  else void indexRepo(value);
});
$('cancel-btn').addEventListener('click', () => indexing?.abort());

const askInput = $<HTMLTextAreaElement>('ask-input');
const autosize = () => {
  askInput.style.height = 'auto';
  askInput.style.height = `${askInput.scrollHeight}px`;
};
askInput.addEventListener('input', autosize);
askInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    $<HTMLFormElement>('ask-form').requestSubmit();
  }
});
$('ask-form').addEventListener('submit', (e) => {
  e.preventDefault();
  const q = askInput.value.trim();
  if (!q || $<HTMLButtonElement>('ask-btn').disabled) return;
  askInput.value = '';
  autosize();
  void ask(q);
});

const dialog = $<HTMLDialogElement>('settings');
$('models').replaceChildren(...SUGGESTED_MODELS.map((m) => Object.assign(el('option'), { value: m })));
function openSettings(): void {
  $<HTMLInputElement>('key-input').value = settings.apiKey;
  $<HTMLInputElement>('gh-input').value = settings.githubToken;
  $<HTMLInputElement>('model-input').value = settings.model;
  $<HTMLSelectElement>('agent-input').value = settings.agent;
  $<HTMLSelectElement>('engine-input').value = settings.engine;
  $<HTMLInputElement>('remember-input').checked = store.persisted(KEY_STORAGE) || store.persisted(GH_TOKEN_STORAGE);
  dialog.showModal();
}
$('settings-btn').addEventListener('click', openSettings);
$('mode-badge').addEventListener('click', openSettings);
dialog.addEventListener('close', () => {
  if (dialog.returnValue !== 'save') return;
  const remember = $<HTMLInputElement>('remember-input').checked;
  settings.apiKey = $<HTMLInputElement>('key-input').value.trim();
  settings.model = $<HTMLInputElement>('model-input').value.trim() || DEFAULT_MODEL;
  store.set(KEY_STORAGE, settings.apiKey || null, remember);
  settings.githubToken = $<HTMLInputElement>('gh-input').value.trim();
  store.set(GH_TOKEN_STORAGE, settings.githubToken || null, remember);
  store.set(MODEL_STORAGE, settings.model === DEFAULT_MODEL ? null : settings.model, true);
  settings.agent = agentMode($<HTMLSelectElement>('agent-input').value);
  store.set(AGENT_STORAGE, settings.agent === 'auto' ? null : settings.agent, true);
  settings.engine = engineOf($<HTMLSelectElement>('engine-input').value);
  store.set(ENGINE_STORAGE, settings.engine === 'cloud' ? null : settings.engine, true);
  if (settings.engine === 'local') void getLocalLlm().load().catch(() => {});
  updateModeNote();
});

// ---- workspace tabs: Ask | Map ----

type Tab = 'ask' | 'map';
let mapState: { index: RepoIndex; render: (highlight?: Hit[]) => void } | null = null;
let lastHits: Hit[] = [];

function showTab(tab: Tab): void {
  for (const t of ['ask', 'map'] as const) {
    $(`tab-${t}`).setAttribute('aria-selected', String(t === tab));
    $(`${t}-panel`).hidden = t !== tab;
  }
  if (tab === 'map') void openMap();
}

async function openMap(): Promise<void> {
  if (!index) return;
  const panel = $('map-panel');
  if (mapState?.index === index) {
    mapState.render(lastHits);
    return;
  }
  const current = index;
  panel.replaceChildren(el('p', 'muted', 'Laying out the code…'));
  const { renderMap } = await import('./map');
  if (index !== current) return;
  mapState = {
    index: current,
    render: renderMap(panel, current, {
      apiKey: () => settings.apiKey,
      model: () => settings.model,
      blobUrl: (path, start, end) => blobUrl(current, path, start, end),
      topics: example?.index === current ? example.example.topics : undefined,
      onAsk: (q) => {
        showTab('ask');
        void ask(q);
      },
    }),
  };
  mapState.render(lastHits);
}

$('tab-ask').addEventListener('click', () => showTab('ask'));
$('tab-map').addEventListener('click', () => showTab('map'));

// ---- views and start: /, /repo?repo=owner/name&q=question, /benchmark ----

let benchmarkShown = false;
async function showBenchmark(): Promise<void> {
  if (benchmarkShown) return;
  benchmarkShown = true;
  const { renderBenchmark } = await import('./benchmark');
  renderBenchmark($('benchmark'));
}

/** Shows whatever the current URL points at, opening the repo if it isn't open yet. */
function route(): void {
  const url = new URL(location.href);
  const view: View = viewOf(url);
  showError(null);
  if (view === 'benchmark') {
    showView('benchmark');
    void showBenchmark();
    return;
  }
  const repo = url.searchParams.get('repo');
  if (view === 'repo' && repo && parseRepo(repo)) {
    showView('repo');
    const question = url.searchParams.get('q')?.trim() || undefined;
    const open = index && `${index.owner}/${index.repo}`.toLowerCase() === repo.toLowerCase();
    if (open || indexing) return;
    $<HTMLInputElement>('repo-input').value = repo;
    const ex = findExample(repo);
    if (ex) void openExample(ex, question);
    else
      void indexRepo(repo, { confirm: true }).then((opened) => {
        if (opened && question) void ask(question);
      });
    return;
  }
  if (view === 'repo') go('home', '', true);
  showView('home');
}

// The bookmarklet: opens the current GitHub repo here. Built from this site's own address.
{
  const site = `${location.origin}${import.meta.env.BASE_URL}repo?repo=`;
  // No regex or backslashes: they don't survive being a URL reliably.
  const code = `(()=>{const p=location.pathname.split('/');if(location.hostname!=='github.com'||!p[1]||!p[2]){alert('Open a GitHub repository page first.');return}window.open(${JSON.stringify(site)}+encodeURIComponent(p[1]+'/'+p[2]),'_blank','noopener')})()`;
  const bm = $<HTMLAnchorElement>('bookmarklet');
  bm.href = `javascript:${code}`;
  bm.addEventListener('click', (e) => {
    e.preventDefault();
    showError(null);
    bm.textContent = 'Drag me to your bookmarks bar ↑';
    setTimeout(() => (bm.textContent = 'Ask this repo'), 2500);
  });
}

renderExamples();
updateModeNote();
interceptLinks((view) => {
  if (view === 'repo' && index) go('repo', repoParams(`${index.owner}/${index.repo}`));
  else go(view, '');
  if (view === 'benchmark') void showBenchmark();
});
window.addEventListener('popstate', route);
route();
