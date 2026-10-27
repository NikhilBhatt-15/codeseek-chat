import { Embedder } from './embedder';
import { DEFAULT_MODEL, GeminiError, streamChat, SUGGESTED_MODELS, type Turn } from './gemini';
import { blobUrl, fetchContents, fetchSnapshot, GitHubError, parseRepo, type RepoSnapshot } from './github';
import { linkCitations, retrievalQuery, systemPrompt, userPrompt } from './prompt';
import { renderMarkdown } from './render';
import {
  chunkRepo,
  dedupeHits,
  embedChunks,
  indexFromSnapshot,
  indexKey,
  loadCached,
  loadLatestCached,
  saveCached,
  search,
  type Hit,
  type RepoIndex,
} from './repoIndex';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const el = <K extends keyof HTMLElementTagNameMap>(tag: K, cls?: string, text?: string) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};

// ---- settings (per-device conveniences; storage may be unavailable) ----

const KEY_STORAGE = 'codeseek-chat:gemini-key';
const MODEL_STORAGE = 'codeseek-chat:model';
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
  model: store.get(MODEL_STORAGE) ?? DEFAULT_MODEL,
};

// ---- state ----

let embedder: Embedder | null = null;
let index: RepoIndex | null = null;
let history: Turn[] = [];
let lastQuestion: string | undefined;
let indexing: AbortController | null = null;
let answering: AbortController | null = null;

const getEmbedder = () => (embedder ??= new Embedder());

// ---- UI helpers ----

function showError(message: string | null): void {
  const box = $('error');
  box.hidden = !message;
  box.textContent = message ?? '';
}

const STEPS = ['Read the repository from GitHub', 'Download files', 'Load EmbeddingGemma 2', 'Embed code'];

function setStep(active: number, note = '', fraction?: number): void {
  const list = $('steps');
  list.replaceChildren(
    ...STEPS.map((label, i) => {
      const li = el('li', i < active ? 'done' : i === active ? 'active' : '', label);
      return li;
    }),
  );
  $('progress-note').textContent = note;
  if (fraction !== undefined) $('bar-fill').style.width = `${Math.round(fraction * 100)}%`;
}

const mb = (bytes: number) => `${(bytes / 1e6).toFixed(0)} MB`;
const plural = (n: number, word: string) => `${n.toLocaleString()} ${word}${n === 1 ? '' : 's'}`;

function updateModeNote(): void {
  $('mode-note').textContent = settings.apiKey
    ? `Answers by ${settings.model}. Your question and the matching excerpts are sent to Google's Gemini API.`
    : 'Search-only mode: showing the best-matching code. Add a free Gemini API key in Settings to get written answers.';
}

// ---- indexing ----

async function indexRepo(input: string): Promise<void> {
  const ref = parseRepo(input);
  if (!ref) {
    showError('Enter a repository as owner/repo or a github.com URL.');
    return;
  }
  showError(null);
  indexing?.abort();
  const controller = new AbortController();
  indexing = controller;
  const { signal } = controller;

  $<HTMLButtonElement>('index-btn').disabled = true;
  $('progress').hidden = false;
  $('chat').hidden = true;
  $('progress-title').textContent = `Indexing ${ref.owner}/${ref.repo}…`;
  setStep(0, 'Listing files…', 0);

  try {
    const snap = await fetchSnapshot(ref);
    signal.throwIfAborted();
    const cached = await loadCached(indexKey(snap.owner, snap.repo, snap.commit));
    if (cached) {
      openChat(cached, 'loaded from this browser’s cache');
      void getEmbedder().load(); // warm up for the first question
      return;
    }
    if (snap.files.length === 0) throw new GitHubError('No source files found in that repository.');

    // Start the model download while files are fetched.
    const emb = getEmbedder();
    let modelLoaded = 0;
    let modelTotal = 0;
    emb.onLoadProgress = (loaded, total) => {
      modelLoaded = loaded;
      modelTotal = total;
    };
    const modelReady = emb.load();
    modelReady.catch(() => {}); // surfaced below

    setStep(1, `${plural(snap.files.length, 'file')} selected`, 0.05);
    const contents = await fetchContents(
      snap,
      (done, total) => setStep(1, `${done} / ${total} files`, 0.05 + 0.15 * (done / total)),
      signal,
    );
    signal.throwIfAborted();

    const chunks = chunkRepo(contents);
    const timer = setInterval(() => {
      const pct = modelTotal ? modelLoaded / modelTotal : 0;
      setStep(2, modelTotal ? `${mb(modelLoaded)} / ${mb(modelTotal)}. Downloaded once, then cached by your browser.` : 'Starting…', 0.2 + 0.2 * pct);
    }, 250);
    let device: string;
    try {
      device = await modelReady;
    } finally {
      clearInterval(timer);
    }
    signal.throwIfAborted();

    const started = performance.now();
    const vectors = await embedChunks(
      chunks,
      (texts) => emb.embedBatch(texts),
      (done, total) => {
        const secs = (performance.now() - started) / 1000;
        const eta = done > 0 ? Math.round((secs / done) * (total - done)) : 0;
        setStep(3, `${done} / ${plural(total, 'chunk')} on ${device === 'webgpu' ? 'your GPU (WebGPU)' : 'your CPU (WASM)'}${eta > 2 ? `, about ${eta}s left` : ''}`, 0.4 + 0.6 * (done / total));
      },
      signal,
    );
    const files = [...contents.values()].filter(Boolean).length;
    const built = indexFromSnapshot(snap, chunks, vectors, files);
    await saveCached(built);
    openChat(built, describeSkips(snap));
  } catch (err) {
    if (signal.aborted) {
      $('progress').hidden = true;
      return;
    }
    if (err instanceof GitHubError && err.rateLimited) {
      const fallback = await loadLatestCached(ref.owner, ref.repo);
      if (fallback) {
        openChat(fallback, 'cached copy: GitHub rate limit reached, so newer commits were not checked');
        return;
      }
    }
    $('progress').hidden = true;
    showError(err instanceof Error ? err.message : String(err));
  } finally {
    if (indexing === controller) indexing = null;
    $<HTMLButtonElement>('index-btn').disabled = false;
  }
}

function describeSkips(snap: RepoSnapshot): string {
  const notes: string[] = [];
  if (snap.skipped) notes.push(`${plural(snap.skipped, 'file')} skipped to stay within browser limits`);
  if (snap.truncated) notes.push('GitHub truncated the file list (very large repo)');
  return notes.join('; ');
}

function openChat(built: RepoIndex, note: string): void {
  index = built;
  history = [];
  lastQuestion = undefined;
  $('progress').hidden = true;
  $('chat').hidden = false;
  $('intro').hidden = true;
  $('messages').replaceChildren();
  const label = $('repo-label');
  const link = el('a', '', `${built.owner}/${built.repo}`);
  link.setAttribute('href', `https://github.com/${built.owner}/${built.repo}/tree/${built.commit}`);
  link.setAttribute('target', '_blank');
  link.setAttribute('rel', 'noopener');
  label.replaceChildren(link, ` @ ${built.branch ?? built.commit.slice(0, 7)}${built.branch ? ` (${built.commit.slice(0, 7)})` : ''}`);
  $('repo-stats').textContent = [`${plural(built.files, 'file')}, ${plural(built.chunks.length, 'chunk')}`, note]
    .filter(Boolean)
    .join(' · ');
  updateModeNote();
  try {
    setRepoParam(`${built.owner}/${built.repo}`);
  } catch {
    // ignore
  }
  $('ask-input').focus();
}

function setRepoParam(repo: string): void {
  const url = new URL(location.href);
  url.searchParams.set('repo', repo);
  window.history.replaceState(null, '', url);
}

// ---- asking ----

function renderSources(hits: Hit[], open: boolean): HTMLDetailsElement {
  const details = el('details', 'sources');
  details.open = open;
  details.append(el('summary', '', `${plural(hits.length, 'source')} from the repo`));
  hits.forEach((h, i) => {
    const box = el('div', 'source');
    const head = el('div', 'source-head');
    const a = el('a', '', `[${i + 1}] ${h.path}:${h.startLine}-${h.endLine}`);
    a.setAttribute('href', blobUrl(index!, h.path, h.startLine, h.endLine));
    a.setAttribute('target', '_blank');
    a.setAttribute('rel', 'noopener');
    head.append(a, el('span', 'muted', h.score.toFixed(3)));
    const pre = el('pre');
    pre.append(el('code', '', h.text));
    box.append(head, pre);
    details.append(box);
  });
  return details;
}

async function ask(question: string): Promise<void> {
  if (!index || !question.trim()) return;
  showError(null);
  answering?.abort();
  const controller = new AbortController();
  answering = controller;

  const messages = $('messages');
  messages.append(el('div', 'msg user', question));
  const bot = el('div', 'msg bot pending');
  const answer = el('div', 'answer');
  bot.append(answer);
  messages.append(bot);
  bot.scrollIntoView({ block: 'nearest' });
  $<HTMLButtonElement>('ask-btn').disabled = true;

  try {
    const query = await getEmbedder().embedQuery(retrievalQuery(question, lastQuestion));
    const hits = dedupeHits(search(index, query, 16)).slice(0, 8);
    lastQuestion = question;
    const current = index;
    const urlFor = (n: number) => {
      const h = hits[n - 1];
      return h ? blobUrl(current, h.path, h.startLine, h.endLine) : null;
    };

    if (!settings.apiKey) {
      answer.innerHTML = renderMarkdown(
        hits.length ? 'Here’s the code that best matches your question:' : 'No matching code found.',
      );
      bot.append(renderSources(hits.slice(0, 5), true));
      return;
    }

    let text = '';
    let frame = 0;
    const paint = () => {
      frame = 0;
      answer.innerHTML = renderMarkdown(linkCitations(text, urlFor));
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
    bot.append(renderSources(hits, false));
    // Keep a short plain-text history (without the bulky excerpts) for follow-ups.
    history = [...history, { role: 'user' as const, text: question }, { role: 'model' as const, text }].slice(-6);
  } catch (err) {
    if (controller.signal.aborted) return;
    const message = err instanceof GeminiError || err instanceof Error ? err.message : String(err);
    answer.textContent = '';
    answer.append(el('p', 'error', message));
  } finally {
    bot.classList.remove('pending');
    $<HTMLButtonElement>('ask-btn').disabled = false;
    if (answering === controller) answering = null;
  }
}

// ---- wiring ----

$('repo-form').addEventListener('submit', (e) => {
  e.preventDefault();
  void indexRepo($<HTMLInputElement>('repo-input').value);
});
$('examples').addEventListener('click', (e) => {
  const repo = (e.target as HTMLElement).dataset.repo;
  if (!repo) return;
  $<HTMLInputElement>('repo-input').value = repo;
  void indexRepo(repo);
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
const models = $('models');
models.replaceChildren(...SUGGESTED_MODELS.map((m) => Object.assign(el('option'), { value: m })));
$('settings-btn').addEventListener('click', () => {
  $<HTMLInputElement>('key-input').value = settings.apiKey;
  $<HTMLInputElement>('model-input').value = settings.model;
  $<HTMLInputElement>('remember-input').checked = store.persisted(KEY_STORAGE);
  dialog.showModal();
});
dialog.addEventListener('close', () => {
  if (dialog.returnValue !== 'save') return;
  const remember = $<HTMLInputElement>('remember-input').checked;
  settings.apiKey = $<HTMLInputElement>('key-input').value.trim();
  settings.model = $<HTMLInputElement>('model-input').value.trim() || DEFAULT_MODEL;
  store.set(KEY_STORAGE, settings.apiKey || null, remember);
  store.set(MODEL_STORAGE, settings.model === DEFAULT_MODEL ? null : settings.model, true);
  updateModeNote();
});

const shared = new URL(location.href).searchParams.get('repo');
if (shared && parseRepo(shared)) $<HTMLInputElement>('repo-input').value = shared;
updateModeNote();
