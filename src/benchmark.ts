// The /benchmark view: the retrieval benchmark (eval/) as charts and tables.
// Data comes straight from the committed result files, so the page can't drift from the README.
import bench from '../eval/benchmark.json';
import q4 from '../eval/results/q4.json';
import variants from '../eval/results/variants.json';

type Scores = Record<string, number>;
interface Row {
  config: string;
  variant: string;
  scores: Scores;
  byKind: Record<string, Scores>;
}

const rows = q4.rows as Row[];
const variantRows = variants.rows as Row[];
const find = (config: string, variant = 'q4') => rows.find((r) => r.config === config && (variant === '-' || r.variant === variant))!;

const el = <K extends keyof HTMLElementTagNameMap>(tag: K, cls?: string, text?: string) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};
const fmt = (x: number) => x.toFixed(2);

const SHIPPED = 'Embeddings 768d';
const RETRIEVERS: { config: string; variant: string; label: string; note: string }[] = [
  { config: SHIPPED, variant: 'q4', label: 'Embeddings, 768 dims', note: 'shipped' },
  { config: 'Embeddings 256d', variant: 'q4', label: 'Embeddings, 256 dims', note: '' },
  { config: 'Embeddings 128d', variant: 'q4', label: 'Embeddings, 128 dims', note: '' },
  { config: 'Routed 768d (hybrid if code-like)', variant: 'q4', label: 'Hybrid only for code-like queries', note: '' },
  { config: 'Hybrid 768d (RRF)', variant: 'q4', label: 'Hybrid: embeddings + BM25 (RRF)', note: '' },
  { config: 'Hybrid 256d + rerank top 20', variant: 'q4', label: 'Hybrid + cross-encoder rerank', note: '' },
  { config: 'BM25 only', variant: '-', label: 'BM25 keyword search only', note: '' },
];
const METRICS: { key: string; label: string; help: string }[] = [
  { key: 'MRR@10', label: 'MRR@10', help: '1 / rank of the first right chunk, averaged (1.0 = always first)' },
  { key: 'recall@1', label: 'recall@1', help: 'share of answer files found in the first chunk' },
  { key: 'recall@5', label: 'recall@5', help: 'share of answer files found in the top 5 chunks' },
  { key: 'recall@10', label: 'recall@10', help: 'share of answer files found in the top 10 chunks' },
];

// ---- a shared hover tooltip ----

let tip: HTMLDivElement | null = null;
function tooltip(target: HTMLElement, lines: () => string[]): void {
  const show = (x: number, y: number) => {
    tip ??= document.body.appendChild(el('div', 'chart-tip'));
    tip.replaceChildren(...lines().map((l, i) => el(i ? 'div' : 'strong', '', l)));
    tip.hidden = false;
    const w = tip.offsetWidth;
    tip.style.left = `${Math.min(window.innerWidth - w - 8, Math.max(8, x + 12))}px`;
    tip.style.top = `${y + 14}px`;
  };
  target.addEventListener('pointermove', (e) => show(e.clientX, e.clientY));
  target.addEventListener('pointerleave', () => tip && (tip.hidden = true));
  target.addEventListener('focus', () => {
    const r = target.getBoundingClientRect();
    show(r.left, r.top);
  });
  target.addEventListener('blur', () => tip && (tip.hidden = true));
}

// ---- horizontal bar chart (one series; the shipped choice is highlighted) ----

interface Bar {
  label: string;
  value: number;
  shown: string; // value text at the bar's end
  highlight?: boolean;
  tip: string[];
}

function barChart(bars: Bar[], max: number, axisLabel: string): HTMLElement {
  const chart = el('div', 'bars');
  chart.setAttribute('role', 'img');
  chart.setAttribute('aria-label', `${axisLabel}: ${bars.map((b) => `${b.label} ${b.shown}`).join(', ')}`);
  for (const b of bars) {
    const row = el('div', `bar-row${b.highlight ? ' hi' : ''}`);
    row.tabIndex = 0;
    const track = el('div', 'bar-track');
    const fill = el('div', 'bar-fill');
    fill.style.width = `${Math.max(1, (b.value / max) * 100)}%`;
    track.append(fill, el('span', 'bar-value', b.shown));
    row.append(el('span', 'bar-label', b.label), track);
    tooltip(row, () => b.tip);
    chart.append(row);
  }
  const axis = el('div', 'bar-axis');
  axis.append(el('span'), el('span', '', axisLabel));
  chart.append(axis);
  return chart;
}

function table(head: string[], body: (string | HTMLElement)[][], caption: string): HTMLDetailsElement {
  const details = el('details', 'data-table');
  details.append(el('summary', '', caption));
  const t = el('table');
  const tr = el('tr');
  head.forEach((h) => tr.append(el('th', '', h)));
  t.append(el('thead'), el('tbody'));
  t.tHead!.append(tr);
  for (const r of body) {
    const row = el('tr');
    r.forEach((c) => {
      const td = el('td');
      td.append(c);
      row.append(td);
    });
    t.tBodies[0]!.append(row);
  }
  details.append(t);
  return details;
}

// ---- sections ----

function heroStats(): HTMLElement {
  const shipped = find(SHIPPED);
  const stats = el('div', 'stats');
  const tile = (value: string, label: string) => {
    const t = el('div', 'stat');
    t.append(el('div', 'stat-value', value), el('div', 'stat-label', label));
    return t;
  };
  stats.append(
    tile(`${Math.round(shipped.scores['recall@5']! * 100)}%`, 'of answer files are in the top 5 results (recall@5)'),
    tile('6×', 'smaller model (4-bit, 174 MB) with the same score as full precision (1,084 MB)'),
    tile(String(bench.questions.length), `labelled questions over ${bench.repos.length} repos in TypeScript, Go and Python`),
  );
  return stats;
}

function retrieverChart(): HTMLElement {
  const box = el('section', 'chart-card');
  box.append(
    el('h2', '', 'Which retriever finds the right code?'),
    el('p', 'muted', 'Same questions, same index, same model (EmbeddingGemma 2, 4-bit). Higher is better.'),
  );
  const controls = el('div', 'seg');
  controls.setAttribute('role', 'group');
  controls.setAttribute('aria-label', 'Metric');
  const help = el('p', 'muted small metric-help');
  const holder = el('div');
  const draw = (metric: (typeof METRICS)[number]) => {
    for (const b of controls.children) b.setAttribute('aria-pressed', String((b as HTMLElement).dataset.key === metric.key));
    help.textContent = `${metric.label}: ${metric.help}.`;
    const bars = RETRIEVERS.map((r) => {
      const row = find(r.config, r.variant);
      return {
        label: r.note ? `${r.label} (${r.note})` : r.label,
        value: row.scores[metric.key]!,
        shown: fmt(row.scores[metric.key]!),
        highlight: r.config === SHIPPED,
        tip: [r.label, ...METRICS.map((m) => `${m.label}: ${fmt(row.scores[m.key]!)}`)],
      };
    });
    holder.replaceChildren(barChart(bars, 1, metric.label));
  };
  for (const m of METRICS) {
    const b = el('button', 'seg-btn', m.label);
    b.type = 'button';
    b.dataset.key = m.key;
    b.onclick = () => draw(m);
    controls.append(b);
  }
  box.append(controls, help, holder);
  draw(METRICS[0]!);
  box.append(
    table(
      ['Retriever', ...METRICS.map((m) => m.label)],
      RETRIEVERS.map((r) => [r.label, ...METRICS.map((m) => fmt(find(r.config, r.variant).scores[m.key]!))]),
      'Show as a table',
    ),
  );
  const findings = el('ul', 'findings');
  for (const f of [
    'Embeddings alone won. Adding BM25 keyword search with reciprocal rank fusion, the textbook upgrade, made ranking slightly worse.',
    'A cross-encoder reranker (ms-marco-MiniLM, trained on web search) hurt the most: recall@1 fell from 0.64 to 0.44.',
    'Keeping all 768 dimensions beat the 256-dim Matryoshka prefix, for about 1 MB more per repo.',
  ])
    findings.append(el('li', '', f));
  box.append(findings);
  return box;
}

function kindTable(): HTMLElement {
  const box = el('section', 'chart-card');
  const cols = [
    { config: 'BM25 only', variant: '-', label: 'BM25' },
    { config: SHIPPED, variant: 'q4', label: 'Embeddings (shipped)' },
    { config: 'Hybrid 768d (RRF)', variant: 'q4', label: 'Hybrid' },
  ];
  const count = (k: string) => bench.questions.filter((q) => q.kind === k).length;
  const kinds = [...new Set(bench.questions.map((q) => q.kind))];
  const names: Record<string, string> = {
    concept: 'Concept ("how can I limit concurrent requests?")',
    'cross-file': 'Cross-file (answer spans two files)',
    identifier: 'Identifier (ThrottleBacklog)',
    'error-message': 'Pasted error message',
  };
  box.append(el('h2', '', 'Where each one wins'), el('p', 'muted', 'recall@5 by question kind. Keyword search only wins on pasted error messages.'));
  const t = el('table', 'kind-table');
  const head = el('tr');
  ['Question kind', ...cols.map((c) => c.label)].forEach((h) => head.append(el('th', '', h)));
  t.append(el('thead'), el('tbody'));
  t.tHead!.append(head);
  for (const k of kinds) {
    const row = el('tr');
    row.append(el('td', '', `${names[k] ?? k} · ${count(k)}`));
    const values = cols.map((c) => find(c.config, c.variant).byKind[k]!['recall@5']!);
    const best = Math.max(...values);
    values.forEach((v) => row.append(el('td', v === best ? 'best' : '', fmt(v))));
    t.tBodies[0]!.append(row);
  }
  box.append(t);
  return box;
}

function sizeChart(): HTMLElement {
  const box = el('section', 'chart-card');
  box.append(
    el('h2', '', 'Does a smaller model cost quality?'),
    el('p', 'muted', 'The same embedding model at four precisions. Bar length is the download every visitor pays; the label is its MRR@10.'),
  );
  const sizes: Record<string, number> = { fp32: 1084, fp16: 542, q8: 314, q4: 174 };
  const bars = ['fp32', 'fp16', 'q8', 'q4'].map((v) => {
    const row = variantRows.find((r) => r.config === SHIPPED && r.variant === v)!;
    return {
      label: v === 'q4' ? 'q4, 4-bit (shipped)' : v,
      value: sizes[v]!,
      shown: `${sizes[v]!.toLocaleString()} MB · MRR ${fmt(row.scores['MRR@10']!)}`,
      highlight: v === 'q4',
      tip: [`${v}: ${sizes[v]!.toLocaleString()} MB download`, ...METRICS.map((m) => `${m.label}: ${fmt(row.scores[m.key]!)}`)],
    };
  });
  box.append(barChart(bars, 1084, 'Download size (MB)'));
  box.append(
    table(
      ['Weights', 'Download', ...METRICS.map((m) => m.label)],
      ['fp32', 'fp16', 'q8', 'q4'].map((v) => {
        const row = variantRows.find((r) => r.config === SHIPPED && r.variant === v)!;
        return [v, `${sizes[v]!.toLocaleString()} MB`, ...METRICS.map((m) => fmt(row.scores[m.key]!))];
      }),
      'Show as a table',
    ),
  );
  box.append(el('p', 'callout', 'The 4-bit model scores within one question of full precision on every metric, at a sixth of the download.'));
  return box;
}

function questionsList(): HTMLElement {
  const box = el('section', 'chart-card');
  box.append(
    el('h2', '', 'The questions'),
    el(
      'p',
      'muted',
      'Each question is labelled with the file(s) that answer it. Identifier and error-message labels come from grep. Repos are pinned to a commit and indexed like the site\'s Quick mode.',
    ),
  );
  box.append(
    table(
      ['Repo', 'Kind', 'Question', 'Answer file(s)'],
      bench.questions.map((q) => [q.repo.split('/')[1]!, q.kind, q.question, q.files.join(', ')]),
      `Show all ${bench.questions.length} questions`,
    ),
  );
  const caveat = el('p', 'muted small');
  caveat.append(
    '64 questions is a small set: one question moves recall@5 by about 1.6 points, so gaps under ~3 points are noise. Reproduce with ',
    el('code', '', 'npm run eval'),
    ' in the ',
  );
  const a = el('a', '', 'repository');
  a.href = 'https://github.com/NikhilBhatt-15/codeseek-chat/tree/main/eval';
  a.target = '_blank';
  a.rel = 'noopener';
  caveat.append(a, '.');
  box.append(caveat);
  return box;
}

export function renderBenchmark(root: HTMLElement): void {
  const intro = el('div', 'intro');
  intro.append(
    el('h1', '', 'Retrieval, measured.'),
    el(
      'p',
      '',
      'Every choice in codeseek chat\'s search was tested on a benchmark of real questions about real repos, including the upgrades that didn\'t make it in.',
    ),
  );
  root.replaceChildren(intro, heroStats(), retrieverChart(), kindTable(), sizeChart(), questionsList());
}
