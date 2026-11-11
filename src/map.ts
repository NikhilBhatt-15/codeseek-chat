// The "Map of the code" tab: every chunk as a dot, laid out so similar code sits together,
// colored by topic. The latest question's matches are ringed, so retrieval is visible.
import { clusterCount, describeClusters, kmeans, prefixRows, project, rng, type ClusterInfo } from './mapLayout';
import { EMBED_DIM } from './protocol';
import type { Hit, RepoIndex } from './repoIndex';

export interface MapOptions {
  apiKey: () => string;
  model: () => string;
  blobUrl: (path: string, start?: number, end?: number) => string;
  onAsk: (question: string) => void;
}

interface Layout {
  points: [number, number][];
  assign: number[];
  clusters: ClusterInfo[];
}

const layouts = new Map<string, Promise<Layout>>();
const SVG = 'http://www.w3.org/2000/svg';
const W = 1000;
const H = 620;
const PAD = 24;

const el = <K extends keyof HTMLElementTagNameMap>(tag: K, cls?: string, text?: string) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};
const svg = <K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number> = {}) => {
  const node = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  return node;
};

async function computeLayout(index: RepoIndex, onProgress: (f: number) => void): Promise<Layout> {
  const rows = prefixRows(index.vectors, EMBED_DIM, Math.min(128, EMBED_DIM));
  const assign = kmeans(rows, clusterCount(rows.length), rng(7));
  const points = await project(rows, rng(42), onProgress);
  return { points, assign, clusters: describeClusters(index.chunks, assign) };
}

/** Asks Gemini for a 1-3 word name per cluster. Falls back silently to the keyword names. */
async function nameWithGemini(clusters: ClusterInfo[], repo: string, apiKey: string, model: string): Promise<Map<number, string>> {
  const brief = clusters
    .map((c) => `Cluster ${c.id} (${c.size} chunks)\nfiles: ${c.files.slice(0, 6).join(', ')}\nidentifiers: ${c.keywords.join(', ')}`)
    .join('\n\n');
  const res = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'x-goog-api-key': apiKey },
    body: JSON.stringify({
      contents: [
        {
          role: 'user',
          parts: [
            {
              text: `These are topic clusters of code chunks from the GitHub repo ${repo}. Give each a short name (1 to 3 words, lowercase, e.g. "retry logic", "url routing", "test helpers") describing what that code does.\n\n${brief}`,
            },
          ],
        },
      ],
      generationConfig: {
        temperature: 0.2,
        responseMimeType: 'application/json',
        responseSchema: {
          type: 'ARRAY',
          items: { type: 'OBJECT', properties: { id: { type: 'INTEGER' }, name: { type: 'STRING' } }, required: ['id', 'name'] },
        },
      },
    }),
  });
  if (!res.ok) throw new Error(`Gemini ${res.status}`);
  const data = (await res.json()) as { candidates?: { content?: { parts?: { text?: string }[] } }[] };
  const text = data.candidates?.[0]?.content?.parts?.map((p) => p.text ?? '').join('') ?? '[]';
  const names = JSON.parse(text) as { id: number; name: string }[];
  return new Map(names.filter((x) => typeof x.name === 'string' && x.name.trim()).map((x) => [x.id, x.name.trim().slice(0, 32)]));
}

/** Draws the map into `panel`; the returned function re-draws with a question's hits ringed. */
export function renderMap(panel: HTMLElement, index: RepoIndex, opts: MapOptions): (highlight?: Hit[]) => void {
  const repo = `${index.owner}/${index.repo}`;
  let layout: Layout | null = null;
  let hits: Hit[] = [];
  let focus: number | null = null; // cluster picked in the legend
  let named = false;

  const status = el('p', 'muted small map-status', 'Laying out the code…');
  const intro = el(
    'p',
    'muted small',
    'Each dot is a piece of code. Code that does similar things sits close together, because its embeddings are similar. Colors are topics found by clustering.',
  );
  const chart = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'map-svg', role: 'img', 'aria-label': `Map of ${repo}'s code` });
  const legend = el('div', 'map-legend');
  const detail = el('div', 'map-detail');
  detail.hidden = true;
  const tip = el('div', 'chart-tip');
  tip.hidden = true;
  panel.replaceChildren(intro, status, chart, legend, detail);
  document.body.append(tip);

  const sx = (x: number) => PAD + x * (W - 2 * PAD);
  const sy = (y: number) => PAD + y * (H - 2 * PAD);

  function showDetail(i: number): void {
    const c = index.chunks[i]!;
    const cluster = layout!.clusters[layout!.assign[i]!]!;
    const head = el('div', 'map-detail-head');
    const a = el('a', '', `${c.path}:${c.startLine}-${c.endLine}`);
    a.href = opts.blobUrl(c.path, c.startLine, c.endLine);
    a.target = '_blank';
    a.rel = 'noopener';
    const ask = el('button', 'small', 'Ask about this code');
    ask.type = 'button';
    ask.onclick = () => opts.onAsk(`What does ${c.path} lines ${c.startLine}-${c.endLine} do?`);
    head.append(a, el('span', `swatch c${cluster.id}`), el('span', 'muted small', cluster.name), ask);
    const pre = el('pre');
    pre.append(el('code', '', c.text.split('\n').slice(0, 24).join('\n')));
    detail.replaceChildren(head, pre);
    detail.hidden = false;
  }

  function draw(): void {
    if (!layout) return;
    const hitSet = new Set(hits.map((h) => h.index));
    const active = hitSet.size > 0 || focus !== null;
    chart.replaceChildren();
    const dots = svg('g', { class: 'map-dots' });
    // Draw ringed hits last, so they sit on top.
    const order = layout.points.map((_, i) => i).sort((a, b) => Number(hitSet.has(a)) - Number(hitSet.has(b)));
    for (const i of order) {
      const [x, y] = layout.points[i]!;
      const cluster = layout.assign[i]!;
      const isHit = hitSet.has(i);
      const dim = active && !isHit && (focus === null ? hitSet.size > 0 : cluster !== focus);
      const dot = svg('circle', { cx: sx(x).toFixed(1), cy: sy(y).toFixed(1), r: isHit ? 8 : 5, class: `dot c${cluster}${isHit ? ' hit' : ''}${dim ? ' dim' : ''}` });
      dot.addEventListener('pointermove', (e) => {
        const c = index.chunks[i]!;
        tip.replaceChildren(el('strong', '', `${c.path}:${c.startLine}-${c.endLine}`), el('div', '', layout!.clusters[cluster]!.name), el('div', 'muted', c.text.split('\n').find((l) => l.trim())?.trim().slice(0, 80) ?? ''));
        tip.hidden = false;
        tip.style.left = `${Math.min(window.innerWidth - tip.offsetWidth - 8, e.clientX + 12)}px`;
        tip.style.top = `${e.clientY + 14}px`;
      });
      dot.addEventListener('pointerleave', () => (tip.hidden = true));
      dot.addEventListener('click', () => showDetail(i));
      dots.append(dot);
    }
    chart.append(dots);
    // Direct labels at each cluster's middle (median point), so names never rely on color alone.
    const labels = svg('g', { class: 'map-labels' });
    for (const c of layout.clusters) {
      const pts = layout.points.filter((_, i) => layout!.assign[i] === c.id);
      const med = (k: 0 | 1) => pts.map((p) => p[k]).sort((a, b) => a - b)[Math.floor(pts.length / 2)]!;
      const text = svg('text', { x: sx(med(0)).toFixed(1), y: (sy(med(1)) - 10).toFixed(1), 'text-anchor': 'middle', class: focus !== null && focus !== c.id ? 'dim' : '' });
      text.textContent = c.name;
      labels.append(text);
    }
    chart.append(labels);

    legend.replaceChildren(
      ...layout.clusters.map((c) => {
        const b = el('button', `legend-item${focus === c.id ? ' on' : ''}`);
        b.type = 'button';
        b.title = `Files: ${c.files.slice(0, 5).join(', ')}`;
        b.append(el('span', `swatch c${c.id}`), el('span', '', c.name), el('span', 'muted', ` ${c.size}`));
        b.onclick = () => {
          focus = focus === c.id ? null : c.id;
          draw();
        };
        return b;
      }),
    );
    status.textContent = hits.length
      ? `Ringed: the ${hits.length} pieces of code that matched your last question. Click a dot to read it.`
      : 'Ask a question in the Ask tab and its matches light up here. Click a dot to read the code, or a topic to focus it.';
  }

  if (!layouts.has(index.key)) {
    layouts.set(index.key, computeLayout(index, (f) => (status.textContent = `Laying out the code… ${Math.round(f * 100)}%`)));
  }
  void layouts.get(index.key)!.then(
    async (l) => {
      layout = l;
      draw();
      const key = opts.apiKey();
      if (key && !named) {
        named = true;
        try {
          const names = await nameWithGemini(l.clusters, repo, key, opts.model());
          for (const c of l.clusters) c.name = names.get(c.id) ?? c.name;
          draw();
        } catch {
          // keep the keyword names
        }
      }
    },
    (err) => (status.textContent = `Couldn't lay out the map: ${err instanceof Error ? err.message : String(err)}`),
  );

  return (highlight?: Hit[]) => {
    hits = highlight ?? [];
    draw();
  };
}
