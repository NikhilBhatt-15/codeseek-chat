// Three views in one page, with real URLs, so the embedding model stays loaded while you move
// between them:   /            home: pitch, examples, how it works
//                 /repo?repo=  a repo's workspace (Ask and Map tabs)
//                 /benchmark   retrieval benchmark results
export type View = 'home' | 'repo' | 'benchmark';

const BASE = import.meta.env.BASE_URL; // "/" in production

/** Which view a URL shows. Old share links (/?repo=owner/name) still open the workspace. */
export function viewOf(url: URL): View {
  const path = url.pathname.slice(BASE.length).replace(/\/+$/, '');
  if (path === 'benchmark') return 'benchmark';
  if (path === 'repo' || url.searchParams.has('repo')) return 'repo';
  return 'home';
}

export const pathFor = (view: View) => `${BASE}${view === 'home' ? '' : view}`;

/** Shows one view's elements ([data-view]) and hides the others. */
export function showView(view: View): void {
  for (const node of document.querySelectorAll<HTMLElement>('main [data-view]')) {
    node.hidden = node.dataset.view !== view;
  }
  document.body.dataset.page = view;
  for (const a of document.querySelectorAll<HTMLAnchorElement>('a[data-route]')) {
    a.classList.toggle('current', a.dataset.route === view);
  }
}

/** Goes to a view. `search` replaces the query string (pass '' to clear it). */
export function go(view: View, search?: URLSearchParams | string, replace = false): void {
  const url = new URL(location.href);
  url.pathname = pathFor(view);
  if (search !== undefined) url.search = typeof search === 'string' ? search : search.toString();
  url.hash = '';
  if (url.href !== location.href) (replace ? history.replaceState : history.pushState).call(history, null, '', url);
  showView(view);
  if (!replace) window.scrollTo({ top: 0 });
}

/** In-app links: <a data-route="benchmark" href="/benchmark"> navigate without a reload. */
export function interceptLinks(onRoute: (view: View) => void): void {
  document.addEventListener('click', (e) => {
    const a = (e.target as HTMLElement).closest<HTMLAnchorElement>('a[data-route]');
    if (!a || e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    e.preventDefault();
    onRoute(a.dataset.route as View);
  });
}
