// Adds an "Ask this repo" button on GitHub repository pages. No permissions: it only reads the
// page URL and opens codeseek chat in a new tab. GitHub navigates without full page loads,
// so the button is re-checked whenever the URL changes.
(() => {
  const SITE = 'https://codeseek-chat.netlify.app/repo';
  // First path segments that are GitHub pages, not owners.
  const RESERVED = new Set([
    'settings', 'orgs', 'organizations', 'marketplace', 'explore', 'topics', 'notifications', 'login', 'logout',
    'signup', 'search', 'trending', 'collections', 'sponsors', 'features', 'pricing', 'about', 'apps', 'new',
    'codespaces', 'issues', 'pulls', 'discussions', 'stars', 'watching', 'dashboard', 'enterprise', 'team',
    'customer-stories', 'readme', 'security', 'site', 'contact', 'account', 'copilot', 'github-copilot', 'models',
  ]);

  function currentRepo() {
    const [owner, repo] = location.pathname.split('/').filter(Boolean);
    if (!owner || !repo || RESERVED.has(owner.toLowerCase())) return null;
    if (!/^[\w.-]+$/.test(owner) || !/^[\w.-]+$/.test(repo)) return null;
    return `${owner}/${repo.replace(/\.git$/, '')}`;
  }

  function update() {
    const repo = currentRepo();
    let button = document.getElementById('codeseek-ask');
    if (!repo) {
      button?.remove();
      return;
    }
    if (!button) {
      button = document.createElement('a');
      button.id = 'codeseek-ask';
      button.target = '_blank';
      button.rel = 'noopener';
      button.innerHTML =
        '<svg viewBox="0 0 32 32" aria-hidden="true"><circle cx="13" cy="13" r="8"/><path d="M19 19l8 8"/></svg><span>Ask this repo</span>';
      document.body.append(button);
    }
    button.href = `${SITE}?repo=${encodeURIComponent(repo)}`;
    button.title = `Ask questions about ${repo} with codeseek chat`;
  }

  let last = '';
  const check = () => {
    if (location.pathname !== last) {
      last = location.pathname;
      update();
    }
  };
  document.addEventListener('turbo:load', check);
  window.addEventListener('popstate', check);
  setInterval(check, 1000);
  check();
})();
