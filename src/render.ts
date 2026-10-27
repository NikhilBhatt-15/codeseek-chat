// Markdown from the model is untrusted: render it, then sanitise before it touches the DOM.
import DOMPurify from 'dompurify';
import { marked } from 'marked';

DOMPurify.addHook('afterSanitizeAttributes', (node) => {
  if (node.tagName === 'A') {
    node.setAttribute('target', '_blank');
    node.setAttribute('rel', 'noopener noreferrer');
  }
});

export function renderMarkdown(markdown: string): string {
  const html = marked.parse(markdown, { async: false, gfm: true, breaks: false });
  return DOMPurify.sanitize(html, { FORBID_TAGS: ['style', 'form', 'input', 'img'], FORBID_ATTR: ['style'] });
}
