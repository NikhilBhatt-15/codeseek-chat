// Turning retrieved chunks into a grounded prompt, and citations back into links.
import type { Hit } from './repoIndex';

export function systemPrompt(repo: string): string {
  return [
    `You are a senior engineer answering questions about the GitHub repository ${repo}.`,
    'Each question comes with numbered source excerpts retrieved from the repository.',
    'Answer from those excerpts. Cite them inline as [1], [2] right after the claims they support.',
    'If the excerpts do not contain the answer, say so plainly and suggest what to search for instead.',
    'Do not invent files, functions or behaviour that the excerpts do not show.',
    'Be concise. Use Markdown, and fenced code blocks with a language tag for code.',
  ].join('\n');
}

const fenceFor = (text: string) => {
  let fence = '```';
  while (text.includes(fence)) fence += '`';
  return fence;
};

const LANGS: Record<string, string> = {
  py: 'python', ts: 'typescript', tsx: 'tsx', js: 'javascript', jsx: 'jsx', rs: 'rust', go: 'go',
  java: 'java', kt: 'kotlin', rb: 'ruby', cs: 'csharp', cpp: 'cpp', c: 'c', h: 'c', md: 'markdown',
  sh: 'bash', yml: 'yaml', yaml: 'yaml', json: 'json', toml: 'toml', html: 'html', css: 'css', sql: 'sql',
};

export function userPrompt(question: string, hits: Hit[]): string {
  const sources = hits.map((h, i) => {
    const ext = h.path.split('.').pop()?.toLowerCase() ?? '';
    const fence = fenceFor(h.text);
    return `[${i + 1}] ${h.path} (lines ${h.startLine}-${h.endLine})\n${fence}${LANGS[ext] ?? ''}\n${h.text}\n${fence}`;
  });
  return `Sources:\n\n${sources.join('\n\n')}\n\nQuestion: ${question}`;
}

/** The text used for retrieval: follow-ups like "and where is it called?" need the previous question. */
export function retrievalQuery(question: string, previousQuestion?: string): string {
  const short = question.trim().split(/\s+/).length <= 6;
  return short && previousQuestion ? `${previousQuestion}\n${question}` : question;
}

/**
 * Rewrites citation markers like [2] into Markdown links. Markers inside code spans/blocks
 * and index-like brackets (`arr[1]`) are left alone, as are numbers with no matching source.
 */
export function linkCitations(markdown: string, urlFor: (n: number) => string | null): string {
  const parts = markdown.split(/(```[\s\S]*?```|`[^`\n]*`)/);
  return parts
    .map((part, i) => {
      if (i % 2 === 1) return part; // code
      return part.replace(/(?<![\w\\])\[(\d{1,2})\](?![(:])/g, (m, num: string) => {
        const url = urlFor(Number(num));
        return url ? `[[${num}]](${url})` : m;
      });
    })
    .join('');
}
