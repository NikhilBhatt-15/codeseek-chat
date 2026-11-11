import { describe, expect, it } from 'vitest';
import { linkCitations } from '../src/prompt';

describe('citations before a colon', () => {
  const url = (n: number) => `https://x/${n}`;
  it('links "[7]:" mid-sentence but not reference definitions', () => {
    expect(linkCitations('after a failed attempt [7]:', url)).toBe('after a failed attempt [[7]](https://x/7):');
    expect(linkCitations('[1]: https://example.com', url)).toBe('[1]: https://example.com');
  });
});

describe('grouped citations', () => {
  const url = (n: number) => (n <= 6 ? `https://x/${n}` : null);
  it('links each number in [1, 6]', () => {
    expect(linkCitations('merged [1, 6].', url)).toBe('merged [[1]](https://x/1)[[6]](https://x/6).');
  });
  it('leaves groups with an unknown source alone', () => {
    expect(linkCitations('see [1, 9]', url)).toBe('see [1, 9]');
  });
});
