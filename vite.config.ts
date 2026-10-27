import { defineConfig } from 'vitest/config';

// Served from https://<user>.github.io/codeseek-chat/ on GitHub Pages.
export default defineConfig({
  base: './',
  worker: { format: 'es' },
  build: { target: 'es2022' },
  test: { environment: 'node' },
});
