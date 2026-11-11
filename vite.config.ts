import { defineConfig } from 'vitest/config';

// Served from the site root on Netlify. /repo and /benchmark are client-side views
// (public/_redirects sends them to index.html).
export default defineConfig({
  base: '/',
  worker: { format: 'es' },
  build: { target: 'es2022' },
  test: { environment: 'node' },
});
