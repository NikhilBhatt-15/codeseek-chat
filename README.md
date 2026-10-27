# codeseek chat

Ask any public GitHub repository a question, in a web page with no backend.

**Live:** https://nikhilbhatt-15.github.io/codeseek-chat/

The repository's code is embedded **in your browser** with Google DeepMind's
[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2), using
[Transformers.js](https://github.com/huggingface/transformers.js) on WebGPU (or WebAssembly
as a fallback). Your question retrieves the most relevant code. With a free Gemini API key,
Gemini writes an answer that cites `file:line` links into GitHub. Without a key you still get
semantic search.

It's the browser companion to [codeseek](https://github.com/NikhilBhatt-15/codeseek), the
command-line tool, and uses the same chunking rules.

## Features

- **No server.** It's a static site on GitHub Pages. Nothing to host and nothing to pay for.
- **Private by default.** Indexing happens on your device. Code goes to Gemini only when you
  ask a question with a key set, and then only the matching excerpts are sent.
- **Bring your own key.** The key goes straight from your browser to Google, in a request
  header. It's kept for the session, or on this device if you choose.
- **Cited answers.** `[1]`, `[2]` markers link to the exact lines on GitHub, and the excerpts
  used are shown under each answer.
- **Fast revisits.** Indexes are cached in IndexedDB per commit. Reopening a repo takes about 2 s.
- **Follow-up questions.** Short follow-ups ("and where is it called?") reuse the previous
  question for retrieval.

## How it works

1. **Read the repo.** Three GitHub API calls resolve the default branch, its commit and the
   file tree. File contents come from `raw.githubusercontent.com`, which doesn't count toward
   GitHub's limit of 60 API requests an hour.
2. **Pick files.** It keeps source files up to 400 files and 2.5 MB, preferring code over docs
   and tests. Vendored code, lockfiles, minified files and binaries are skipped, as in the CLI.
3. **Chunk.** Files are split at top-level declarations, with overlapping windows for long
   blocks. This is a TypeScript port of codeseek's chunker, checked by the same tests.
4. **Embed.** A Web Worker loads only the 270M text encoder (q4, about 175 MB, downloaded
   once and then cached by the browser). Each chunk is embedded as
   `title: <path> | text: <code>`, cut to 256 dimensions (Matryoshka) and normalised.
5. **Answer.** The question is embedded with the code-retrieval prompt, matched by exact
   cosine similarity, and the top 8 excerpts are sent to Gemini (streaming) with instructions
   to cite them.

Measured in headless Edge: the first visit downloads the model in about 25 s. Indexing
codeseek's 21 files (72 chunks) took about 60 s, and each question's retrieval takes 0.2 to 0.4 s.

## Limits

- Public repositories only. Very large repos are sampled: the file count and skipped files
  are shown.
- GitHub allows 60 unauthenticated API requests an hour per network, and each repo uses 3.
- Gemini's free tier has its own rate limits per key.
- Without WebGPU, embedding runs on the CPU through WebAssembly and is several times slower.

## Development

```sh
npm install
npm run dev        # http://localhost:5173
npm test           # unit tests (chunking, GitHub, retrieval, prompts, SSE parsing)
npm run build      # type-check and production build into dist/
```

Pushing to `main` runs the tests and deploys to GitHub Pages through
`.github/workflows/deploy.yml`.

## License

MIT. EmbeddingGemma 2 is distributed by Google under Apache 2.0.
