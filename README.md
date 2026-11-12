# codeseek chat

Ask any public GitHub repository a question, in a web page with no backend.

**Live:** https://codeseek-chat.netlify.app/

The repository's code is embedded **in your browser** with Google DeepMind's
[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2), using
[Transformers.js](https://github.com/huggingface/transformers.js) on WebGPU (or WebAssembly
as a fallback). Your question retrieves the most relevant code. With a free Gemini API key,
Gemini writes an answer that cites `file:line` links into GitHub. Without a key you still get
semantic search, or you can switch on **offline AI**: a small LLM that writes answers in your
browser with no key, so nothing leaves your device.

Retrieval is measured, not assumed: a 64-question benchmark over four repos compares embeddings,
BM25, hybrid fusion, a reranker, vector sizes and model precisions. [Results below](#retrieval-evaluation),
and on the site's [benchmark page](https://codeseek-chat.netlify.app/benchmark).

It's the browser companion to [codeseek](https://github.com/NikhilBhatt-15/codeseek), the
command-line tool, and uses the same chunking rules.

## Features

- **Three pages.** Home (examples and how it works), a repo workspace at
  `/repo?repo=owner/name` with **Ask** and **Map** tabs, and the
  [benchmark](https://codeseek-chat.netlify.app/benchmark) with charts of the eval results. It's one
  page app with real URLs, so the models stay loaded as you move around.
- **Map of the code.** Every chunk is a dot, laid out with UMAP so that code doing similar things
  sits together, and colored by topic (k-means on the embeddings). Topics are named by Gemini when
  a key is set, or by each cluster's most distinctive identifiers when not. The matches for your
  latest question are ringed, so you can see retrieval happen. Click a dot to read the code.
- **Offline AI.** Choose *Offline AI* in Settings, or click the button under a search result, and
  [Qwen3.5 0.8B](https://huggingface.co/onnx-community/Qwen3.5-0.8B-Text-ONNX) (469 MB, 4-bit,
  downloaded once) writes the answer on your GPU with the same `[1]` citations. No key, no server.
- **Ask from GitHub.** A bookmarklet on the home page, or a small Chrome extension (in
  [Releases](https://github.com/NikhilBhatt-15/codeseek-chat/releases)) that adds an
  **Ask this repo** button to every GitHub repo page. The extension asks for no permissions.
- **Instant examples.** Three example repos ship with pre-built indexes, each with example
  questions that answer the moment you click. The model downloads in the background while you read.
- **Shareable links.** `/repo?repo=owner/repo&q=question` opens a repo and asks the question.
  Every answer has a **Copy link to this answer** button. For example:
  [where does ky decide to retry a request?](https://codeseek-chat.netlify.app/repo?repo=sindresorhus/ky&q=Where%20does%20ky%20decide%20whether%20to%20retry%20a%20failed%20request%3F)
- **How it found this.** Each answer shows the best-matching files with their similarity scores.
- **Multi-step search.** With a key, Gemini drives three tools over the in-browser index:
  `search_code`, `find_references` (exact, whole-word) and `read_lines`. For questions like "where
  is this called?" it searches, follows the identifier to its callers, reads more code, then
  answers. The steps it took are shown above the answer. *Auto* (the default) uses it for
  reference-style questions; it can be turned off or always on in Settings.
- **No server.** It's a static site on Netlify. Nothing to host and nothing to pay for.
- **Private by default.** Indexing happens on your device. Code goes to Gemini only when you
  ask a question with a key set, and then only the matching excerpts are sent.
- **Bring your own key.** The key goes straight from your browser to Google, in a request
  header. It's kept for the session, or on this device if you choose.
- **Cited answers.** `[1]`, `[2]` markers link to the exact lines on GitHub, and the excerpts
  used are shown under each answer.
- **Quick or Full.** Quick is the default. You see a time estimate for your device before
  indexing starts (shared links wait for a click). Extend to Full later without re-embedding what's done.
- **Search without a key.** Without a Gemini key you get the matching code as cards with file
  names, line numbers and links into GitHub, plus a link to get a free key.
- **Phone-friendly.** On phones the examples still answer instantly. The 175 MB model only
  downloads if you ask for it, since a desktop browser with WebGPU works much better.
- **Fast revisits.** Indexes are cached in IndexedDB per commit. Reopening a repo takes about 2 s.
- **Follow-up questions.** Short follow-ups ("and where is it called?") reuse the previous
  question for retrieval.

## How it works

1. **Read the repo.** Two GitHub API calls resolve the latest commit and the file tree. File contents come from `raw.githubusercontent.com`, which doesn't count toward
   GitHub's limit of 60 API requests an hour.
2. **Pick files.** Code over docs and tests, within the size you choose: **Quick** (up to
   400 KB, about 400 chunks, a few minutes on a laptop GPU) or **Full** (up to 2.5 MB).
   A Quick index offers **Index the rest**, which adds just the skipped files at the same
   commit. Time estimates use your device's measured speed. Vendored code, lockfiles, minified files and binaries are skipped, as in the CLI.
3. **Chunk.** Files are split at top-level declarations, with overlapping windows for long
   blocks. This is a TypeScript port of codeseek's chunker, checked by the same tests.
4. **Embed.** A Web Worker loads only the 270M text encoder (q4, about 175 MB, downloaded
   once and then cached by the browser). Each chunk is embedded as
   `title: <path> | text: <code>` and normalised, keeping all 768 dimensions (see below for why).
5. **Answer.** The question is embedded with the code-retrieval prompt, matched by exact
   cosine similarity, and the top 8 excerpts are sent to Gemini (streaming) with instructions
   to cite them. Reference-style questions go through the multi-step agent instead.

Measured in headless Edge: the first visit downloads the model in about 25 s. Indexing
codeseek's 21 files (72 chunks) took about 60 s, and each question's retrieval takes 0.2 to 0.4 s.

## Retrieval evaluation

`npm run eval` scores retrieval on a fixed benchmark ([`eval/benchmark.json`](eval/benchmark.json)):
**64 questions over 4 repos** in 3 languages, each labelled with the file(s) that answer it.

- Repos, pinned to a commit: [ky](https://github.com/sindresorhus/ky) (TypeScript, 349 chunks),
  [chi](https://github.com/go-chi/chi) (Go, 398), [itsdangerous](https://github.com/pallets/itsdangerous)
  (Python, 85) and [codeseek](https://github.com/NikhilBhatt-15/codeseek) (Python, 72), indexed exactly like
  the site's Quick mode.
- Question kinds: 38 *concept* ("how can I limit concurrent requests?"), 6 *cross-file* (the answer
  spans two files), 14 *identifier* (`ThrottleBacklog`) and 6 *error message*
  (`panic: chi: all middlewares must be defined before routes on a mux`). Identifier and error
  labels come from `grep` for the defining file.
- **recall@k**: share of a question's answer files found in the top *k* chunks (after removing
  overlapping chunks, as the app does). **MRR@10**: 1 / rank of the first answer chunk.

All with the shipped model (EmbeddingGemma 2, q4):

| Retriever | recall@1 | recall@5 | recall@10 | MRR@10 |
|---|---|---|---|---|
| BM25 only (code-aware tokens) | 0.38 | 0.79 | 0.89 | 0.58 |
| **Embeddings, 768 dims (shipped)** | **0.68** | **0.94** | **1.00** | **0.81** |
| Embeddings, 256 dims | 0.64 | 0.91 | 1.00 | 0.77 |
| Embeddings, 128 dims | 0.58 | 0.88 | 0.96 | 0.72 |
| Hybrid 768d: embeddings + BM25, reciprocal rank fusion | 0.63 | 0.92 | 0.95 | 0.77 |
| Hybrid 256d | 0.59 | 0.90 | 0.97 | 0.74 |
| Routed 768d: hybrid only for code-like queries | 0.66 | 0.94 | 1.00 | 0.80 |
| Embeddings 768d, then its top 20 reranked by a cross-encoder (ms-marco-MiniLM-L-6) | 0.50 | 0.87 | 0.94 | 0.68 |

recall@5 by question kind:

| Retriever | concept (38) | cross-file (6) | identifier (14) | error message (6) |
|---|---|---|---|---|
| BM25 only | 0.79 | 0.58 | 0.79 | **1.00** |
| Embeddings 768d | **0.95** | **1.00** | **0.93** | 0.83 |
| Hybrid 768d (RRF) | **0.95** | 0.83 | **0.93** | 0.83 |

What this changed in the app:

- **Hybrid search didn't earn its place.** The textbook upgrade, BM25 + reciprocal rank fusion,
  was slightly *worse* than embeddings alone (MRR 0.77 vs 0.81), and it lost a cross-file
  question (recall@5 0.83 vs 1.00 on that kind). BM25 *alone* wins on pasted error messages
  (6/6 vs 5/6), but mixing it in as hybrid didn't help even there (5/6). Exact matching moved to
  where it is reliable: the agent's `find_references` tool.
- **This reranker hurt a lot.** Reordering the shipped ranking's own top 20 with
  ms-marco-MiniLM-L-6, the right file came first for 53% of questions instead of 72% (hit@1),
  and MRR fell from 0.81 to 0.68. It mostly promoted tests: a test file was ranked first for 13
  questions instead of 6 (implementation code: 46 instead of 55). That model is trained on web
  search passages, not code, so this is a result about *a web-trained reranker*, not rerankers
  in general. A code-trained one (bge-reranker-v2-m3, for example) wasn't tested and might help.
- **768 dims instead of 256.** With the shipped q4 model, the full vector beat the Matryoshka
  prefix on every metric (recall@1 +4 points, MRR +4). At the other precisions 768 still wins
  on recall@1 and MRR, while recall@5 is mixed. The cost is 3× the vector storage (about 1 MB for
  a 350-chunk repo), and search stays around 1 ms, so the app now keeps all 768.

### Model size: q4 vs full precision

The same model at four precisions (768 dims, same 64 questions):

| Weights | Download | recall@1 | recall@5 | recall@10 | MRR@10 |
|---|---|---|---|---|---|
| fp32 | 1,084 MB | 0.67 | 0.92 | 0.98 | 0.80 |
| fp16 | 542 MB | 0.67 | 0.92 | 0.98 | 0.80 |
| q8 | 314 MB | 0.67 | 0.92 | 0.98 | 0.80 |
| **q4 (shipped)** | **174 MB** | **0.68** | **0.94** | **1.00** | **0.81** |

**The 4-bit model is 6× smaller than full precision and lost nothing measurable.** It scored
within one question of fp32 on every metric. For a model every visitor downloads, that's the
whole difference between a usable first visit and a 1 GB wait.

### Answer quality: offline model vs Gemini

`npm run eval:answers` gives both models the same top 8 excerpts for each of the 64 questions
and scores the written answers.

| Model | Cites the right file¹ | Cites anything | Valid citations |
|---|---|---|---|
| Gemini Flash Lite (API, your key) | 88% | 98% | 98% |
| Qwen3.5 0.8B (offline, in the browser) | 61% | 72% | 97% |

¹ The share of answers that cite an excerpt from a labelled answer file (retrieval put one in the
top 8 for all 64 questions).

**The free offline model gets about 70% of Gemini's score**, with no key and nothing leaving the
device. Most of the gap is answers that cite nothing: 28% of the offline model's answers, against
2% of Gemini's. A graded comparison (gpt-oss-120b as the judge, so Gemini doesn't grade its own
answers) is half done: Groq's free tier allows one system's worth of grading per day.

Caveats: 64 questions is small, so one question moves recall@5 by about 1.6 points. Differences
under ~3 points are noise. I wrote the questions and labels myself, knowing the code. The numbers
are most useful for comparing configurations against each other.

Reproduce: `npm run eval` (q4, about 10 minutes on a laptop CPU the first time; embeddings are
cached in `eval/.cache`). `VARIANTS=q4,q8,fp16,fp32` compares model precisions and
`RERANKER=<model id>` adds a reranker. Results are written to [`eval/results/`](eval/results/).

## Limits

- Public repositories only. Very large repos are sampled: the file count and skipped files
  are shown.
- GitHub allows 60 unauthenticated API requests an hour per network, and each repo uses 2
  (cached repos too, to check for new commits).
  An optional GitHub token in Settings raises this to 5,000 an hour. A fine-grained token with
  *Public repositories (read-only)* access and no permissions is enough. If the limit is hit,
  repos already cached in your browser still open.
- Gemini's free tier has its own rate limits per key.
- Without WebGPU, embedding runs on the CPU through WebAssembly and is several times slower.

## Development

```sh
npm install
npm run dev        # http://localhost:5173
npm test           # unit tests (chunking, GitHub, retrieval, prompts, SSE parsing)
npm run build      # type-check and production build into dist/
npm run build:examples   # re-embed the example repos into public/examples/ (CPU, a few minutes)
npm run eval             # retrieval benchmark (see above)
npm run eval:answers     # answer quality: offline model vs Gemini (set GEMINI_API_KEY for Gemini + judge)
```

The Chrome extension lives in `extension/`. To try it, open `chrome://extensions`, turn on
Developer mode and *Load unpacked* that folder.

The example indexes are made with the same model, chunking and document format as the browser,
running on the CPU through onnxruntime-node. Their questions and saved answers live in
`src/examples.ts`. The answers were written from the excerpts the index retrieved, and cite them.

The site is hosted on Netlify's free plan. To deploy, build locally and upload:

```sh
npm run build
npx netlify-cli deploy --prod --dir dist
```

Pushing to `main` also runs the tests in GitHub Actions (`.github/workflows/deploy.yml`).

## License

MIT. EmbeddingGemma 2 is distributed by Google under Apache 2.0.
