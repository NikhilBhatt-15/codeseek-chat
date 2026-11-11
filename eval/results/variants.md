64 questions over 4 repos. Recall is over chunks (after overlap de-duplication).

| Retriever | Model | recall@1 | recall@5 | recall@10 | hit@5 | MRR@10 |
|---|---|---|---|---|---|---|
| BM25 only | - | 0.38 | 0.79 | 0.89 | 0.81 | 0.58 |
| Embeddings 768d | q4 | 0.68 | 0.94 | 1.00 | 0.94 | 0.81 |
| Hybrid 768d (RRF) | q4 | 0.63 | 0.92 | 0.95 | 0.94 | 0.77 |
| Routed 768d (hybrid if code-like) | q4 | 0.66 | 0.94 | 1.00 | 0.94 | 0.80 |
| Embeddings 256d | q4 | 0.64 | 0.91 | 1.00 | 0.92 | 0.77 |
| Hybrid 256d (RRF) | q4 | 0.59 | 0.90 | 0.97 | 0.92 | 0.74 |
| Routed 256d (hybrid if code-like) | q4 | 0.61 | 0.91 | 1.00 | 0.92 | 0.76 |
| Embeddings 128d | q4 | 0.58 | 0.88 | 0.96 | 0.89 | 0.72 |
| Hybrid 128d (RRF) | q4 | 0.59 | 0.91 | 0.95 | 0.94 | 0.74 |
| Routed 128d (hybrid if code-like) | q4 | 0.56 | 0.90 | 0.96 | 0.91 | 0.72 |
| Embeddings 768d | q8 | 0.67 | 0.92 | 0.98 | 0.92 | 0.80 |
| Hybrid 768d (RRF) | q8 | 0.63 | 0.88 | 0.98 | 0.91 | 0.79 |
| Routed 768d (hybrid if code-like) | q8 | 0.66 | 0.95 | 1.00 | 0.95 | 0.80 |
| Embeddings 256d | q8 | 0.59 | 0.95 | 0.98 | 0.95 | 0.76 |
| Hybrid 256d (RRF) | q8 | 0.54 | 0.91 | 0.98 | 0.94 | 0.74 |
| Routed 256d (hybrid if code-like) | q8 | 0.56 | 0.94 | 0.98 | 0.94 | 0.74 |
| Embeddings 128d | q8 | 0.49 | 0.90 | 0.93 | 0.91 | 0.68 |
| Hybrid 128d (RRF) | q8 | 0.56 | 0.90 | 0.96 | 0.92 | 0.74 |
| Routed 128d (hybrid if code-like) | q8 | 0.48 | 0.90 | 0.96 | 0.91 | 0.68 |
| Embeddings 768d | fp16 | 0.67 | 0.92 | 0.98 | 0.92 | 0.80 |
| Hybrid 768d (RRF) | fp16 | 0.63 | 0.88 | 0.98 | 0.91 | 0.79 |
| Routed 768d (hybrid if code-like) | fp16 | 0.66 | 0.95 | 1.00 | 0.95 | 0.80 |
| Embeddings 256d | fp16 | 0.58 | 0.94 | 0.98 | 0.94 | 0.75 |
| Hybrid 256d (RRF) | fp16 | 0.57 | 0.91 | 0.98 | 0.94 | 0.75 |
| Routed 256d (hybrid if code-like) | fp16 | 0.55 | 0.94 | 0.98 | 0.94 | 0.74 |
| Embeddings 128d | fp16 | 0.49 | 0.90 | 0.96 | 0.91 | 0.68 |
| Hybrid 128d (RRF) | fp16 | 0.56 | 0.90 | 0.96 | 0.92 | 0.74 |
| Routed 128d (hybrid if code-like) | fp16 | 0.48 | 0.90 | 0.96 | 0.91 | 0.68 |
| Embeddings 768d | fp32 | 0.67 | 0.92 | 0.98 | 0.92 | 0.80 |
| Hybrid 768d (RRF) | fp32 | 0.63 | 0.88 | 0.98 | 0.91 | 0.79 |
| Routed 768d (hybrid if code-like) | fp32 | 0.66 | 0.95 | 1.00 | 0.95 | 0.80 |
| Embeddings 256d | fp32 | 0.58 | 0.94 | 0.98 | 0.94 | 0.75 |
| Hybrid 256d (RRF) | fp32 | 0.57 | 0.91 | 0.98 | 0.94 | 0.75 |
| Routed 256d (hybrid if code-like) | fp32 | 0.55 | 0.94 | 0.98 | 0.94 | 0.74 |
| Embeddings 128d | fp32 | 0.49 | 0.90 | 0.96 | 0.91 | 0.68 |
| Hybrid 128d (RRF) | fp32 | 0.56 | 0.90 | 0.96 | 0.92 | 0.74 |
| Routed 128d (hybrid if code-like) | fp32 | 0.48 | 0.90 | 0.96 | 0.91 | 0.68 |

recall@5 by question kind:

| Retriever | Model | concept (38) | cross-file (6) | identifier (14) | error-message (6) |
|---|---|---|---|---|---|
| BM25 only | - | 0.79 | 0.58 | 0.79 | 1.00 |
| Embeddings 768d | q4 | 0.95 | 1.00 | 0.93 | 0.83 |
| Hybrid 768d (RRF) | q4 | 0.95 | 0.83 | 0.93 | 0.83 |
| Routed 768d (hybrid if code-like) | q4 | 0.95 | 1.00 | 0.93 | 0.83 |
| Embeddings 256d | q4 | 0.92 | 0.92 | 0.93 | 0.83 |
| Hybrid 256d (RRF) | q4 | 0.92 | 0.75 | 0.93 | 0.83 |
| Routed 256d (hybrid if code-like) | q4 | 0.92 | 0.92 | 0.93 | 0.83 |
| Embeddings 128d | q4 | 0.87 | 0.92 | 0.86 | 1.00 |
| Hybrid 128d (RRF) | q4 | 0.92 | 0.75 | 0.93 | 1.00 |
| Routed 128d (hybrid if code-like) | q4 | 0.87 | 0.92 | 0.93 | 1.00 |
| Embeddings 768d | q8 | 0.97 | 1.00 | 0.86 | 0.67 |
| Hybrid 768d (RRF) | q8 | 0.89 | 0.75 | 0.93 | 0.83 |
| Routed 768d (hybrid if code-like) | q8 | 0.97 | 1.00 | 0.93 | 0.83 |
| Embeddings 256d | q8 | 0.95 | 1.00 | 0.93 | 1.00 |
| Hybrid 256d (RRF) | q8 | 0.95 | 0.75 | 0.93 | 0.83 |
| Routed 256d (hybrid if code-like) | q8 | 0.95 | 1.00 | 0.93 | 0.83 |
| Embeddings 128d | q8 | 0.89 | 0.92 | 0.86 | 1.00 |
| Hybrid 128d (RRF) | q8 | 0.92 | 0.75 | 0.93 | 0.83 |
| Routed 128d (hybrid if code-like) | q8 | 0.89 | 0.92 | 0.93 | 0.83 |
| Embeddings 768d | fp16 | 0.97 | 1.00 | 0.86 | 0.67 |
| Hybrid 768d (RRF) | fp16 | 0.89 | 0.75 | 0.93 | 0.83 |
| Routed 768d (hybrid if code-like) | fp16 | 0.97 | 1.00 | 0.93 | 0.83 |
| Embeddings 256d | fp16 | 0.95 | 1.00 | 0.93 | 0.83 |
| Hybrid 256d (RRF) | fp16 | 0.95 | 0.75 | 0.93 | 0.83 |
| Routed 256d (hybrid if code-like) | fp16 | 0.95 | 1.00 | 0.93 | 0.83 |
| Embeddings 128d | fp16 | 0.89 | 0.92 | 0.86 | 1.00 |
| Hybrid 128d (RRF) | fp16 | 0.92 | 0.75 | 0.93 | 0.83 |
| Routed 128d (hybrid if code-like) | fp16 | 0.89 | 0.92 | 0.93 | 0.83 |
| Embeddings 768d | fp32 | 0.97 | 1.00 | 0.86 | 0.67 |
| Hybrid 768d (RRF) | fp32 | 0.89 | 0.75 | 0.93 | 0.83 |
| Routed 768d (hybrid if code-like) | fp32 | 0.97 | 1.00 | 0.93 | 0.83 |
| Embeddings 256d | fp32 | 0.95 | 1.00 | 0.93 | 0.83 |
| Hybrid 256d (RRF) | fp32 | 0.95 | 0.75 | 0.93 | 0.83 |
| Routed 256d (hybrid if code-like) | fp32 | 0.95 | 1.00 | 0.93 | 0.83 |
| Embeddings 128d | fp32 | 0.89 | 0.92 | 0.86 | 1.00 |
| Hybrid 128d (RRF) | fp32 | 0.92 | 0.75 | 0.93 | 0.83 |
| Routed 128d (hybrid if code-like) | fp32 | 0.89 | 0.92 | 0.93 | 0.83 |

recall@5 by repo:

| Retriever | Model | sindresorhus/ky | go-chi/chi | pallets/itsdangerous | NikhilBhatt-15/codeseek |
|---|---|---|---|---|---|
| BM25 only | - | 0.65 | 0.76 | 0.82 | 0.96 |
| Embeddings 768d | q4 | 0.82 | 1.00 | 0.93 | 1.00 |
| Hybrid 768d (RRF) | q4 | 0.88 | 0.95 | 0.89 | 0.96 |
| Routed 768d (hybrid if code-like) | q4 | 0.88 | 0.95 | 0.93 | 1.00 |
| Embeddings 256d | q4 | 0.88 | 1.00 | 0.82 | 0.93 |
| Hybrid 256d (RRF) | q4 | 0.94 | 0.87 | 0.82 | 0.96 |
| Routed 256d (hybrid if code-like) | q4 | 1.00 | 0.95 | 0.75 | 0.93 |
| Embeddings 128d | q4 | 0.88 | 0.89 | 0.82 | 0.93 |
| Hybrid 128d (RRF) | q4 | 0.94 | 0.87 | 0.89 | 0.96 |
| Routed 128d (hybrid if code-like) | q4 | 0.94 | 0.89 | 0.82 | 0.93 |
| Embeddings 768d | q8 | 0.88 | 0.95 | 0.86 | 1.00 |
| Hybrid 768d (RRF) | q8 | 0.94 | 0.87 | 0.75 | 0.96 |
| Routed 768d (hybrid if code-like) | q8 | 1.00 | 0.95 | 0.86 | 1.00 |
| Embeddings 256d | q8 | 0.88 | 1.00 | 0.93 | 1.00 |
| Hybrid 256d (RRF) | q8 | 0.94 | 0.92 | 0.82 | 0.96 |
| Routed 256d (hybrid if code-like) | q8 | 0.94 | 0.95 | 0.86 | 1.00 |
| Embeddings 128d | q8 | 0.88 | 0.89 | 0.89 | 0.93 |
| Hybrid 128d (RRF) | q8 | 0.94 | 0.87 | 0.82 | 0.96 |
| Routed 128d (hybrid if code-like) | q8 | 0.94 | 0.89 | 0.82 | 0.93 |
| Embeddings 768d | fp16 | 0.88 | 0.95 | 0.86 | 1.00 |
| Hybrid 768d (RRF) | fp16 | 0.94 | 0.87 | 0.75 | 0.96 |
| Routed 768d (hybrid if code-like) | fp16 | 1.00 | 0.95 | 0.86 | 1.00 |
| Embeddings 256d | fp16 | 0.88 | 1.00 | 0.86 | 1.00 |
| Hybrid 256d (RRF) | fp16 | 0.94 | 0.92 | 0.82 | 0.96 |
| Routed 256d (hybrid if code-like) | fp16 | 0.94 | 0.95 | 0.86 | 1.00 |
| Embeddings 128d | fp16 | 0.88 | 0.89 | 0.89 | 0.93 |
| Hybrid 128d (RRF) | fp16 | 0.94 | 0.87 | 0.82 | 0.96 |
| Routed 128d (hybrid if code-like) | fp16 | 0.94 | 0.89 | 0.82 | 0.93 |
| Embeddings 768d | fp32 | 0.88 | 0.95 | 0.86 | 1.00 |
| Hybrid 768d (RRF) | fp32 | 0.94 | 0.87 | 0.75 | 0.96 |
| Routed 768d (hybrid if code-like) | fp32 | 1.00 | 0.95 | 0.86 | 1.00 |
| Embeddings 256d | fp32 | 0.88 | 1.00 | 0.86 | 1.00 |
| Hybrid 256d (RRF) | fp32 | 0.94 | 0.92 | 0.82 | 0.96 |
| Routed 256d (hybrid if code-like) | fp32 | 0.94 | 0.95 | 0.86 | 1.00 |
| Embeddings 128d | fp32 | 0.88 | 0.89 | 0.89 | 0.93 |
| Hybrid 128d (RRF) | fp32 | 0.94 | 0.87 | 0.82 | 0.96 |
| Routed 128d (hybrid if code-like) | fp32 | 0.94 | 0.89 | 0.82 | 0.93 |

Embedding speed on this machine's CPU (chunks/s):

| Model | chunks/s |
|---|---|
| q4 | 1.46 |
| q8 | 1.21 |
| fp16 | 1.21 |
| fp32 | 1.92 |
