// Shared between the page and the embedding worker.

export const MODEL_ID = 'onnx-community/embeddinggemma-2-ONNX';
// Full size: on the retrieval benchmark (eval/) 768 dims beat the 256-dim Matryoshka prefix on
// every metric (MRR 0.81 vs 0.77), and exact search over a browser-sized index stays ~1 ms.
export const EMBED_DIM = 768;
export const MAX_TOKENS = 512;
// Bump when chunking, prompts or model settings change, so cached indexes are rebuilt.
export const INDEX_VERSION = 4;

export type WorkerRequest = { type: 'load' } | { type: 'embed'; id: number; texts: string[] };

export type WorkerResponse =
  | { type: 'load-progress'; loaded: number; total: number }
  | { type: 'ready'; device: 'webgpu' | 'wasm' }
  | { type: 'embedded'; id: number; vectors: Float32Array }
  | { type: 'error'; id?: number; message: string };
