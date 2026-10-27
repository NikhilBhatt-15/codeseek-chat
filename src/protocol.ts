// Shared between the page and the embedding worker.

export const MODEL_ID = 'onnx-community/embeddinggemma-2-ONNX';
export const EMBED_DIM = 256; // Matryoshka truncation of the native 768
export const MAX_TOKENS = 512;
// Bump when chunking, prompts or model settings change, so cached indexes are rebuilt.
export const INDEX_VERSION = 2;

export type WorkerRequest = { type: 'load' } | { type: 'embed'; id: number; texts: string[] };

export type WorkerResponse =
  | { type: 'load-progress'; loaded: number; total: number }
  | { type: 'ready'; device: 'webgpu' | 'wasm' }
  | { type: 'embedded'; id: number; vectors: Float32Array }
  | { type: 'error'; id?: number; message: string };
