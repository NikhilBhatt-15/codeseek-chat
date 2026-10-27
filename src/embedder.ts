// Page-side client for the embedding worker.
import { EMBED_DIM, type WorkerRequest, type WorkerResponse } from './protocol';

export const formatDocument = (path: string, text: string) => `title: ${path} | text: ${text}`;
export const formatQuery = (query: string) => `task: code retrieval | query: ${query}`;

export class Embedder {
  private worker: Worker;
  private ready: Promise<'webgpu' | 'wasm'> | null = null;
  private pending = new Map<number, { resolve: (v: Float32Array) => void; reject: (e: Error) => void }>();
  private nextId = 1;
  onLoadProgress: (loaded: number, total: number) => void = () => {};

  constructor() {
    this.worker = new Worker(new URL('./embed.worker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = (e: MessageEvent<WorkerResponse>) => this.handle(e.data);
  }

  private readyResolve?: (d: 'webgpu' | 'wasm') => void;
  private readyReject?: (e: Error) => void;

  private handle(msg: WorkerResponse): void {
    if (msg.type === 'load-progress') this.onLoadProgress(msg.loaded, msg.total);
    else if (msg.type === 'ready') this.readyResolve?.(msg.device);
    else if (msg.type === 'embedded') {
      this.pending.get(msg.id)?.resolve(msg.vectors);
      this.pending.delete(msg.id);
    } else if (msg.type === 'error') {
      const err = new Error(msg.message);
      if (msg.id === undefined) {
        this.readyReject?.(err);
        this.ready = null; // allow a retry
      } else {
        this.pending.get(msg.id)?.reject(err);
        this.pending.delete(msg.id);
      }
    }
  }

  /** Loads the model once (downloads ~175 MB the first time, then served from the browser cache). */
  load(): Promise<'webgpu' | 'wasm'> {
    this.ready ??= new Promise((resolve, reject) => {
      this.readyResolve = resolve;
      this.readyReject = reject;
      this.post({ type: 'load' });
    });
    return this.ready;
  }

  private post(msg: WorkerRequest): void {
    this.worker.postMessage(msg);
  }

  /** Embeds texts as one batch. Returns rows of EMBED_DIM floats, L2-normalised. */
  async embedBatch(texts: string[]): Promise<Float32Array> {
    await this.load();
    const id = this.nextId++;
    const result = new Promise<Float32Array>((resolve, reject) => this.pending.set(id, { resolve, reject }));
    this.post({ type: 'embed', id, texts });
    const vectors = await result;
    if (vectors.length !== texts.length * EMBED_DIM) throw new Error('unexpected embedding size');
    return vectors;
  }

  async embedQuery(query: string): Promise<Float32Array> {
    return this.embedBatch([formatQuery(query)]);
  }
}
