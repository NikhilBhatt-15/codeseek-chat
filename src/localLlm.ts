// Page-side client for the offline LLM worker. The worker (and its ~470 MB model) is only
// created when the visitor turns offline answers on.
import type { LlmRequest, LlmResponse } from './llmProtocol';

export class LocalLlm {
  private worker: Worker | null = null;
  private ready: Promise<'webgpu' | 'wasm'> | null = null;
  private nextId = 1;
  private jobs = new Map<number, { onToken: (t: string) => void; resolve: (tps: number) => void; reject: (e: Error) => void }>();
  private readyResolve?: (d: 'webgpu' | 'wasm') => void;
  private readyReject?: (e: Error) => void;
  onProgress: (loaded: number, total: number) => void = () => {};
  device: 'webgpu' | 'wasm' | null = null;

  private ensureWorker(): Worker {
    if (!this.worker) {
      this.worker = new Worker(new URL('./llm.worker.ts', import.meta.url), { type: 'module' });
      this.worker.onmessage = (e: MessageEvent<LlmResponse>) => this.handle(e.data);
    }
    return this.worker;
  }

  private handle(msg: LlmResponse): void {
    if (msg.type === 'progress') this.onProgress(msg.loaded, msg.total);
    else if (msg.type === 'ready') {
      this.device = msg.device;
      this.readyResolve?.(msg.device);
    } else if (msg.type === 'token') this.jobs.get(msg.id)?.onToken(msg.text);
    else if (msg.type === 'done') {
      this.jobs.get(msg.id)?.resolve(msg.tokensPerSecond);
      this.jobs.delete(msg.id);
    } else if (msg.type === 'error') {
      const err = new Error(msg.message);
      if (msg.id === undefined) {
        this.readyReject?.(err);
        this.ready = null;
      } else {
        this.jobs.get(msg.id)?.reject(err);
        this.jobs.delete(msg.id);
      }
    }
  }

  load(): Promise<'webgpu' | 'wasm'> {
    this.ready ??= new Promise((resolve, reject) => {
      this.readyResolve = resolve;
      this.readyReject = reject;
      this.post({ type: 'load' });
    });
    return this.ready;
  }

  get loaded(): boolean {
    return this.device !== null;
  }

  private post(msg: LlmRequest): void {
    this.ensureWorker().postMessage(msg);
  }

  /** Streams an answer. Resolves with the generation speed in tokens per second. */
  async generate(
    messages: { role: 'system' | 'user' | 'assistant'; content: string }[],
    onToken: (text: string) => void,
    signal?: AbortSignal,
    maxNewTokens = 600,
  ): Promise<number> {
    await this.load();
    const id = this.nextId++;
    const done = new Promise<number>((resolve, reject) => this.jobs.set(id, { onToken, resolve, reject }));
    signal?.addEventListener('abort', () => this.post({ type: 'stop' }), { once: true });
    this.post({ type: 'generate', id, messages, maxNewTokens });
    return done;
  }
}
