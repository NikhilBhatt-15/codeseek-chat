// Runs EmbeddingGemma 2 (text encoder only, q4) off the main thread.
import {
  AutoConfig,
  AutoModel,
  AutoTokenizer,
  env,
  type PreTrainedModel,
  type PreTrainedTokenizer,
} from '@huggingface/transformers';
import type { WorkerRequest, WorkerResponse } from './protocol';
import { EMBED_DIM, MAX_TOKENS, MODEL_ID } from './protocol';

env.allowLocalModels = false; // always fetch from the Hugging Face Hub (then browser-cached)

let model: PreTrainedModel | null = null;
let tokenizer: PreTrainedTokenizer | null = null;

const post = (msg: WorkerResponse, transfer: Transferable[] = []) =>
  (self as unknown as Worker).postMessage(msg, transfer);

async function pickDevice(): Promise<'webgpu' | 'wasm'> {
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
  if (!gpu) return 'wasm';
  try {
    return (await gpu.requestAdapter()) ? 'webgpu' : 'wasm';
  } catch {
    return 'wasm';
  }
}

async function load(): Promise<void> {
  const device = await pickDevice();
  const files = new Map<string, { loaded: number; total: number }>();
  const progress_callback = (p: { status: string; file?: string; loaded?: number; total?: number }) => {
    if (p.status !== 'progress' || !p.file || !p.total) return;
    files.set(p.file, { loaded: p.loaded ?? 0, total: p.total });
    let loaded = 0;
    let total = 0;
    for (const f of files.values()) {
      loaded += f.loaded;
      total += f.total;
    }
    post({ type: 'load-progress', loaded, total });
  };

  const config = await AutoConfig.from_pretrained(MODEL_ID);
  // Text only: skip the vision and audio encoders (saves ~360 MB of downloads).
  Object.assign(config, { vision_config: null, audio_config: null });
  tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
  model = await AutoModel.from_pretrained(MODEL_ID, { config, device, dtype: 'q4', progress_callback });
  post({ type: 'ready', device });
}

async function embed(id: number, texts: string[]): Promise<void> {
  if (!model || !tokenizer) throw new Error('model not loaded');
  const inputs = tokenizer(texts, { padding: true, truncation: true, max_length: MAX_TOKENS });
  const out = (await model(inputs)) as { sentence_embedding: { slice(...a: unknown[]): any } };
  // Matryoshka: keep the first EMBED_DIM dimensions, then re-normalise.
  const vecs = out.sentence_embedding.slice(null, [0, EMBED_DIM]).normalize(2, -1);
  const data = Float32Array.from(vecs.data as ArrayLike<number>);
  post({ type: 'embedded', id, vectors: data }, [data.buffer]);
}

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const msg = event.data;
  try {
    if (msg.type === 'load') await load();
    else if (msg.type === 'embed') await embed(msg.id, msg.texts);
  } catch (err) {
    post({ type: 'error', id: msg.type === 'embed' ? msg.id : undefined, message: String(err) });
  }
};
