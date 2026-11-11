// Offline answers: a small instruction-tuned LLM (Qwen3.5 0.8B) running in the browser,
// in its own worker so generation never blocks the page.
import { env, pipeline, TextStreamer, type TextGenerationPipeline } from '@huggingface/transformers';
import { LOCAL_LLM_ID, type LlmRequest, type LlmResponse } from './llmProtocol';

env.allowLocalModels = false;

let generator: TextGenerationPipeline | null = null;
let stopRequested = false;

const post = (msg: LlmResponse) => (self as unknown as Worker).postMessage(msg);

async function hasWebGPU(): Promise<boolean> {
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
  try {
    return Boolean(gpu && (await gpu.requestAdapter()));
  } catch {
    return false;
  }
}

async function load(): Promise<void> {
  const webgpu = await hasWebGPU();
  const files = new Map<string, { loaded: number; total: number }>();
  generator = (await pipeline('text-generation', LOCAL_LLM_ID, {
    device: webgpu ? 'webgpu' : 'wasm',
    // q4f16 needs WebGPU's fp16 support; plain q4 runs anywhere.
    dtype: webgpu ? 'q4f16' : 'q4',
    progress_callback: (p: { status: string; file?: string; loaded?: number; total?: number }) => {
      if (p.status !== 'progress' || !p.file || !p.total) return;
      files.set(p.file, { loaded: p.loaded ?? 0, total: p.total });
      let loaded = 0;
      let total = 0;
      for (const f of files.values()) {
        loaded += f.loaded;
        total += f.total;
      }
      post({ type: 'progress', loaded, total });
    },
  })) as TextGenerationPipeline;
  post({ type: 'ready', device: webgpu ? 'webgpu' : 'wasm' });
}

async function generate(id: number, messages: { role: string; content: string }[], maxNewTokens: number): Promise<void> {
  if (!generator) throw new Error('model not loaded');
  stopRequested = false;
  let tokens = 0;
  const started = performance.now();
  const streamer = new TextStreamer(generator.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (text: string) => post({ type: 'token', id, text }),
    // Text arrives a word at a time; count actual tokens for the speed shown to the user.
    token_callback_function: (ids: bigint[] | number[]) => {
      tokens += ids.length;
    },
  });
  await generator(messages, {
    max_new_tokens: maxNewTokens,
    do_sample: false,
    repetition_penalty: 1.1,
    streamer,
    // Stop between tokens when the user cancels.
    stopping_criteria: [() => [stopRequested]],
  } as Record<string, unknown>);
  post({ type: 'done', id, tokensPerSecond: tokens / ((performance.now() - started) / 1000) });
}

self.onmessage = async (event: MessageEvent<LlmRequest>) => {
  const msg = event.data;
  try {
    if (msg.type === 'load') await load();
    else if (msg.type === 'generate') await generate(msg.id, msg.messages, msg.maxNewTokens);
    else if (msg.type === 'stop') stopRequested = true;
  } catch (err) {
    post({ type: 'error', id: msg.type === 'generate' ? msg.id : undefined, message: String(err) });
  }
};
