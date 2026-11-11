// Shared between the page and the offline LLM worker.

export const LOCAL_LLM_ID = 'onnx-community/Qwen3.5-0.8B-Text-ONNX';
export const LOCAL_LLM_LABEL = 'Qwen3.5 0.8B';
export const LOCAL_LLM_MB = 469; // q4f16 download on WebGPU (551 MB as q4 on the CPU)

export type LlmRequest =
  | { type: 'load' }
  | { type: 'generate'; id: number; messages: { role: string; content: string }[]; maxNewTokens: number }
  | { type: 'stop' };

export type LlmResponse =
  | { type: 'progress'; loaded: number; total: number }
  | { type: 'ready'; device: 'webgpu' | 'wasm' }
  | { type: 'token'; id: number; text: string }
  | { type: 'done'; id: number; tokensPerSecond: number }
  | { type: 'error'; id?: number; message: string };
