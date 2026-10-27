// Minimal streaming client for the Gemini API, called directly from the browser
// with the visitor's own key. No backend sees the key or the conversation.

export const DEFAULT_MODEL = 'gemini-flash-latest';
export const SUGGESTED_MODELS = ['gemini-flash-latest', 'gemini-3.5-flash-lite', 'gemini-3.8-flash'];

export interface Turn {
  role: 'user' | 'model';
  text: string;
}

export class GeminiError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
  }
}

/** Splits an SSE byte stream into the JSON payloads of its `data:` events. */
export function parseSseEvents(buffer: string): { events: string[]; rest: string } {
  const events: string[] = [];
  const blocks = buffer.split(/\r?\n\r?\n/);
  const rest = blocks.pop() ?? '';
  for (const block of blocks) {
    const data = block
      .split(/\r?\n/)
      .filter((l) => l.startsWith('data:'))
      .map((l) => l.slice(5).trimStart())
      .join('\n');
    if (data) events.push(data);
  }
  return { events, rest };
}

export function textOfChunk(payload: unknown): string {
  const p = payload as { candidates?: { content?: { parts?: { text?: string; thought?: boolean }[] } }[] };
  return (p.candidates?.[0]?.content?.parts ?? [])
    .filter((part) => !part.thought)
    .map((part) => part.text ?? '')
    .join('');
}

function friendlyError(status: number, body: string): string {
  let message = body;
  try {
    message = (JSON.parse(body) as { error?: { message?: string } }).error?.message ?? body;
  } catch {
    // not JSON
  }
  if (status === 400 && /API key/i.test(message)) return 'That Gemini API key was rejected. Check it in Settings.';
  if (status === 403) return `Gemini refused the request: ${message}`;
  if (status === 404) return `Unknown model. Pick another in Settings. (${message})`;
  if (status === 429) return 'Gemini rate limit reached for this key. Wait a minute and try again.';
  return `Gemini error ${status}: ${message}`;
}

export async function streamChat(opts: {
  apiKey: string;
  model: string;
  system: string;
  turns: Turn[];
  onText: (delta: string) => void;
  signal?: AbortSignal;
  fetchImpl?: typeof fetch;
}): Promise<string> {
  const { apiKey, model, system, turns, onText, signal, fetchImpl = fetch } = opts;
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:streamGenerateContent?alt=sse`;
  const res = await fetchImpl(url, {
    method: 'POST',
    signal,
    headers: { 'Content-Type': 'application/json', 'x-goog-api-key': apiKey },
    body: JSON.stringify({
      systemInstruction: { parts: [{ text: system }] },
      contents: turns.map((t) => ({ role: t.role, parts: [{ text: t.text }] })),
      generationConfig: { temperature: 0.2 },
    }),
  });
  if (!res.ok || !res.body) throw new GeminiError(friendlyError(res.status, await res.text()), res.status);

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = '';
  let full = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    const parsed = parseSseEvents(buffer + value);
    buffer = parsed.rest;
    for (const data of parsed.events) {
      const delta = textOfChunk(JSON.parse(data));
      if (delta) {
        full += delta;
        onText(delta);
      }
    }
  }
  return full;
}
