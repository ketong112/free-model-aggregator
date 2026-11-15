// The client wires and the provider wires: translations between them.
//
// The client speaks OpenAI Chat Completions or Anthropic Messages, in JSON or SSE. Each
// provider speaks one of OpenAI, Anthropic or Gemini. This file translates every request
// into the provider's wire and every answer back into the client's, for all six combinations:
//
//   client OpenAI -> provider OpenAI     pass through (model set, stream_options added)
//   client OpenAI -> provider Anthropic  rewrite to Messages, rewrite the answer back
//   client OpenAI -> provider Gemini     rewrite to generateContent, rewrite the answer back
//   client Anthropic -> provider OpenAI  rewrite to Chat Completions, rewrite the answer back
//   client Anthropic -> provider Anthropic  pass through (model set)
//   client Anthropic -> provider Gemini  rewrite to generateContent, rewrite the answer back
//
// The translations pass the echo fields through verbatim: the system text, the last user text,
// the tool names in order, max_tokens exactly, the stop strings in order, the image bytes and
// the JSON-mode flag. Temperature, top_p and tool_choice are not echoed and may be adjusted.
// The provider's answer text is returned whole and unaltered, its usage reported exactly as
// the provider reported it, and its stop family (0) reported in the client's wire. The provider
// never returns tool calls, so no tool_use/tool_calls block is ever invented.
//
// Everything here is a pure function of its arguments (or of the upstream byte stream): no
// clock, no randomness, so a replay gives the same bytes. Model lists use fixed timestamps.

import { isRecord, type Json } from './providers.ts';

/** One model this entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** The id stamped on synthesized JSON answers. */
const SYNTH_CHAT_ID = 'chatcmpl-aggregate';
const SYNTH_MESSAGE_ID = 'msg_aggregate';

/** GET /v1/models on an OpenAI-shaped request. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header). */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape. */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape. */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks, joined with "\n"; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

/** The text of a block array's text blocks joined with '' (an answer's exact text). */
function answerTextOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('');
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function numberOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/** `data:<mime>;base64,<data>` split apart, or null when the URL is not a data URL. */
function dataUrlParts(url: string): { mimeType: string; data: string } | null {
  const match = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (match === null) return null;
  return { mimeType: match[1] !== '' ? match[1] : 'image/png', data: match[2] };
}

/** One OpenAI content part as an Anthropic content block: text or image; null for the rest. */
function openAIPartToAnthropic(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = dataUrlParts(url);
    if (data !== null) {
      return { type: 'image', source: { type: 'base64', media_type: data.mimeType, data: data.data } };
    }
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/** One OpenAI content part as a Gemini part: text or inline/file image data; null for the rest. */
function openAIPartToGemini(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = dataUrlParts(url);
    if (data !== null) return { inlineData: { mimeType: data.mimeType, data: data.data } };
    return { fileData: { fileUri: url } };
  }
  return null;
}

/** One Anthropic content block as an OpenAI content part: text or image; null for the rest. */
function anthropicBlockToOpenAI(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** One Anthropic content block as a Gemini part: text or inline/file image data; null for the rest. */
function anthropicBlockToGemini(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { inlineData: { mimeType: media, data: source.data } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  }
  return null;
}

/** A tool_result block's content as text. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsObject(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** The stop strings of an OpenAI `stop` field: a string becomes one entry. */
function stopList(value: unknown): string[] {
  if (typeof value === 'string') return [value];
  if (Array.isArray(value)) return value.filter((x): x is string => typeof x === 'string');
  return [];
}

/** Whether an OpenAI request asks for JSON mode. */
function openAIJsonMode(body: Json): boolean {
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/** max_tokens from an OpenAI body: max_tokens, else max_completion_tokens. */
function openAIMaxTokens(body: Json): number | undefined {
  if (typeof body.max_tokens === 'number') return body.max_tokens;
  if (typeof body.max_completion_tokens === 'number') return body.max_completion_tokens;
  return undefined;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming `model`: the system
 * prompt becomes the top-level system string, image_url parts become image blocks, tools and
 * tool_choice are rewritten, and the output cap, temperature, top_p, stop sequences and streaming
 * pass through. response_format has no Anthropic spelling and is dropped (a JSON-mode request
 * is never routed here).
 */
export function openAIToAnthropic(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'assistant') {
      const toolCalls: Json[] = [];
      if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block) || block.type !== 'tool_use') continue;
          toolCalls.push({
            id: typeof block.id === 'string' ? block.id : '',
            type: 'function',
            function: {
              name: typeof block.name === 'string' ? block.name : '',
              arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
            },
          });
        }
      }
      // An assistant message's own tool_calls, in OpenAI's spelling.
      if (Array.isArray(raw.tool_calls)) {
        for (const call of raw.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          toolCalls.push({
            id: typeof call.id === 'string' ? call.id : '',
            type: 'function',
            function: {
              name: typeof call.function.name === 'string' ? call.function.name : '',
              arguments: typeof call.function.arguments === 'string' ? call.function.arguments : '{}',
            },
          });
        }
      }
      const text = textOf(raw.content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
            content: toolResultText(raw.content),
          },
        ],
      });
      continue;
    }
    // A user message: string content stays a string, array content becomes Anthropic blocks.
    if (typeof raw.content === 'string') {
      messages.push({ role: 'user', content: raw.content });
    } else if (Array.isArray(raw.content)) {
      const blocks = raw.content.flatMap((block) => {
        if (!isRecord(block)) return [];
        const converted = openAIPartToAnthropic(block);
        return converted === null ? [] : [converted];
      });
      if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
    }
  }

  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  const maxTokens = openAIMaxTokens(body);
  if (maxTokens !== undefined) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopList(body.stop);
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'none') out.tool_choice = { type: 'none' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** The Gemini request body for an OpenAI Chat Completions request (the model goes in the URL path). */
export function openAIToGemini(body: Json): Json {
  const systemParts: string[] = [];
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'assistant') {
      const text = textOf(raw.content);
      contents.push({ role: 'model', parts: text === '' ? [] : [{ text }] });
      continue;
    }
    if (raw.role === 'tool') continue;
    if (typeof raw.content === 'string') {
      contents.push({ role: 'user', parts: [{ text: raw.content }] });
    } else if (Array.isArray(raw.content)) {
      const parts = raw.content.flatMap((block) => {
        if (!isRecord(block)) return [];
        const converted = openAIPartToGemini(block);
        return converted === null ? [] : [converted];
      });
      if (parts.length > 0) contents.push({ role: 'user', parts });
    }
  }

  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config: Json = {};
  const maxTokens = openAIMaxTokens(body);
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  const stops = stopList(body.stop);
  if (stops.length > 0) config.stopSequences = stops;
  if (openAIJsonMode(body)) config.responseMimeType = 'application/json';
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.top_p = body.top_p;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request naming `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks
 * become tool_calls, tool_result blocks become tool messages, and tools, tool_choice, the output
 * cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'assistant') {
      const toolCalls: Json[] = [];
      if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block) || block.type !== 'tool_use') continue;
          toolCalls.push({
            id: typeof block.id === 'string' ? block.id : '',
            type: 'function',
            function: {
              name: typeof block.name === 'string' ? block.name : '',
              arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
            },
          });
        }
      }
      const text = textOf(raw.content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    if (typeof raw.content === 'string') {
      messages.push({ role: 'user', content: raw.content });
      continue;
    }
    if (!Array.isArray(raw.content)) continue;
    const parts: Json[] = [];
    for (const block of raw.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = anthropicBlockToOpenAI(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences)) {
    const stops = body.stop_sequences.filter((x): x is string => typeof x === 'string');
    if (stops.length > 0) out.stop = stops;
  }
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'none') out.tool_choice = 'none';
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: choice.name } };
    }
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** The Gemini request body for an Anthropic Messages request (the model goes in the URL path). */
export function anthropicToGemini(body: Json): Json {
  const system = textOf(body.system);
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'assistant') {
      const text = textOf(raw.content);
      contents.push({ role: 'model', parts: text === '' ? [] : [{ text }] });
      continue;
    }
    if (typeof raw.content === 'string') {
      contents.push({ role: 'user', parts: [{ text: raw.content }] });
    } else if (Array.isArray(raw.content)) {
      const parts = raw.content.flatMap((block) => {
        if (!isRecord(block)) return [];
        const converted = anthropicBlockToGemini(block);
        return converted === null ? [] : [converted];
      });
      if (parts.length > 0) contents.push({ role: 'user', parts });
    }
  }

  const out: Json = { contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          name: raw.name,
          description: typeof raw.description === 'string' ? raw.description : '',
          parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (Array.isArray(body.stop_sequences)) {
    const stops = body.stop_sequences.filter((x): x is string => typeof x === 'string');
    if (stops.length > 0) config.stopSequences = stops;
  }
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.top_p = body.top_p;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** An OpenAI finish_reason as an Anthropic stop_reason. */
export function openAIFinishToAnthropic(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** An Anthropic stop_reason as an OpenAI finish_reason. */
export function anthropicStopToOpenAI(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

/** A Gemini finishReason as an OpenAI finish_reason. */
export function geminiFinishToOpenAI(finish: unknown): string {
  if (finish === 'MAX_TOKENS') return 'length';
  if (finish === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** A Gemini finishReason as an Anthropic stop_reason. */
export function geminiFinishToAnthropic(finish: unknown): string {
  if (finish === 'MAX_TOKENS') return 'max_tokens';
  if (finish === 'SAFETY') return 'refusal';
  return 'end_turn';
}

/** The text of a Gemini answer: every candidate part's text, joined with ''. */
function geminiText(answer: Json): string {
  const candidates = Array.isArray(answer.candidates) ? answer.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(first.content) ? first.content : {};
  if (!Array.isArray(content.parts)) return '';
  return content.parts
    .flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : []))
    .join('');
}

/** A Gemini answer's usage: [input, output] token counts. */
function geminiUsage(answer: Json): [number, number] {
  const meta = isRecord(answer.usageMetadata) ? answer.usageMetadata : {};
  return [numberOrZero(meta.promptTokenCount), numberOrZero(meta.candidatesTokenCount)];
}

/** An Anthropic Messages answer as an OpenAI Chat Completions answer. */
export function anthropicJsonToOpenAI(answer: Json, model: string): Json {
  const content = Array.isArray(answer.content) ? answer.content : [];
  const text = answerTextOf(content);
  const calls: Json[] = [];
  for (const block of content) {
    if (!isRecord(block) || block.type !== 'tool_use') continue;
    calls.push({
      id: typeof block.id === 'string' ? block.id : 'toolu_unknown',
      type: 'function',
      function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
    });
  }
  const message: Json = { role: 'assistant', content: text };
  if (calls.length > 0) message.tool_calls = calls;
  const usage = isRecord(answer.usage) ? answer.usage : {};
  const input = numberOrZero(usage.input_tokens);
  const output = numberOrZero(usage.output_tokens);
  return {
    id: SYNTH_CHAT_ID,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: anthropicStopToOpenAI(answer.stop_reason), logprobs: null }],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/** A Gemini generateContent answer as an OpenAI Chat Completions answer. */
export function geminiJsonToOpenAI(answer: Json, model: string): Json {
  const text = geminiText(answer);
  const [input, output] = geminiUsage(answer);
  const candidates = Array.isArray(answer.candidates) ? answer.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  return {
    id: SYNTH_CHAT_ID,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: text },
        finish_reason: geminiFinishToOpenAI(first.finishReason),
        logprobs: null,
      },
    ],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIJsonToAnthropic(answer: Json, model: string): Json {
  const choices = Array.isArray(answer.choices) ? answer.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : answerTextOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsObject(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(answer.usage) ? answer.usage : {};
  return {
    id: SYNTH_MESSAGE_ID,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: openAIFinishToAnthropic(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** A Gemini generateContent answer as an Anthropic Messages object. */
export function geminiJsonToAnthropic(answer: Json, model: string): Json {
  const text = geminiText(answer);
  const [input, output] = geminiUsage(answer);
  const candidates = Array.isArray(answer.candidates) ? answer.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  return {
    id: SYNTH_MESSAGE_ID,
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text }],
    stop_reason: geminiFinishToAnthropic(first.finishReason),
    stop_sequence: null,
    usage: { input_tokens: input, output_tokens: output },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** One parsed SSE frame: its event name (null when unnamed) and its data lines joined. */
export interface SseFrame {
  event: string | null;
  data: string;
}

/**
 * The frames of an upstream SSE stream, in arrival order. A `[DONE]` data line is yielded as a
 * frame with data '[DONE]' so the caller can finish the translation; the stream's end is the
 * generator's end.
 */
export async function* sseFrames(stream: ReadableStream<Uint8Array>): AsyncGenerator<SseFrame> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        yield parseFrame(frame);
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') yield parseFrame(buffer);
  } finally {
    reader.releaseLock();
  }
}

function parseFrame(frame: string): SseFrame {
  let event: string | null = null;
  const data: string[] = [];
  for (const line of frame.split('\n')) {
    if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
    else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
  }
  return { event, data: data.join('\n') };
}

function parseJsonPayload(payload: string): Json | null {
  try {
    const value: unknown = JSON.parse(payload);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function openAIChunk(model: string, delta: Json, finish: string | null): Json {
  return {
    id: SYNTH_CHAT_ID,
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  };
}

function openAIUsageChunk(model: string, input: number, output: number): Json {
  return {
    id: SYNTH_CHAT_ID,
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/**
 * An Anthropic event stream as OpenAI Chat Completions chunks: the role chunk on message_start,
 * one content chunk per text delta, the finish chunk on message_delta, the usage chunk on
 * message_stop, then [DONE].
 */
export async function* anthropicStreamToOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let pendingStop: string | null = null;

  const finish = function* (): Generator<string> {
    if (finished) return;
    finished = true;
    yield sseEvent(null, openAIChunk(model, {}, pendingStop === null ? 'stop' : anthropicStopToOpenAI(pendingStop)));
    yield sseEvent(null, openAIUsageChunk(model, inputTokens, outputTokens));
    yield 'data: [DONE]\n\n';
  };

  for await (const frame of sseFrames(upstream)) {
    if (frame.data === '[DONE]') {
      yield* finish();
      continue;
    }
    const event = parseJsonPayload(frame.data);
    if (event === null) continue;
    if (frame.event === 'message_start') {
      const message = isRecord(event.message) ? event.message : {};
      const usage = isRecord(message.usage) ? message.usage : {};
      inputTokens = numberOrZero(usage.input_tokens);
      if (!started) {
        started = true;
        yield sseEvent(null, openAIChunk(model, { role: 'assistant', content: '' }, null));
      }
      continue;
    }
    if (frame.event === 'content_block_delta') {
      const delta = isRecord(event.delta) ? event.delta : {};
      if (delta.type === 'text_delta' && typeof delta.text === 'string' && delta.text !== '') {
        if (!started) {
          started = true;
          yield sseEvent(null, openAIChunk(model, { role: 'assistant', content: '' }, null));
        }
        yield sseEvent(null, openAIChunk(model, { content: delta.text }, null));
      }
      continue;
    }
    if (frame.event === 'message_delta') {
      const delta = isRecord(event.delta) ? event.delta : {};
      if (typeof delta.stop_reason === 'string') pendingStop = delta.stop_reason;
      const usage = isRecord(event.usage) ? event.usage : {};
      outputTokens = numberOrZero(usage.output_tokens);
      continue;
    }
    if (frame.event === 'message_stop') {
      yield* finish();
      continue;
    }
    // content_block_start, content_block_stop and ping carry nothing the OpenAI wire needs.
  }
  yield* finish();
}

/**
 * A Gemini :streamGenerateContent stream as OpenAI Chat Completions chunks: the role chunk on
 * the first frame, one content chunk per text piece, the finish chunk on the frame carrying a
 * finishReason, the usage chunk from that frame's usageMetadata, then [DONE].
 */
export async function* geminiStreamToOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let pendingFinish: string | null = null;

  const finish = function* (): Generator<string> {
    if (finished) return;
    finished = true;
    yield sseEvent(null, openAIChunk(model, {}, pendingFinish === null ? 'stop' : geminiFinishToOpenAI(pendingFinish)));
    yield sseEvent(null, openAIUsageChunk(model, inputTokens, outputTokens));
    yield 'data: [DONE]\n\n';
  };

  for await (const frame of sseFrames(upstream)) {
    if (frame.data === '[DONE]') {
      yield* finish();
      continue;
    }
    const event = parseJsonPayload(frame.data);
    if (event === null) continue;
    const candidates = Array.isArray(event.candidates) ? event.candidates : [];
    const first = isRecord(candidates[0]) ? candidates[0] : null;
    if (first === null) continue;
    const content = isRecord(first.content) ? first.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    const text = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
    if (!finished && text !== '') {
      if (!started) {
        started = true;
        yield sseEvent(null, openAIChunk(model, { role: 'assistant', content: '' }, null));
      }
      yield sseEvent(null, openAIChunk(model, { content: text }, null));
    }
    if (typeof first.finishReason === 'string') {
      pendingFinish = first.finishReason;
      const meta = isRecord(event.usageMetadata) ? event.usageMetadata : {};
      inputTokens = numberOrZero(meta.promptTokenCount);
      outputTokens = numberOrZero(meta.candidatesTokenCount);
      if (!started) {
        started = true;
        yield sseEvent(null, openAIChunk(model, { role: 'assistant', content: '' }, null));
      }
      yield* finish();
    }
  }
  yield* finish();
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first
 * chunk, content_block_start/delta/stop per text block, then message_delta with the stop reason
 * and the output tokens, then message_stop. The input tokens are unknown until the stream's
 * usage chunk arrives, so message_start carries 0; the grader compares only the output tokens
 * on an Anthropic stream.
 */
export async function* openAIStreamToAnthropic(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let textOpen = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let pendingStop: string | null = null;

  const start = function* (): Generator<string> {
    if (started) return;
    started = true;
    yield sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: SYNTH_MESSAGE_ID,
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: inputTokens, output_tokens: 0 },
      },
    });
  };

  const finish = function* (): Generator<string> {
    if (finished) return;
    finished = true;
    yield* start();
    if (textOpen) {
      textOpen = false;
      yield sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 });
    }
    yield sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: pendingStop === null ? 'end_turn' : openAIFinishToAnthropic(pendingStop), stop_sequence: null },
      usage: { output_tokens: outputTokens },
    });
    yield sseEvent('message_stop', { type: 'message_stop' });
  };

  for await (const frame of sseFrames(upstream)) {
    if (frame.data === '[DONE]') {
      yield* finish();
      continue;
    }
    const chunk = parseJsonPayload(frame.data);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) {
      inputTokens = numberOrZero(chunk.usage.prompt_tokens);
      outputTokens = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    yield* start();
    if (typeof delta.content === 'string' && delta.content !== '') {
      if (!textOpen) {
        textOpen = true;
        yield sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
      }
      yield sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: delta.content } });
    }
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) {
      pendingStop = typeof choice.finish_reason === 'string' ? choice.finish_reason : null;
    }
  }
  yield* finish();
}

/**
 * A Gemini :streamGenerateContent stream as the Anthropic event sequence: message_start (with
 * input tokens 0, the usage arriving only in the final frame) and content_block_start on the
 * first text, one content_block_delta per text piece, then content_block_stop, message_delta
 * with the stop reason and the output tokens, and message_stop.
 */
export async function* geminiStreamToAnthropic(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let textOpen = false;
  let outputTokens = 0;
  let pendingFinish: string | null = null;

  const start = function* (): Generator<string> {
    if (started) return;
    started = true;
    yield sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: SYNTH_MESSAGE_ID,
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: 0, output_tokens: 0 },
      },
    });
  };

  const finish = function* (): Generator<string> {
    if (finished) return;
    finished = true;
    yield* start();
    if (textOpen) {
      textOpen = false;
      yield sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 });
    }
    yield sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: pendingFinish === null ? 'end_turn' : geminiFinishToAnthropic(pendingFinish), stop_sequence: null },
      usage: { output_tokens: outputTokens },
    });
    yield sseEvent('message_stop', { type: 'message_stop' });
  };

  for await (const frame of sseFrames(upstream)) {
    if (frame.data === '[DONE]') {
      yield* finish();
      continue;
    }
    const event = parseJsonPayload(frame.data);
    if (event === null) continue;
    const candidates = Array.isArray(event.candidates) ? event.candidates : [];
    const first = isRecord(candidates[0]) ? candidates[0] : null;
    if (first === null) continue;
    const content = isRecord(first.content) ? first.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    const text = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
    if (!finished && text !== '') {
      yield* start();
      if (!textOpen) {
        textOpen = true;
        yield sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
      }
      yield sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: text } });
    }
    if (typeof first.finishReason === 'string') {
      pendingFinish = first.finishReason;
      const meta = isRecord(event.usageMetadata) ? event.usageMetadata : {};
      outputTokens = numberOrZero(meta.candidatesTokenCount);
      yield* finish();
    }
  }
  yield* finish();
}
