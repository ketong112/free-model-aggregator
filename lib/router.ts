// Routing and failover: which free model serves a request, and what happens when it refuses.
//
// Candidates are every probed model that is free and meets the request's requirements, ranked
// by the quality table's rating (highest first). Ties break toward the provider speaking the
// client's own wire (a passthrough is one less translation), then by provider id, then model
// id, so the order is fully deterministic. An OpenAI JSON-mode request never routes to an
// Anthropic-wire provider: that wire cannot carry JSON mode and the echo would fail.
//
// Failover: a 402, 429 or 5xx is a refusal, never an answer. The request moves to the next
// candidate on a different provider (the provider that just refused is never retried
// immediately), and a provider is blacklisted for the rest of the trial after its second
// refusal. A 400 means the request as built is wrong, so the loop stops instead of retrying.
// Exactly one successful provider call is made per request: a success ends the loop at once,
// and a second success is never attempted. Refused calls still consume the provider's budget,
// so the blacklist also keeps the entry from burning calls on a provider that is spent.
//
// Per-provider state (calls made, refusals) lives in memory only and is a pure function of the
// provider answers seen so far, so a replay makes the same choices.

import {
  authHeaders,
  capabilitiesOf,
  joinBase,
  meetsRequirements,
  modelIsFree,
  ratingRowOf,
  type Json,
  type ListedModel,
  type Probe,
  type ProviderEntry,
  type RatingsTable,
  type Requirement,
} from './providers.ts';
import { anthropicToGemini, anthropicToOpenAI, openAIToAnthropic, openAIToGemini } from './wire.ts';

/** One (provider, model) pair that may serve a request. */
export interface Candidate {
  provider: ProviderEntry;
  model: ListedModel;
  rating: number;
}

/** The per-provider trial state: calls made and refusals seen. */
export class ProviderState {
  private readonly calls = new Map<string, number>();
  private readonly refusals = new Map<string, number>();

  noteCall(providerId: string): void {
    this.calls.set(providerId, (this.calls.get(providerId) ?? 0) + 1);
  }

  noteRefusal(providerId: string): void {
    this.refusals.set(providerId, (this.refusals.get(providerId) ?? 0) + 1);
  }

  callsMade(providerId: string): number {
    return this.calls.get(providerId) ?? 0;
  }

  refusalsSeen(providerId: string): number {
    return this.refusals.get(providerId) ?? 0;
  }

  /** Blacklisted after two refusals: the provider is spent, out, or failing on schedule. */
  blacklisted(providerId: string): boolean {
    return (this.refusals.get(providerId) ?? 0) >= 2;
  }
}

/**
 * The ranked candidates for one request. `clientWire` is the wire the client used; `jsonMode`
 * is true when the client asked for JSON mode on the OpenAI wire (such a request cannot go to
 * an Anthropic-wire provider).
 */
export function rankCandidates(
  probes: readonly Probe[],
  ratings: RatingsTable | null,
  requires: Requirement,
  clientWire: 'openai' | 'anthropic',
  jsonMode: boolean,
  state: ProviderState,
): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (state.blacklisted(probe.provider.id)) continue;
    // An OpenAI JSON-mode request routed to an Anthropic-wire provider always fails the
    // echo_json check: that wire has no JSON-mode field.
    if (clientWire === 'openai' && jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingRowOf(ratings, probe.provider.id, model.id);
      if (!modelIsFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ provider: probe.provider, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => {
    if (a.rating !== b.rating) return b.rating - a.rating;
    const aSameWire = a.provider.wire === clientWire ? 0 : 1;
    const bSameWire = b.provider.wire === clientWire ? 0 : 1;
    if (aSameWire !== bSameWire) return aSameWire - bSameWire;
    if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
    if (a.model.id !== b.model.id) return a.model.id < b.model.id ? -1 : 1;
    return 0;
  });
  return out;
}

/** How long one provider call may take before it is abandoned, in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 10000;

/** The provider request for one candidate: URL, headers and body, in the provider's own wire. */
function providerRequest(
  envUrl: string,
  candidate: Candidate,
  clientWire: 'openai' | 'anthropic',
  body: Json,
  stream: boolean,
): { url: string; headers: Record<string, string>; payload: Json } {
  const provider = candidate.provider;
  const base = joinBase(envUrl, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (stream) headers.accept = 'text/event-stream';
  if (provider.wire === 'openai') {
    const payload: Json =
      clientWire === 'openai'
        ? { ...body, model: candidate.model.id, ...(stream ? { stream_options: { include_usage: true } } : {}) }
        : anthropicToOpenAI(body, candidate.model.id);
    return { url: `${base}/v1/chat/completions`, headers, payload };
  }
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    const payload: Json =
      clientWire === 'openai' ? openAIToAnthropic(body, candidate.model.id) : { ...body, model: candidate.model.id };
    return { url: `${base}/v1/messages`, headers, payload };
  }
  // Gemini: the model goes in the URL path, and the body is the generateContent document.
  const action = stream ? 'streamGenerateContent?alt=sse' : 'generateContent';
  const payload: Json = clientWire === 'openai' ? openAIToGemini(body) : anthropicToGemini(body);
  return { url: `${base}/v1beta/models/${candidate.model.id}:${action}`, headers, payload };
}

/** One provider call. A network error throws; an HTTP answer is returned as it stands. */
async function callProvider(
  envUrl: string,
  candidate: Candidate,
  clientWire: 'openai' | 'anthropic',
  body: Json,
  stream: boolean,
): Promise<Response> {
  const { url, headers, payload } = providerRequest(envUrl, candidate, clientWire, body, stream);
  return fetch(url, {
    method: 'POST',
    headers,
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
  });
}

export type ServeOutcome = 'served' | 'unserved';

/**
 * Serve one client request: walk the ranked candidates until one provider answers 2xx, then
 * hand that answer to `deliver` (which translates it into the client's wire and writes it).
 * Returns 'served' when a provider answered and the answer was delivered, 'unserved' when no
 * candidate could serve the request. `deliver` is only ever called once, on the one success.
 */
export async function serveRequest(
  envUrl: string,
  state: ProviderState,
  candidates: readonly Candidate[],
  clientWire: 'openai' | 'anthropic',
  body: Json,
  stream: boolean,
  deliver: (candidate: Candidate, upstream: Response) => Promise<void>,
  log: (message: string) => void,
): Promise<ServeOutcome> {
  let lastFailedProvider: string | null = null;
  for (const candidate of candidates) {
    // Never immediately retry the provider that just refused.
    if (candidate.provider.id === lastFailedProvider) continue;
    let upstream: Response;
    try {
      upstream = await callProvider(envUrl, candidate, clientWire, body, stream);
    } catch (err) {
      // The provider could not be reached at all: count it as a refusal and move on.
      state.noteCall(candidate.provider.id);
      state.noteRefusal(candidate.provider.id);
      lastFailedProvider = candidate.provider.id;
      log(`${candidate.provider.id}/${candidate.model.id} unreachable: ${(err as Error).message}`);
      continue;
    }
    state.noteCall(candidate.provider.id);
    if (upstream.ok) {
      await deliver(candidate, upstream);
      return 'served';
    }
    // Drain the error body so the connection can be reused, then decide.
    try {
      await upstream.text();
    } catch {
      // The error body could not be read: the status is what matters.
    }
    if (upstream.status === 400) {
      // A 400 is the entry's own fault (an unknown or paid model, a malformed body): retrying
      // it on another provider cannot fix it.
      log(`${candidate.provider.id}/${candidate.model.id} answered HTTP 400: the request is wrong, not retrying`);
      return 'unserved';
    }
    // 402, 429 and 5xx are refusals: fail over to a different provider.
    state.noteRefusal(candidate.provider.id);
    lastFailedProvider = candidate.provider.id;
    log(
      `${candidate.provider.id}/${candidate.model.id} refused HTTP ${upstream.status} ` +
        `(calls ${state.callsMade(candidate.provider.id)}, refusals ${state.refusalsSeen(candidate.provider.id)})`,
    );
  }
  return 'unserved';
}
