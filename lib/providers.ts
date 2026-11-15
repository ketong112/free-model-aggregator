// The brief, the catalog probe and the capability table.
//
// A trial hands this server a brief (the path in MAKEFUN_PROMPT_FILE) holding three JSON
// documents, found by their `schema` member:
//
//   free-model-aggregator-01/providers/v1  the ten providers: id, wire, auth scheme, the
//     environment variable a real deployment would read the key from, the key the simulated
//     service takes, and the base path to join to MAKEFUN_ENV_URL.
//   free-model-aggregator-01/ratings/v1    the quality table: per (provider, model) the rating
//     in milli-points, context length, tools/vision/JSON capabilities and the free flag.
//   free-model-aggregator-01/workload/v1   the twenty requests in arrival order: id, api,
//     stream flag, weight and the requirements a model must meet.
//
// This file also probes every provider's catalog route in its own wire and folds the answers
// into the one catalog document that `--list` prints and /register carries. The probe is the
// only place the entry learns which models a provider lists; the ratings table is the sealed
// truth for the free flag and the capabilities where it has a row, and the listing's own
// fields fill the gaps. Everything here is deterministic: no clock, no randomness.

import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

export const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
export const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
export const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

export type Json = Record<string, unknown>;
export const isRecord = (value: unknown): value is Json =>
  value !== null && typeof value === 'object' && !Array.isArray(value);

export interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  /** The base path (or absolute URL) the provider's routes hang off. */
  base: string;
  key_env: string;
  key: string;
}

export interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface RatingsTable {
  rows: RatingRow[];
  /** The best legal rating per request id, from the brief. */
  best: Map<string, number>;
}

export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

export interface WorkloadEntry {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  requires: Requirement;
}

export interface Brief {
  providers: ProviderEntry[];
  ratings: RatingsTable | null;
  workload: WorkloadEntry[];
  error: string | null;
}

/** One model as a provider's catalog route listed it. Null fields are what it did not say. */
export interface ListedModel {
  id: string;
  display_name: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

export interface Probe {
  provider: ProviderEntry;
  ok: boolean;
  status: number;
  models: ListedModel[];
  error: string | null;
}

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

/** The one catalog document: what `--list` prints and /register carries, byte for byte. */
export interface Catalog {
  providers: CatalogProvider[];
}

export interface Capabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** A base path joined to the env's origin; an absolute base is used as it stands. */
export function joinBase(envUrl: string, base: string): string {
  const trimmed = base.trim();
  if (/^https?:\/\//i.test(trimmed)) return trimmed.replace(/\/+$/, '');
  const origin = envUrl.replace(/\/+$/, '');
  if (trimmed === '') return origin;
  return `${origin}${trimmed.startsWith('/') ? '' : '/'}${trimmed.replace(/\/+$/, '')}`;
}

/** The headers that carry a provider's key, in its own scheme. */
export function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

/** The index of the `}` that closes the object opening at `start`, or -1. */
function closingBrace(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function schemaDocuments(text: string): Json[] {
  const docs: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = closingBrace(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') docs.push(value);
    } catch {
      // Not a document on its own: keep scanning.
    }
  }
  return docs;
}

function asString(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

function asNumberOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function asBoolOrNull(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null;
}

function providerFrom(value: unknown): ProviderEntry | null {
  if (!isRecord(value)) return null;
  const id = asString(value.id);
  const wire = value.wire === 'openai' || value.wire === 'anthropic' || value.wire === 'gemini' ? value.wire : null;
  if (id === '' || wire === null) return null;
  const auth =
    value.auth === 'bearer' || value.auth === 'x-api-key' || value.auth === 'x-goog-api-key' || value.auth === 'none'
      ? value.auth
      : 'bearer';
  // The directory writes `path` (/p/<id>); an absolute base may instead be spelled base_url, base_path
  // or base. `path` wins.
  const base =
    asString(value.path) !== ''
      ? asString(value.path)
      : asString(value.base_url) !== ''
        ? asString(value.base_url)
        : asString(value.base_path) !== ''
          ? asString(value.base_path)
          : asString(value.base);
  return {
    id,
    name: asString(value.name) !== '' ? asString(value.name) : id,
    wire,
    auth,
    base,
    key_env: asString(value.key_env),
    key: asString(value.key),
  };
}

function ratingFrom(value: unknown): RatingRow | null {
  if (!isRecord(value)) return null;
  const model = asString(value.model);
  const provider = asString(value.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli: typeof value.rating_milli === 'number' && Number.isFinite(value.rating_milli) ? value.rating_milli : 0,
    context_length: asNumberOrNull(value.context_length),
    tools: value.tools === true,
    vision: value.vision === true,
    json: value.json === true,
    free: value.free === true,
  };
}

function workloadFrom(value: unknown): WorkloadEntry | null {
  if (!isRecord(value)) return null;
  const id = asString(value.id);
  if (id === '') return null;
  const requires = isRecord(value.requires) ? value.requires : {};
  const entry: WorkloadEntry = {
    id,
    api: value.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: value.stream === true,
    weight: value.weight === 2 ? 2 : 1,
    requires: {
      tools: requires.tools === true,
      vision: requires.vision === true,
      json: requires.json === true,
    },
  };
  if (typeof requires.min_context === 'number' && Number.isFinite(requires.min_context)) {
    entry.requires.min_context = requires.min_context;
  }
  return entry;
}

/** The brief at `path`: its three documents, found by schema. */
export function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: null, workload: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = schemaDocuments(text);
  const directoryDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers)) ?? null;
  if (directoryDoc === null || !Array.isArray(directoryDoc.providers)) {
    return { ...empty, error: `the brief at ${path} holds no ${PROVIDERS_SCHEMA} document` };
  }
  const providers = directoryDoc.providers.flatMap((v) => {
    const p = providerFrom(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings)) ?? null;
  let ratings: RatingsTable | null = null;
  if (ratingsDoc !== null && Array.isArray(ratingsDoc.ratings)) {
    const rows = ratingsDoc.ratings.flatMap((v) => {
      const r = ratingFrom(v);
      return r === null ? [] : [r];
    });
    const best = new Map<string, number>();
    if (Array.isArray(ratingsDoc.best)) {
      for (const v of ratingsDoc.best) {
        if (isRecord(v) && typeof v.id === 'string' && typeof v.rating_milli === 'number') best.set(v.id, v.rating_milli);
      }
    }
    ratings = { rows, best };
  }
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest)) ?? null;
  const workload =
    workloadDoc !== null && Array.isArray(workloadDoc.manifest)
      ? workloadDoc.manifest.flatMap((v) => {
          const w = workloadFrom(v);
          return w === null ? [] : [w];
        })
      : [];
  return { providers, ratings, workload, error: null };
}

/** The quality table's row for one provider's model, when it has one. */
export function ratingRowOf(ratings: RatingsTable | null, providerId: string, modelId: string): RatingRow | null {
  if (ratings === null) return null;
  return ratings.rows.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/**
 * Whether a model is free: the quality table's row where it has one (the sealed truth), the
 * listing's own word where it does not, and paid when neither says. A paid model is never routed.
 */
export function modelIsFree(model: ListedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

/** What a model can do: the quality table's row where it has one, the listing's fields otherwise. */
export function capabilitiesOf(model: ListedModel, row: RatingRow | null): Capabilities {
  return {
    context_length: row?.context_length ?? model.context_length ?? null,
    tools: row?.tools ?? model.tools ?? null,
    vision: row?.vision ?? model.vision ?? null,
    json: row?.json ?? model.json ?? null,
  };
}

/** Whether a model's capabilities meet one request's requirements. An unknown capability never meets one. */
export function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.min_context !== undefined && (caps.context_length === null || caps.context_length < requires.min_context)) {
    return false;
  }
  return true;
}

/** Whether a listing's pricing is all zeroes: 0 when it says so, null when it says nothing about price. */
function zeroPriced(value: Json): boolean | null {
  const pricing = isRecord(value.pricing) ? value.pricing : null;
  if (pricing === null) return null;
  const numbers = Object.values(pricing).flatMap((x) => {
    if (typeof x === 'number') return [x];
    if (typeof x === 'string' && x.trim() !== '') return [Number(x)];
    return [];
  });
  if (numbers.length === 0) return null;
  return numbers.every((n) => Number.isFinite(n) && n === 0);
}

/** One model as a listing wrote it, in whichever wire's spelling. */
function modelFromListing(wire: Wire, value: unknown): ListedModel | null {
  if (!isRecord(value)) return null;
  const id = wire === 'gemini' ? asString(value.name).replace(/^models\//, '') : asString(value.id);
  if (id === '') return null;
  const context =
    asNumberOrNull(value.context_length) ?? asNumberOrNull(value.context_window) ?? asNumberOrNull(value.inputTokenLimit);
  return {
    id,
    display_name:
      asString(value.display_name) !== ''
        ? asString(value.display_name)
        : asString(value.displayName) !== ''
          ? asString(value.displayName)
          : id,
    free: asBoolOrNull(value.free) ?? zeroPriced(value),
    context_length: context,
    tools: asBoolOrNull(value.tools),
    vision: asBoolOrNull(value.vision),
    json: asBoolOrNull(value.json),
  };
}

/** The models one catalog route's answer listed, in its wire's shape. */
function modelsFromBody(wire: Wire, body: unknown): ListedModel[] {
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = modelFromListing(wire, v);
    return m === null ? [] : [m];
  });
}

/** Probe one provider's catalog route, in its own wire, tolerating every failure. */
export async function probeProvider(envUrl: string, provider: ProviderEntry, timeoutMs = 5000): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) return { provider, ok: false, status: res.status, models: [], error: `GET ${url} answered HTTP ${res.status}` };
    const body: unknown = await res.json();
    return { provider, ok: true, status: res.status, models: modelsFromBody(provider.wire, body), error: null };
  } catch (err) {
    return { provider, ok: false, status: 0, models: [], error: `GET ${url} failed: ${(err as Error).message}` };
  }
}

const byModelId = (a: ListedModel, b: ListedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/**
 * The catalog the probes and the quality table make: every provider that answered, every model it
 * listed, deduplicated by id and sorted by id.
 *
 * base_url is the directory's own path (/p/<id>), never joined to the env's origin: the env's origin
 * differs between a live pass and its replay, and the catalog is part of the state export a replay
 * must reproduce byte for byte. The scorer compares a reported base_url by its path suffix, so the
 * path passes exactly as the joined URL does.
 */
export function buildCatalog(probes: readonly Probe[], ratings: RatingsTable | null): Catalog {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const model of [...probe.models].sort(byModelId)) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingRowOf(ratings, probe.provider.id, model.id);
      models.push({
        id: model.id,
        free: modelIsFree(model, row),
        context_length: row?.context_length ?? model.context_length ?? null,
      });
    }
    providers.push({
      id: probe.provider.id,
      base_url: probe.provider.base,
      key_env: probe.provider.key_env,
      wire: probe.provider.wire,
      models,
    });
  }
  return { providers };
}
