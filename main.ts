// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The server reads the brief at $MAKEFUN_PROMPT_FILE, probes every provider's catalog route,
// builds one catalog, registers its port and that catalog with the env at $MAKEFUN_ENV_URL,
// then answers the client's twenty requests and exits 0 the moment GET /work says done.
//
//   POST /v1/chat/completions  OpenAI Chat Completions, JSON or SSE when stream:true
//   POST /v1/messages          Anthropic Messages, JSON or SSE when stream:true
//   GET  /v1/models            the OpenAI list for Authorization: Bearer, the Anthropic list
//                             for x-api-key; exactly the free model ids registered, deduplicated
//
// Routing ranks every free model that meets a request's requirements by the quality table's
// rating and fails over to the next provider on a refusal (see lib/router.ts). Every wire
// combination is translated both ways (see lib/wire.ts). `node main.ts --list` prints the
// catalog and exits 0; outside a trial it prints {"providers":[]}.
//
// Determinism: every choice is a function of the brief, the requests and the provider answers
// seen so far. No wall clock, no unseeded randomness, fixed timestamps in model lists.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  isRecord,
  modelIsFree,
  probeProvider,
  ratingRowOf,
  readBrief,
  type Catalog,
  type Json,
  type ListedModel,
  type Probe,
  type Requirement,
} from './lib/providers.ts';
import { ProviderState, rankCandidates, serveRequest, type Candidate } from './lib/router.ts';
import {
  anthropicError,
  anthropicJsonToOpenAI,
  anthropicModelList,
  anthropicStreamToOpenAI,
  geminiJsonToAnthropic,
  geminiJsonToOpenAI,
  geminiStreamToAnthropic,
  geminiStreamToOpenAI,
  openAIError,
  openAIJsonToAnthropic,
  openAIModelList,
  openAIStreamToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one register attempt may take, in milliseconds. */
const REGISTER_TIMEOUT_MS = 5000;
/** Register attempts before giving up. */
const REGISTER_ATTEMPTS = 5;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
const state = new ProviderState();
let probes: Probe[] = [];
let catalog: Catalog = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(probes, brief.ratings);
}

const byModelId = (a: ListedModel, b: ListedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, deduplicated by id, in brief order. */
function servedModels(): ServedModel[] {
  const seen = new Set<string>();
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byModelId)) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingRowOf(brief.ratings, probe.provider.id, model.id);
      if (!modelIsFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json, clientWire: 'openai' | 'anthropic'): Requirement {
  const manifest = brief.workload;
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The manifest is the declared source of each request's requirements; this fallback is a
  // guess for a brief that holds none.
  return requirementsFromBody(body, clientWire);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json, clientWire: 'openai' | 'anthropic'): Requirement {
  let vision = false;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'image' || block.type === 'image_url') vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    // Only the OpenAI wire can ask for JSON mode; an Anthropic body carries no such field.
    json: clientWire === 'openai' && (format === 'json_object' || format === 'json_schema'),
  };
}

/** Whether an OpenAI client body asks for JSON mode. */
function openAIJsonRequested(body: Json): boolean {
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** Copy a provider's event stream to the client as it arrives, byte for byte. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  for await (const chunk of stream) res.write(Buffer.from(chunk));
}

/**
 * Deliver one provider's answer to the client: JSON translated into the client's wire, or an
 * SSE stream translated (or piped straight through when the wires already match).
 */
async function deliver(
  candidate: Candidate,
  upstream: Response,
  clientWire: 'openai' | 'anthropic',
  stream: boolean,
  res: ServerResponse,
): Promise<void> {
  const providerWire = candidate.provider.wire;
  const model = candidate.model.id;
  if (!stream) {
    const text = await upstream.text();
    let answer: unknown;
    try {
      answer = JSON.parse(text);
    } catch {
      return sendError(res, clientWire, 502, 'upstream_error', 'the provider answered a body that is not JSON');
    }
    if (!isRecord(answer)) {
      return sendError(res, clientWire, 502, 'upstream_error', 'the provider answered a JSON value that is not one object');
    }
    if (clientWire === 'openai') {
      if (providerWire === 'openai') sendJson(res, 200, answer);
      else if (providerWire === 'anthropic') sendJson(res, 200, anthropicJsonToOpenAI(answer, model));
      else sendJson(res, 200, geminiJsonToOpenAI(answer, model));
    } else {
      if (providerWire === 'anthropic') sendJson(res, 200, answer);
      else if (providerWire === 'openai') sendJson(res, 200, openAIJsonToAnthropic(answer, model));
      else sendJson(res, 200, geminiJsonToAnthropic(answer, model));
    }
    return;
  }
  if (upstream.body === null) {
    return sendError(res, clientWire, 502, 'upstream_error', 'the provider answered a stream with no body');
  }
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  try {
    if (clientWire === 'openai' && providerWire === 'openai') {
      await pipeStream(upstream.body, res);
    } else if (clientWire === 'openai' && providerWire === 'anthropic') {
      for await (const event of anthropicStreamToOpenAI(upstream.body, model)) res.write(event);
    } else if (clientWire === 'openai' && providerWire === 'gemini') {
      for await (const event of geminiStreamToOpenAI(upstream.body, model)) res.write(event);
    } else if (clientWire === 'anthropic' && providerWire === 'openai') {
      for await (const event of openAIStreamToAnthropic(upstream.body, model)) res.write(event);
    } else if (clientWire === 'anthropic' && providerWire === 'anthropic') {
      await pipeStream(upstream.body, res);
    } else {
      for await (const event of geminiStreamToAnthropic(upstream.body, model)) res.write(event);
    }
  } finally {
    res.end();
  }
}

/** Serve one client request on either wire: rank the candidates, fail over on refusal. */
async function serve(
  req: IncomingMessage,
  res: ServerResponse,
  clientWire: 'openai' | 'anthropic',
): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    return sendError(res, clientWire, 400, 'invalid_request_error', 'the request body is not one JSON object');
  }
  const requires = requirementsFor(body, clientWire);
  const stream = body.stream === true;
  const jsonMode = clientWire === 'openai' && openAIJsonRequested(body);
  const candidates = rankCandidates(probes, brief.ratings, requires, clientWire, jsonMode, state);
  if (candidates.length === 0) {
    return sendError(
      res,
      clientWire,
      503,
      clientWire === 'anthropic' ? 'overloaded_error' : 'no_free_model',
      'no free model on the directory meets this request',
    );
  }
  const outcome = await serveRequest(ENV_URL, state, candidates, clientWire, body, stream, (candidate, upstream) =>
    deliver(candidate, upstream, clientWire, stream, res),
  log);
  if (outcome === 'unserved') {
    if (!res.headersSent) {
      sendError(
        res,
        clientWire,
        503,
        clientWire === 'anthropic' ? 'overloaded_error' : 'service_unavailable',
        'every free model that meets this request refused it',
      );
    }
  }
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serve(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serve(req, res, 'anthropic');
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: Catalog): Promise<void> {
  let lastError: Error | null = null;
  for (let attempt = 1; attempt <= REGISTER_ATTEMPTS; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(REGISTER_TIMEOUT_MS),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') {
        log(`registered: the workload holds ${reply.requests} requests`);
      }
      return;
    } catch (err) {
      lastError = err as Error;
      log(`register attempt ${attempt} failed: ${lastError.message}`);
      if (attempt < REGISTER_ATTEMPTS) await sleep(250);
    }
  }
  throw lastError ?? new Error('register failed');
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and
    // stop before any server starts. A process whose stdout is not flushed loses the tail, so wait.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
