import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { openDatabase, type Database } from "./db.js";
import { formatDocForEmbedding } from "./llm.js";
import { getDefaultRemoteLLM } from "./llm-remote.js";
import { chunkDocument } from "./store.js";
import { qdrantDomainForCollection, type QdrantDomain } from "./qdrant.js";

type ImportState = {
  lastDocumentId: number;
  documents: number;
  points: number;
  updatedAt: string;
};

export type DocumentRow = {
  id: number;
  collection: string;
  path: string;
  title: string;
  hash: string;
  modified_at: string;
  body: string;
};

export type QdrantPoint = {
  id: number;
  vector: Record<string, unknown>;
  payload: Record<string, unknown>;
};

const REMOTE_CHUNK_MAX_CHARS = 900 * 3;
const REMOTE_CHUNK_OVERLAP_CHARS = Math.floor(900 * 0.15) * 3;
const REMOTE_CHUNK_WINDOW_CHARS = 200 * 3;
const EMBEDDING_DIMENSIONS = 768;
const EMBED_BATCH_SIZE = 32;
export const QDRANT_EMBED_SPEC = "embeddinggemma-768|v1";
export const QDRANT_CHUNK_SPEC = "chars-2700-overlap-405-window-600|v2-direct";
export const QDRANT_LEGACY_CHUNK_SPEC = "legacy-sqlite";

class NonRetryableQdrantError extends Error {}

function sleep(milliseconds: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, milliseconds));
}

export function isRetryableQdrantStatus(status: number): boolean {
  return status === 408
    || status === 409
    || status === 425
    || status === 429
    || status >= 500;
}

function sanitizeText(value: string): string {
  let wellFormed = "";
  for (let index = 0; index < value.length; index++) {
    const code = value.charCodeAt(index);
    if (code >= 0xD800 && code <= 0xDBFF) {
      const next = value.charCodeAt(index + 1);
      if (next >= 0xDC00 && next <= 0xDFFF) {
        wellFormed += value[index]! + value[index + 1]!;
        index += 1;
      } else {
        wellFormed += "\uFFFD";
      }
    } else if (code >= 0xDC00 && code <= 0xDFFF) {
      wellFormed += "\uFFFD";
    } else {
      wellFormed += value[index];
    }
  }
  return wellFormed
    .replace(/\u0000/g, " ")
    .replace(/[\u0001-\u0008\u000B\u000C\u000E-\u001F\u007F-\u009F]/g, " ");
}

function argument(name: string, fallback?: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : fallback;
}

function integerArgument(name: string, fallback: number): number {
  const value = Number(argument(name, String(fallback)));
  if (!Number.isSafeInteger(value) || value < 0) throw new Error(`${name} must be a non-negative integer`);
  return value;
}

function config() {
  const url = (process.env.QMD_QDRANT_URL || process.env.QDRANT_URL || "").replace(/\/$/, "");
  const apiKey = process.env.QMD_QDRANT_API_KEY || process.env.QDRANT_API_KEY || "";
  if (!url || !apiKey) throw new Error("QMD_QDRANT_URL and QMD_QDRANT_API_KEY are required");
  return {
    url,
    apiKey,
    aliases: {
      public: process.env.QMD_QDRANT_PUBLIC_COLLECTION || "cellect_public_current",
      shape: process.env.QMD_QDRANT_SHAPE_COLLECTION || "tenant_shape_current",
    } satisfies Record<QdrantDomain, string>,
  };
}

async function upsert(
  endpoint: ReturnType<typeof config>,
  domain: QdrantDomain,
  points: QdrantPoint[],
): Promise<void> {
  if (points.length === 0) return;
  const url = `${endpoint.url}/collections/${encodeURIComponent(endpoint.aliases[domain])}/points?wait=true`;
  let lastError: unknown;
  for (let attempt = 1; attempt <= 6; attempt++) {
    try {
      const response = await fetch(url, {
        method: "PUT",
        headers: {
          "api-key": endpoint.apiKey,
          "content-type": "application/json",
        },
        body: JSON.stringify({ points }),
        signal: AbortSignal.timeout(120_000),
      });
      if (response.ok) return;
      const detail = (await response.text()).slice(0, 1000);
      const message = `Qdrant ${response.status} ${response.statusText}: ${detail}`;
      if (response.status >= 400 && !isRetryableQdrantStatus(response.status)) {
        throw new NonRetryableQdrantError(message);
      }
      throw new Error(message);
    } catch (error) {
      if (error instanceof NonRetryableQdrantError) throw error;
      lastError = error;
      if (attempt < 6) await sleep(Math.min(30_000, 1000 * (2 ** (attempt - 1))));
    }
  }
  throw lastError;
}

async function deleteDocumentPoints(
  endpoint: ReturnType<typeof config>,
  domain: QdrantDomain,
  documentId: number,
): Promise<void> {
  const url = `${endpoint.url}/collections/${encodeURIComponent(endpoint.aliases[domain])}/points/delete?wait=true`;
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "api-key": endpoint.apiKey,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      filter: { must: [{ key: "document_id", match: { value: String(documentId) } }] },
    }),
    signal: AbortSignal.timeout(120_000),
  });
  if (!response.ok) {
    const detail = (await response.text()).slice(0, 1000);
    throw new Error(`Qdrant delete ${response.status} ${response.statusText}: ${detail}`);
  }
}

async function documentPointIds(
  endpoint: ReturnType<typeof config>,
  domain: QdrantDomain,
  documentId: number,
): Promise<Array<number | string>> {
  const url = `${endpoint.url}/collections/${encodeURIComponent(endpoint.aliases[domain])}/points/scroll`;
  const ids: Array<number | string> = [];
  let offset: number | string | null = null;
  do {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "api-key": endpoint.apiKey,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        filter: { must: [{ key: "document_id", match: { value: String(documentId) } }] },
        limit: 100,
        with_payload: false,
        with_vector: false,
        ...(offset === null ? {} : { offset }),
      }),
      signal: AbortSignal.timeout(120_000),
    });
    if (!response.ok) {
      const detail = (await response.text()).slice(0, 1000);
      throw new Error(`Qdrant scroll ${response.status} ${response.statusText}: ${detail}`);
    }
    const data = await response.json() as {
      result: { points: Array<{ id: number | string }>; next_page_offset?: number | string | null };
    };
    ids.push(...data.result.points.map(point => point.id));
    offset = data.result.next_page_offset ?? null;
  } while (offset !== null);
  return ids;
}

export function stalePointIds(
  previousIds: Array<number | string>,
  nextIds: Array<number | string>,
): Array<number | string> {
  const keep = new Set(nextIds.map(String));
  return previousIds.filter(id => !keep.has(String(id)));
}

async function deletePointIds(
  endpoint: ReturnType<typeof config>,
  domain: QdrantDomain,
  ids: Array<number | string>,
): Promise<void> {
  if (ids.length === 0) return;
  const url = `${endpoint.url}/collections/${encodeURIComponent(endpoint.aliases[domain])}/points/delete?wait=true`;
  for (let offset = 0; offset < ids.length; offset += 100) {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "api-key": endpoint.apiKey,
        "content-type": "application/json",
      },
      body: JSON.stringify({ points: ids.slice(offset, offset + 100) }),
      signal: AbortSignal.timeout(120_000),
    });
    if (!response.ok) {
      const detail = (await response.text()).slice(0, 1000);
      throw new Error(`Qdrant point delete ${response.status} ${response.statusText}: ${detail}`);
    }
  }
}

function pointId(documentId: number, seq: number): number {
  const id = documentId * 10_000_000 + seq + 1;
  if (!Number.isSafeInteger(id)) throw new Error(`Document ${documentId} cannot be represented as a stable point ID`);
  return id;
}

function sparse(text: string): Record<string, unknown> {
  return {
    text: sanitizeText(text),
    model: "qdrant/bm25",
    options: { language: "english", avg_len: 650 },
  };
}

function payload(
  document: DocumentRow,
  seq: number,
  position: number,
  chunkLength: number,
  pointKind: "title" | "chunk",
) {
  return {
    document_id: String(document.id),
    source_collection: sanitizeText(document.collection),
    source_type: "qmd",
    modified_at: document.modified_at,
    path: sanitizeText(document.path),
    title: sanitizeText(document.title),
    hash: document.hash,
    seq,
    position,
    chunk_length: chunkLength,
    point_kind: pointKind,
  };
}

export function expectedPointCount(chunkCount: number): number {
  return 1 + Math.max(1, chunkCount);
}

export type Embedder = {
  embedBatch(texts: string[]): Promise<Array<{ embedding: number[]; model: string } | null>>;
};

async function embedChunks(
  document: DocumentRow,
  chunks: Array<{ text: string; pos: number }>,
  embedder: Embedder,
): Promise<number[][]> {
  const embeddings: number[][] = [];
  for (let offset = 0; offset < chunks.length; offset += EMBED_BATCH_SIZE) {
    const batch = chunks.slice(offset, offset + EMBED_BATCH_SIZE);
    const texts = batch.map(chunk => sanitizeText(formatDocForEmbedding(chunk.text, document.title)));
    let pending = batch.map((_, index) => index);
    const results: Array<number[] | null> = batch.map(() => null);

    for (let attempt = 1; attempt <= 3 && pending.length > 0; attempt++) {
      const retryResults = await embedder.embedBatch(pending.map(index => texts[index]!));
      if (retryResults.length !== pending.length) {
        throw new Error(`Embedding endpoint returned ${retryResults.length} results for ${pending.length} inputs`);
      }
      const nextPending: number[] = [];
      retryResults.forEach((result, retryIndex) => {
        const batchIndex = pending[retryIndex]!;
        if (!result) {
          nextPending.push(batchIndex);
          return;
        }
        if (result.embedding.length !== EMBEDDING_DIMENSIONS || result.embedding.some(value => !Number.isFinite(value))) {
          throw new Error(
            `Unexpected embedding for document ${document.id} chunk ${offset + batchIndex}: `
            + `${result.embedding.length} dimensions`,
          );
        }
        results[batchIndex] = result.embedding;
      });
      pending = nextPending;
      if (pending.length > 0 && attempt < 3) await sleep(1000 * attempt);
    }
    if (pending.length > 0) {
      throw new Error(`Embedding failed for document ${document.id} chunks ${pending.map(index => offset + index).join(",")}`);
    }
    embeddings.push(...results.map(result => result!));
  }
  return embeddings;
}

export async function buildQdrantPoints(document: DocumentRow, embedder: Embedder): Promise<QdrantPoint[]> {
  const chunks = chunkDocument(
    document.body,
    REMOTE_CHUNK_MAX_CHARS,
    REMOTE_CHUNK_OVERLAP_CHARS,
    REMOTE_CHUNK_WINDOW_CHARS,
  );
  const embeddings = await embedChunks(document, chunks, embedder);

  const titlePoint: QdrantPoint = {
    id: pointId(document.id, -1),
    vector: { title_bm25: sparse(document.title) },
    payload: payload(document, -1, 0, 0, "title"),
  };

  return [titlePoint, ...chunks.map((chunk, seq) => {
    const vector: Record<string, unknown> = {
      body_bm25: sparse(chunk.text),
      dense: embeddings[seq]!,
    };
    return {
      id: pointId(document.id, seq),
      vector,
      payload: payload(document, seq, chunk.pos, chunk.text.length, "chunk"),
    };
  })];
}

function readState(path: string): ImportState {
  if (!existsSync(path)) {
    return { lastDocumentId: 0, documents: 0, points: 0, updatedAt: new Date(0).toISOString() };
  }
  return JSON.parse(readFileSync(path, "utf8")) as ImportState;
}

function saveState(path: string, state: ImportState): void {
  writeFileSync(path, `${JSON.stringify(state, null, 2)}\n`, { mode: 0o600 });
}

export function initializeManifest(db: Database): void {
  db.exec(`
    CREATE TABLE IF NOT EXISTS qdrant_documents (
      document_id INTEGER PRIMARY KEY,
      hash TEXT NOT NULL,
      collection TEXT NOT NULL,
      point_count INTEGER NOT NULL,
      embed_spec TEXT NOT NULL DEFAULT '${QDRANT_EMBED_SPEC}',
      chunk_spec TEXT NOT NULL DEFAULT '${QDRANT_CHUNK_SPEC}',
      synced_at TEXT NOT NULL
    )
  `);
  const columns = db.prepare(`PRAGMA table_info(qdrant_documents)`).all() as Array<{ name: string }>;
  if (!columns.some(column => column.name === "embed_spec")) {
    // The deployed SQLite vectors were produced by this same embedding model
    // and chunker. Adopt that verified generation without re-embedding all
    // 6.25 million chunks merely to transition the persistence path.
    db.exec(`ALTER TABLE qdrant_documents ADD COLUMN embed_spec TEXT NOT NULL DEFAULT '${QDRANT_EMBED_SPEC}'`);
  }
  const migratedColumns = db.prepare(`PRAGMA table_info(qdrant_documents)`).all() as Array<{ name: string }>;
  if (!migratedColumns.some(column => column.name === "chunk_spec")) {
    // Existing points remain valid embeddinggemma vectors, but some were
    // produced by QMD's older token-based chunker. Preserve that provenance;
    // touched documents move to the direct chunk spec, while --rebuild
    // deliberately normalizes an entire generation.
    db.exec(`ALTER TABLE qdrant_documents ADD COLUMN chunk_spec TEXT NOT NULL DEFAULT '${QDRANT_LEGACY_CHUNK_SPEC}'`);
  }
  db.prepare(`
    UPDATE qdrant_documents
    SET embed_spec = ?
    WHERE embed_spec = 'embeddinggemma-768|chars-2700-overlap-405-window-600|v2-direct'
  `).run(QDRANT_EMBED_SPEC);
}

async function importDocument(
  manifest: Database,
  endpoint: ReturnType<typeof config>,
  document: DocumentRow,
  embedder: Embedder,
  batchSize: number,
  concurrency: number,
  replaceExisting: boolean,
): Promise<number> {
  const domain = qdrantDomainForCollection(document.collection);
  const previous = manifest.prepare(`
    SELECT hash, collection FROM qdrant_documents WHERE document_id = ?
  `).get(document.id) as { hash: string; collection: string } | null | undefined;

  // Finish embedding before changing the live collection. A failed GPU call
  // must leave the prior searchable points and manifest row untouched.
  const points = await buildQdrantPoints(document, embedder);
  const previousDomain = previous ? qdrantDomainForCollection(previous.collection) : null;
  const previousIds = replaceExisting && previousDomain === domain
    ? await documentPointIds(endpoint, domain, document.id)
    : [];
  const batches: QdrantPoint[][] = [];
  for (let offset = 0; offset < points.length; offset += batchSize) {
    batches.push(points.slice(offset, offset + batchSize));
  }
  for (let offset = 0; offset < batches.length; offset += concurrency) {
    await Promise.all(
      batches.slice(offset, offset + concurrency).map(batch => upsert(endpoint, domain, batch)),
    );
  }
  if (replaceExisting && previousDomain !== null) {
    if (previousDomain !== domain) {
      await deleteDocumentPoints(endpoint, previousDomain, document.id);
    } else {
      await deletePointIds(endpoint, domain, stalePointIds(previousIds, points.map(point => point.id)));
    }
  }
  manifest.prepare(`
    INSERT INTO qdrant_documents (document_id, hash, collection, point_count, embed_spec, chunk_spec, synced_at)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(document_id) DO UPDATE SET
      hash = excluded.hash,
      collection = excluded.collection,
      point_count = excluded.point_count,
      embed_spec = excluded.embed_spec,
      chunk_spec = excluded.chunk_spec,
      synced_at = excluded.synced_at
  `).run(
    document.id,
    document.hash,
    document.collection,
    points.length,
    QDRANT_EMBED_SPEC,
    QDRANT_CHUNK_SPEC,
    new Date().toISOString(),
  );
  return points.length;
}

async function syncChangedDocuments(
  source: Database,
  manifest: Database,
  endpoint: ReturnType<typeof config>,
  embedder: Embedder,
  batchSize: number,
  concurrency: number,
  documentConcurrency: number,
): Promise<{ updated: number; removed: number; points: number }> {
  const documentQuery = source.prepare(`
    SELECT d.id, d.collection, d.path, d.title, d.hash, d.modified_at, c.doc AS body
    FROM documents d
    JOIN content c ON c.hash = d.hash
    WHERE d.active = 1 AND d.id > ?
    ORDER BY d.id
    LIMIT 25
  `);
  const manifestLookup = manifest.prepare(`
    SELECT hash, collection, point_count, embed_spec, chunk_spec FROM qdrant_documents WHERE document_id = ?
  `);
  let updated = 0;
  let points = 0;
  let cursor = 0;
  while (true) {
    const documents = documentQuery.all(cursor) as DocumentRow[];
    if (documents.length === 0) break;
    const changed: Array<{ document: DocumentRow; replaceExisting: boolean }> = [];
    for (const document of documents) {
      cursor = document.id;
      const previous = manifestLookup.get(document.id) as {
        hash: string;
        collection: string;
        point_count: number;
        embed_spec: string;
        chunk_spec: string;
      } | null | undefined;
      if (
        previous?.hash === document.hash
        && previous.collection === document.collection
        && previous.embed_spec === QDRANT_EMBED_SPEC
      ) {
        if (previous.chunk_spec === QDRANT_LEGACY_CHUNK_SPEC) continue;
        if (previous.chunk_spec === QDRANT_CHUNK_SPEC) {
          const chunkCount = chunkDocument(
            document.body,
            REMOTE_CHUNK_MAX_CHARS,
            REMOTE_CHUNK_OVERLAP_CHARS,
            REMOTE_CHUNK_WINDOW_CHARS,
          ).length;
          if (previous.point_count === expectedPointCount(chunkCount)) continue;
        }
      }
      changed.push({ document, replaceExisting: Boolean(previous) });
    }
    for (let offset = 0; offset < changed.length; offset += documentConcurrency) {
      const group = changed.slice(offset, offset + documentConcurrency);
      const counts = await Promise.all(group.map(item => importDocument(
        manifest,
        endpoint,
        item.document,
        embedder,
        batchSize,
        concurrency,
        item.replaceExisting,
      )));
      updated += group.length;
      points += counts.reduce((sum, count) => sum + count, 0);
      if (Math.floor((updated - group.length) / 100) !== Math.floor(updated / 100)) {
        console.log(JSON.stringify({ status: "syncing", updated, points }));
      }
    }
  }

  const stale = manifest.prepare(`
    SELECT document_id, collection FROM qdrant_documents ORDER BY document_id
  `).all() as Array<{ document_id: number; collection: string }>;
  const activeLookup = source.prepare(`SELECT 1 AS present FROM documents WHERE id = ? AND active = 1`);
  let removed = 0;
  for (const document of stale) {
    if (activeLookup.get(document.document_id)) continue;
    await deleteDocumentPoints(endpoint, qdrantDomainForCollection(document.collection), document.document_id);
    manifest.prepare(`DELETE FROM qdrant_documents WHERE document_id = ?`).run(document.document_id);
    removed += 1;
  }
  return { updated, removed, points };
}

export async function importQdrant(): Promise<void> {
  const dbPath = argument("--db", process.env.QMD_INDEX_PATH || "/home/node/.qmd/index.sqlite")!;
  const statePath = argument("--state", process.env.QMD_QDRANT_IMPORT_STATE || "/home/node/.qmd/qdrant-import.json")!;
  const manifestPath = argument("--manifest", process.env.QMD_QDRANT_MANIFEST || "/home/node/.qmd/qdrant-manifest.sqlite")!;
  const batchSize = Math.max(1, integerArgument("--batch-size", 96));
  const concurrency = Math.max(1, Math.min(16, integerArgument("--concurrency", 4)));
  const documentConcurrency = Math.max(1, Math.min(32, integerArgument("--document-concurrency", 4)));
  const documentLimit = integerArgument("--limit", 0);
  const documentId = integerArgument("--document-id", 0);
  const rebuild = process.argv.includes("--rebuild");
  const domainArg = argument("--domain", "all");
  if (!domainArg || !["all", "public", "shape"].includes(domainArg)) {
    throw new Error("--domain must be all, public, or shape");
  }

  const state = readState(statePath);
  if (rebuild) {
    state.lastDocumentId = 0;
    state.documents = 0;
    state.points = 0;
    state.updatedAt = new Date().toISOString();
    saveState(statePath, state);
  }
  const afterId = integerArgument("--after-id", state.lastDocumentId);
  const db = openDatabase(dbPath);
  const manifest = openDatabase(manifestPath);
  initializeManifest(manifest);
  const endpoint = config();
  const embedder = getDefaultRemoteLLM();
  const documentQuery = db.prepare(`
    SELECT d.id, d.collection, d.path, d.title, d.hash, d.modified_at, c.doc AS body
    FROM documents d
    JOIN content c ON c.hash = d.hash
    WHERE d.active = 1 AND d.id > ?
    ORDER BY d.id
    LIMIT ?
  `);

  let cursor = afterId;
  let processedThisRun = 0;
  const startedAt = Date.now();
  let reconciliation: { updated: number; removed: number; points: number };
  try {
    if (documentId > 0) {
      const document = db.prepare(`
        SELECT d.id, d.collection, d.path, d.title, d.hash, d.modified_at, c.doc AS body
        FROM documents d
        JOIN content c ON c.hash = d.hash
        WHERE d.active = 1 AND d.id = ?
      `).get(documentId) as DocumentRow | null | undefined;
      if (!document) throw new Error(`Active document ${documentId} was not found`);
      const points = await importDocument(manifest, endpoint, document, embedder, batchSize, concurrency, true);
      console.log(JSON.stringify({
        status: "complete",
        documentId,
        points,
        embedSpec: QDRANT_EMBED_SPEC,
        chunkSpec: QDRANT_CHUNK_SPEC,
      }));
      return;
    }
    while (documentLimit === 0 || processedThisRun < documentLimit) {
      const fetchLimit = Math.min(32, documentLimit === 0 ? 32 : documentLimit - processedThisRun);
      const documents = documentQuery.all(cursor, fetchLimit) as DocumentRow[];
      if (documents.length === 0) break;

      for (let offset = 0; offset < documents.length; offset += documentConcurrency) {
        const group = documents.slice(offset, offset + documentConcurrency);
        const selected = group.filter(document => {
          const domain = qdrantDomainForCollection(document.collection);
          return domainArg === "all" || domain === domainArg;
        });
        const previousDocumentCount = state.documents;
        const pointCounts = await Promise.all(
          selected.map(document => importDocument(
            manifest,
            endpoint,
            document,
            embedder,
            batchSize,
            concurrency,
            rebuild,
          )),
        );
        cursor = group.at(-1)!.id;
        state.lastDocumentId = cursor;
        state.documents += selected.length;
        state.points += pointCounts.reduce((sum, count) => sum + count, 0);
        state.updatedAt = new Date().toISOString();
        saveState(statePath, state);
        processedThisRun += selected.length;

        if (
          Math.floor(previousDocumentCount / 100) !== Math.floor(state.documents / 100)
          || processedThisRun === selected.length
        ) {
          const elapsedSeconds = Math.max(1, Math.round((Date.now() - startedAt) / 1000));
          console.log(JSON.stringify({
            status: "running",
            lastDocumentId: state.lastDocumentId,
            documents: state.documents,
            points: state.points,
            documentsPerSecond: Number((processedThisRun / elapsedSeconds).toFixed(2)),
          }));
        }
        if (documentLimit > 0 && processedThisRun >= documentLimit) break;
      }
    }
    reconciliation = documentLimit === 0
      ? await syncChangedDocuments(db, manifest, endpoint, embedder, batchSize, concurrency, documentConcurrency)
      : { updated: 0, removed: 0, points: 0 };
    if (documentLimit === 0 && domainArg === "all") {
      const totals = manifest.prepare(`
        SELECT count(*) AS documents, coalesce(sum(point_count), 0) AS points
        FROM qdrant_documents
      `).get() as { documents: number; points: number };
      state.documents = totals.documents;
      state.points = totals.points;
      state.updatedAt = new Date().toISOString();
      saveState(statePath, state);
    }
  } finally {
    db.close();
    manifest.close();
  }

  console.log(JSON.stringify({ status: "complete", ...state, reconciliation }));
}

if (import.meta.main) {
  await importQdrant();
}
