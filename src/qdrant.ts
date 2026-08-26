import type { Database } from "./db.js";
import { formatQueryForEmbedding, type LLM } from "./llm.js";
import { readFileSync } from "node:fs";

export type QdrantDomain = "public" | "shape";

export type QdrantSearch = {
  type: "lex" | "vec" | "hyde";
  query: string;
};

export type QdrantSearchOptions = {
  collections: string[];
  limit: number;
  candidateLimit?: number;
  llm: LLM;
  scope?: QdrantScope;
  hooks?: {
    onEmbedStart?: (count: number) => void;
    onEmbedDone?: (durationMs: number) => void;
  };
};

export type QdrantScope = {
  tenant: string;
  scopes: string[];
  access: string[];
};

export type QdrantDocumentResult = {
  file: string;
  displayPath: string;
  title: string;
  body: string;
  bestChunk: string;
  bestChunkPos: number;
  score: number;
  docid: string;
  collectionName: string;
  externalDocumentId?: string;
};

export type QdrantLexQuery = {
  positiveText: string;
  excludedTerms: string[];
};

type QdrantConfig = {
  url: string;
  apiKey: string;
  aliases: Record<QdrantDomain, string>;
  allowedDomains: Set<QdrantDomain>;
};

type QdrantPoint = {
  id: number | string;
  score: number;
  payload?: Record<string, unknown>;
};

type QdrantGroup = {
  id: string | number;
  hits: QdrantPoint[];
};

const SHAPE_COLLECTIONS = new Set(["wip", "shape_docusign"]);
const PUBLIC_COLLECTION_PREFIXES = [
  "jersey_city_",
  "nj_",
  "hudson_county_",
  "hoboken_",
  "weehawken_",
  "west_new_york_",
];

export function qdrantDomainForCollection(collection: string): QdrantDomain {
  if (
    SHAPE_COLLECTIONS.has(collection)
    || collection.startsWith("project-")
    || collection.startsWith("email-")
    || collection.startsWith("gdrive_")
    || collection.startsWith("gdrive-")
    || collection.startsWith("rooms-")
  ) {
    return "shape";
  }
  if (PUBLIC_COLLECTION_PREFIXES.some(prefix => collection.startsWith(prefix))) {
    return "public";
  }
  throw new Error(`Qdrant security domain is not classified for collection: ${collection}`);
}

export function isQdrantConfigured(): boolean {
  return Boolean(process.env.QMD_QDRANT_URL || process.env.QDRANT_URL);
}

export function parseQdrantLexQuery(query: string): QdrantLexQuery {
  const positive: string[] = [];
  const excludedTerms: string[] = [];
  for (const raw of query.match(/-?"[^"]+"|-?\S+/g) ?? []) {
    const excluded = raw.startsWith("-");
    const withoutPrefix = excluded ? raw.slice(1) : raw;
    const value = withoutPrefix.replace(/^"|"$/g, "").trim();
    if (!value) continue;
    if (excluded) excludedTerms.push(value.toLowerCase());
    else positive.push(value);
  }
  return { positiveText: positive.join(" "), excludedTerms };
}

export function dedupeQdrantSearches(searches: QdrantSearch[]): QdrantSearch[] {
  const seen = new Set<string>();
  return searches.filter(search => {
    const key = `${search.type}\0${search.query.trim()}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function loadQdrantConfig(): QdrantConfig {
  const url = (process.env.QMD_QDRANT_URL || process.env.QDRANT_URL || "").replace(/\/$/, "");
  const apiKeyFile = process.env.QMD_QDRANT_API_KEY_FILE?.trim();
  const apiKey = apiKeyFile
    ? readFileSync(apiKeyFile, "utf8").trim()
    : process.env.QMD_QDRANT_API_KEY || process.env.QDRANT_API_KEY || "";
  if (!url) throw new Error("QMD_QDRANT_URL is required for the Qdrant backend");
  if (!apiKey) throw new Error("QMD_QDRANT_API_KEY is required for the Qdrant backend");

  const rawAllowed = process.env.QMD_QDRANT_ALLOWED_DOMAINS || "";
  const allowedDomains = new Set(
    rawAllowed.split(",").map(value => value.trim()).filter(Boolean) as QdrantDomain[],
  );
  for (const domain of allowedDomains) {
    if (domain !== "public" && domain !== "shape") {
      throw new Error(`Unknown Qdrant security domain: ${domain}`);
    }
  }
  if (allowedDomains.size === 0) {
    throw new Error("QMD_QDRANT_ALLOWED_DOMAINS must explicitly allow public and/or shape");
  }

  return {
    url,
    apiKey,
    aliases: {
      public: process.env.QMD_QDRANT_PUBLIC_COLLECTION || "cellect_public_current",
      shape: process.env.QMD_QDRANT_SHAPE_COLLECTION || "tenant_shape_current",
    },
    allowedDomains,
  };
}

async function qdrantRequest<T>(config: QdrantConfig, path: string, body: unknown): Promise<T> {
  const response = await fetch(`${config.url}${path}`, {
    method: "POST",
    headers: {
      "api-key": config.apiKey,
      "content-type": "application/json",
    },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(60_000),
  });
  if (!response.ok) {
    const detail = (await response.text()).slice(0, 1000);
    throw new Error(`Qdrant ${response.status} ${response.statusText}: ${detail}`);
  }
  return response.json() as Promise<T>;
}

export function qdrantSearchFilter(
  collections: string[],
  scope?: QdrantScope,
): Record<string, unknown> {
  const must: Record<string, unknown>[] = [
    { key: "source_collection", match: { any: collections } },
  ];
  if (scope) {
    if (!scope.tenant || scope.scopes.length === 0 || scope.access.length === 0) {
      throw new Error("Scoped Qdrant search requires tenant, scopes, and access claims");
    }
    must.push(
      { key: "tenant_id", match: { value: scope.tenant } },
      { key: "scope_keys", match: { any: scope.scopes } },
      { key: "access_classes", match: { any: scope.access } },
    );
  }
  return { must };
}

function bodyPointQuery(
  search: QdrantSearch,
  embedding: number[] | null,
  filter: Record<string, unknown>,
  limit: number,
): Record<string, unknown>[] {
  if (search.type === "lex") {
    const parsed = parseQdrantLexQuery(search.query);
    if (!parsed.positiveText) throw new Error("Qdrant lexical search requires a positive term");
    const inference = {
      text: parsed.positiveText,
      model: "qdrant/bm25",
      options: { language: "english", avg_len: 650 },
    };
    return [{ query: inference, using: "body_bm25", filter, limit }];
  }
  if (!embedding) return [];
  return [{ query: embedding, using: "dense", filter, limit }];
}

function titlePointQuery(
  search: QdrantSearch,
  filter: Record<string, unknown>,
  limit: number,
): Record<string, unknown>[] {
  if (search.type !== "lex") return [];
  const parsed = parseQdrantLexQuery(search.query);
  if (!parsed.positiveText) throw new Error("Qdrant lexical search requires a positive term");
  return [{
    query: {
      text: parsed.positiveText,
      model: "qdrant/bm25",
      options: { language: "english", avg_len: 650 },
    },
    using: "title_bm25",
    filter,
    limit,
  }];
}

async function queryGrouped(
  config: QdrantConfig,
  domain: QdrantDomain,
  prefetch: Record<string, unknown>[],
  limit: number,
): Promise<QdrantPoint[]> {
  if (prefetch.length === 0) return [];
  const body = prefetch.length === 1
    ? { ...prefetch[0], group_by: "document_id", group_size: 1, limit, with_payload: true }
    : {
      prefetch,
      query: { fusion: "rrf" },
      group_by: "document_id",
      group_size: 1,
      limit,
      with_payload: true,
    };
  const result = await qdrantRequest<{ result: { groups: QdrantGroup[] } }>(
    config,
    `/collections/${encodeURIComponent(config.aliases[domain])}/points/query/groups`,
    body,
  );
  return result.result.groups.flatMap(group => group.hits.slice(0, 1));
}

export function mergeQdrantRankedPoints(lists: QdrantPoint[][], limit: number): QdrantPoint[] {
  const merged = new Map<string, { point: QdrantPoint; score: number }>();
  for (const list of lists) {
    list.forEach((point, rank) => {
      const documentId = String(point.payload?.document_id ?? "");
      if (!documentId) return;
      // Normalized RRF keeps a single-list result useful while rewarding a
      // document that appears in both title and body/dense ranked lists.
      const contribution = 60 / (61 + rank);
      const existing = merged.get(documentId);
      if (!existing) {
        merged.set(documentId, { point, score: contribution });
        return;
      }
      existing.score += contribution;
      if (existing.point.payload?.point_kind === "title" && point.payload?.point_kind !== "title") {
        existing.point = point;
      }
    });
  }
  const divisor = Math.max(1, lists.length);
  return [...merged.values()]
    .map(({ point, score }) => ({ ...point, score: score / divisor }))
    .sort((left, right) => right.score - left.score)
    .slice(0, limit);
}

async function queryDomain(
  config: QdrantConfig,
  domain: QdrantDomain,
  searches: QdrantSearch[],
  embeddings: Array<number[] | null>,
  collections: string[],
  limit: number,
  scope?: QdrantScope,
): Promise<QdrantPoint[]> {
  const filter = qdrantSearchFilter(collections, scope);
  // Grouping happens after prefetch, so fetch more chunks than the requested
  // number of document groups. Strict mode caps every query stage at 100.
  const prefetchLimit = Math.min(100, Math.max(40, limit * 2));
  const bodyPrefetch = searches.flatMap((search, index) => (
    bodyPointQuery(search, embeddings[index] ?? null, filter, prefetchLimit)
  ));
  const titlePrefetch = searches.flatMap(search => titlePointQuery(search, filter, prefetchLimit));
  const lists = await Promise.all(
    [bodyPrefetch, titlePrefetch]
      .filter(prefetch => prefetch.length > 0)
      .map(prefetch => queryGrouped(config, domain, prefetch, limit)),
  );
  return mergeQdrantRankedPoints(lists, limit);
}

function hydratePoints(
  db: Database,
  points: QdrantPoint[],
  limit: number,
  primaryQuery: string,
  excludedTerms: string[],
): QdrantDocumentResult[] {
  const statement = db.prepare(`
    SELECT d.id, d.collection, d.path, d.title, d.hash, c.doc AS body
    FROM documents d
    JOIN content c ON c.hash = d.hash
    WHERE d.id = ? AND d.active = 1
  `);
  const seen = new Set<number>();
  const results: QdrantDocumentResult[] = [];

  for (const point of points) {
    const documentId = Number(point.payload?.document_id);
    if (!Number.isSafeInteger(documentId) || seen.has(documentId)) continue;
    const row = statement.get(documentId) as {
      id: number;
      collection: string;
      path: string;
      title: string;
      hash: string;
      body: string;
    } | null | undefined;
    if (!row || row.collection !== point.payload?.source_collection) continue;
    if (excludedTerms.length > 0) {
      const searchable = `${row.title}\n${row.body}`.toLowerCase();
      if (excludedTerms.some(term => searchable.includes(term))) continue;
    }

    const position = Number(point.payload?.position ?? 0);
    let safePosition = Number.isSafeInteger(position) && position >= 0 ? position : 0;
    const chunkLength = Number(point.payload?.chunk_length ?? 2700);
    let safeLength = Number.isSafeInteger(chunkLength) && chunkLength > 0
      ? Math.min(chunkLength, 5000)
      : 2700;
    if (point.payload?.point_kind === "title") {
      const terms = primaryQuery
        .replace(/["']/g, "")
        .split(/\s+/)
        .map(term => term.replace(/^-/, "").toLowerCase())
        .filter(term => term.length > 2);
      const lowerBody = row.body.toLowerCase();
      const match = terms.map(term => lowerBody.indexOf(term)).find(index => index >= 0) ?? -1;
      safePosition = match >= 0 ? Math.max(0, match - 400) : 0;
      safeLength = 2700;
    }
    const displayPath = `${row.collection}/${row.path}`;
    seen.add(documentId);
    results.push({
      file: `qmd://${displayPath}`,
      displayPath,
      title: row.title,
      body: row.body,
      bestChunk: row.body.slice(safePosition, safePosition + safeLength),
      bestChunkPos: safePosition,
      score: Math.max(0, Math.min(1, point.score)),
      docid: row.hash.slice(0, 6),
      collectionName: row.collection,
      ...(typeof point.payload?.external_document_id === "string"
        ? { externalDocumentId: point.payload.external_document_id }
        : {}),
    });
    if (results.length >= limit) break;
  }
  return results;
}

export async function searchQdrant(
  db: Database,
  searches: QdrantSearch[],
  options: QdrantSearchOptions,
): Promise<QdrantDocumentResult[]> {
  const config = loadQdrantConfig();
  const collections = [...new Set(options.collections.filter(Boolean))];
  if (collections.length === 0) {
    throw new Error("Qdrant search requires at least one explicit QMD collection");
  }

  const grouped: Record<QdrantDomain, string[]> = { public: [], shape: [] };
  for (const collection of collections) grouped[qdrantDomainForCollection(collection)].push(collection);
  for (const domain of ["public", "shape"] as const) {
    if (grouped[domain].length > 0 && !config.allowedDomains.has(domain)) {
      throw new Error(`Qdrant security domain is not allowed by this runtime: ${domain}`);
    }
  }

  const vectorIndexes = searches
    .map((search, index) => ({ search, index }))
    .filter(({ search }) => search.type === "vec" || search.type === "hyde");
  const embeddings: Array<number[] | null> = searches.map(() => null);
  if (vectorIndexes.length > 0) {
    const formatted = vectorIndexes.map(({ search }) => formatQueryForEmbedding(search.query));
    options.hooks?.onEmbedStart?.(formatted.length);
    const embedStart = Date.now();
    const embedded = await options.llm.embedBatch(formatted);
    options.hooks?.onEmbedDone?.(Date.now() - embedStart);
    vectorIndexes.forEach(({ index }, resultIndex) => {
      embeddings[index] = embedded[resultIndex]?.embedding ?? null;
    });
  }

  // The production collections enforce Qdrant strict_mode max_query_limit=100.
  // Clamp oversized CLI/API candidate requests instead of turning `--all` or
  // a large candidateLimit into a backend 400.
  const candidateLimit = Math.min(100, Math.max(options.limit, options.candidateLimit ?? 40));
  const domainResults = await Promise.all(
    (["public", "shape"] as const)
      .filter(domain => grouped[domain].length > 0)
      .map(domain => queryDomain(
        config,
        domain,
        searches,
        embeddings,
        grouped[domain],
        candidateLimit,
        options.scope,
      )),
  );

  const merged = domainResults.flat().sort((a, b) => b.score - a.score);
  const primaryQuery = searches.find(search => search.type === "lex")?.query
    ?? searches.find(search => search.type === "vec")?.query
    ?? searches[0]?.query
    ?? "";
  const excludedTerms = searches
    .filter(search => search.type === "lex")
    .flatMap(search => parseQdrantLexQuery(search.query).excludedTerms);
  return hydratePoints(db, merged, options.limit, primaryQuery, excludedTerms);
}
