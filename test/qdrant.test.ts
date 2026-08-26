import { afterEach, describe, expect, test, vi } from "vitest";
import {
  dedupeQdrantSearches,
  mergeQdrantRankedPoints,
  parseQdrantLexQuery,
  qdrantDomainForCollection,
  qdrantSearchFilter,
} from "../src/qdrant.js";
import {
  aclPayloadForDocument,
  buildQdrantPoints,
  expectedPointCount,
  initializeManifest,
  isRetryableQdrantStatus,
  QDRANT_EMBED_SPEC,
  QDRANT_LEGACY_CHUNK_SPEC,
  stalePointIds,
} from "../src/qdrant-import.js";
import type { Database } from "../src/db.js";
import { hybridQuery, vectorSearchQuery, type Store } from "../src/store.js";
import type { LLM } from "../src/llm.js";

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

describe("Qdrant importer retry policy", () => {
  test.each([408, 409, 425, 429, 500, 503])("retries transient HTTP %s", status => {
    expect(isRetryableQdrantStatus(status)).toBe(true);
  });

  test.each([400, 401, 403, 404, 422])("fails fast for permanent HTTP %s", status => {
    expect(isRetryableQdrantStatus(status)).toBe(false);
  });
});

describe("Qdrant importer point accounting", () => {
  test.each([
    [0, 2],
    [1, 2],
    [13, 14],
  ])("maps %s vector rows to %s title/body points", (rows, points) => {
    expect(expectedPointCount(rows)).toBe(points);
  });
});

describe("Qdrant direct embedding", () => {
  const document = {
    id: 7,
    collection: "jersey_city_code",
    path: "chapter.md",
    title: "Chapter",
    hash: "abc123",
    modified_at: "2026-08-11T00:00:00Z",
    body: "A short document about zoning.",
  };

  test("builds dense and BM25 points without SQLite vectors", async () => {
    const embedBatch = vi.fn(async (texts: string[]) => texts.map(() => ({
      embedding: Array(768).fill(0.25),
      model: "embeddinggemma",
    })));
    const points = await buildQdrantPoints(document, { embedBatch });

    expect(embedBatch).toHaveBeenCalledOnce();
    expect(points).toHaveLength(2);
    expect(points[0]?.vector).toHaveProperty("title_bm25");
    expect(points[1]?.vector).toHaveProperty("body_bm25");
    expect(points[1]?.vector.dense).toHaveLength(768);
    expect(points[1]?.payload).toMatchObject({ document_id: "7", point_kind: "chunk" });
  });

  test("rejects a wrong embedding dimension before producing points", async () => {
    await expect(buildQdrantPoints(document, {
      embedBatch: async texts => texts.map(() => ({ embedding: [0.1, 0.2], model: "wrong" })),
    })).rejects.toThrow("Unexpected embedding");
  });

  test("removes only stale point IDs after an overlap-safe upsert", () => {
    expect(stalePointIds([100, 101, 102, "103"], [100, 101, 104, 103])).toEqual([102]);
  });

  test("migrates the deployed manifest to a versioned embedding spec", () => {
    const statements: string[] = [];
    let pragmaReads = 0;
    const db = {
      exec: (sql: string) => { statements.push(sql); },
      prepare: (sql: string) => sql.includes("PRAGMA")
        ? ({ all: () => {
          pragmaReads += 1;
          return [
            { name: "document_id" },
            { name: "hash" },
            { name: "collection" },
            { name: "point_count" },
            ...(pragmaReads > 1 ? [{ name: "embed_spec" }] : []),
            { name: "synced_at" },
          ];
        } })
        : ({ run: () => ({ changes: 1 }) }),
    } as unknown as Database;
    initializeManifest(db);
    expect(statements.some(sql => sql.includes("ALTER TABLE qdrant_documents ADD COLUMN embed_spec"))).toBe(true);
    expect(statements.some(sql => sql.includes("ALTER TABLE qdrant_documents ADD COLUMN chunk_spec"))).toBe(true);
    expect(statements.join("\n")).toContain(QDRANT_EMBED_SPEC);
    expect(statements.join("\n")).toContain(QDRANT_LEGACY_CHUNK_SPEC);
  });
});

describe("Qdrant collection security domains", () => {
  test.each([
    ["jersey_city_code", "public"],
    ["hoboken_transcripts", "public"],
    ["project-the-holland", "shape"],
    ["email-alessiaperini-shapeequity-com", "shape"],
    ["gdrive_alex-shape-tech", "shape"],
    ["shape_docusign", "shape"],
    ["wip", "shape"],
    ["rooms-shape-24-bright", "shape"],
  ])("maps %s to %s", (collection, domain) => {
    expect(qdrantDomainForCollection(collection)).toBe(domain);
  });

  test("fails closed for a new unclassified collection", () => {
    expect(() => qdrantDomainForCollection("tenant-acme")).toThrow("not classified");
  });
});

describe("Qdrant scoped ACL", () => {
  test("adds every mandatory authorization dimension to the retrieval filter", () => {
    expect(qdrantSearchFilter(["rooms-shape-24-bright"], {
      tenant: "shape",
      scopes: ["project:24-bright-street"],
      access: ["documents", "construction"],
    })).toEqual({ must: [
      { key: "source_collection", match: { any: ["rooms-shape-24-bright"] } },
      { key: "tenant_id", match: { value: "shape" } },
      { key: "scope_keys", match: { any: ["project:24-bright-street"] } },
      { key: "access_classes", match: { any: ["documents", "construction"] } },
    ] });
  });

  test("fails closed when an ACL is incomplete", () => {
    expect(() => qdrantSearchFilter(["rooms-shape-24-bright"], {
      tenant: "shape", scopes: [], access: ["documents"],
    })).toThrow("requires tenant, scopes, and access");
    expect(() => aclPayloadForDocument({ path: "missing.md" }, {
      QMD_ACL_MANIFEST_REQUIRED: "1",
    })).toThrow("QMD_ACL_MANIFEST is required");
  });
});

describe("Qdrant document-level rank fusion", () => {
  test("rewards cross-representation matches and keeps the body hit", () => {
    const title = { id: 1, score: 10, payload: { document_id: "7", point_kind: "title" } };
    const body = { id: 2, score: 9, payload: { document_id: "7", point_kind: "chunk" } };
    const other = { id: 3, score: 8, payload: { document_id: "8", point_kind: "chunk" } };
    const result = mergeQdrantRankedPoints([[title], [other, body]], 10);

    expect(result[0]?.payload?.document_id).toBe("7");
    expect(result[0]?.payload?.point_kind).toBe("chunk");
    expect(result[0]?.score).toBeGreaterThan(result[1]?.score ?? 0);
  });
});

describe("Qdrant lexical query translation", () => {
  test("removes syntax punctuation and keeps negative terms out of BM25 input", () => {
    expect(parseQdrantLexQuery('\"Looseleaf Supplement\" -baseball')).toEqual({
      positiveText: "Looseleaf Supplement",
      excludedTerms: ["baseball"],
    });
  });
});

describe("Qdrant expanded search plans", () => {
  test("deduplicates only identical type/query pairs", () => {
    expect(dedupeQdrantSearches([
      { type: "lex", query: "same" },
      { type: "vec", query: "same" },
      { type: "lex", query: "same" },
    ])).toEqual([
      { type: "lex", query: "same" },
      { type: "vec", query: "same" },
    ]);
  });

  test.each([
    ["hybrid", hybridQuery],
    ["vector", vectorSearchQuery],
  ] as const)("uses query expansion for %s search", async (_name, search) => {
    vi.stubEnv("QMD_QDRANT_URL", "https://qdrant.test");
    vi.stubEnv("QMD_QDRANT_API_KEY", "test-key");
    vi.stubEnv("QMD_QDRANT_ALLOWED_DOMAINS", "public");

    const expandQuery = vi.fn().mockResolvedValue([
      { type: "lex", query: "expanded keywords" },
      { type: "vec", query: "expanded semantics" },
    ]);
    const store = {
      db: { prepare: vi.fn(() => ({ get: vi.fn() })) },
      expandQuery,
    } as unknown as Store;
    const embedBatch = vi.fn(async (texts: string[]) => texts.map(() => ({
        embedding: [0.1, 0.2],
        model: "test-embed",
      })));
    const llm = { embedBatch } as unknown as LLM;
    const fetchMock = vi.fn(async (_input: string | URL | Request, _init?: RequestInit) => new Response(
      JSON.stringify({ result: { groups: [] } }),
      { status: 200, headers: { "content-type": "application/json" } },
    ));
    vi.stubGlobal("fetch", fetchMock);

    await search(store, "original query", {
      collection: "jersey_city_code",
      limit: 3,
      llm,
    });

    expect(expandQuery).toHaveBeenCalledWith("original query", undefined, undefined, llm);
    expect(embedBatch).toHaveBeenCalled();
    const bodies = fetchMock.mock.calls.map(call => String(call[1]?.body ?? ""));
    if (_name === "hybrid") {
      expect(bodies.some(body => body.includes("expanded keywords"))).toBe(true);
    }
    expect(embedBatch.mock.calls[0]?.[0]).toHaveLength(2);
  });
});
