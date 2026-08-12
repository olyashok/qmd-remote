import { afterEach, describe, expect, test, vi } from "vitest";
import { RemoteLLM } from "../src/llm-remote.js";

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

describe("RemoteLLM generation authorization", () => {
  test("uses the runtime-only generation API key for generation and health", async () => {
    vi.stubEnv("QMD_GENERATE_API_KEY", "runtime-product-key");
    const fetchMock = vi.fn(async (input: string | URL | Request, _init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/health")) {
        return new Response("ok", { status: 200 });
      }
      return new Response(JSON.stringify({
        choices: [{ text: "lex: expanded query" }],
        model: "fast",
      }), { status: 200, headers: { "content-type": "application/json" } });
    });
    vi.stubGlobal("fetch", fetchMock);
    const llm = new RemoteLLM({ generateUrl: "http://generate.test", generateModel: "fast" });

    await expect(llm.generate("expand this")).resolves.toMatchObject({ model: "fast" });
    await llm.checkHealth();

    for (const call of fetchMock.mock.calls) {
      const headers = new Headers(call[1]?.headers);
      expect(headers.get("authorization")).toBe("Bearer runtime-product-key");
    }
    expect(llm.getConfig()).not.toHaveProperty("generateApiKey");
  });
});

describe("RemoteLLM embedding sanitization", () => {
  test("replaces unpaired UTF-16 surrogates before sending a batch", async () => {
    let requestBody = "";
    vi.stubGlobal("fetch", vi.fn(async (_input: string | URL | Request, init?: RequestInit) => {
      requestBody = String(init?.body ?? "");
      return new Response(JSON.stringify({
        data: [{ index: 0, embedding: [0.25] }],
        model: "embeddinggemma",
      }), { status: 200, headers: { "content-type": "application/json" } });
    }));
    const llm = new RemoteLLM({ embedUrl: "http://embed.test" });

    await expect(llm.embedBatch([`before\uDC00after`])).resolves.toHaveLength(1);
    expect(requestBody).toContain("before�after");
    expect(requestBody).not.toContain("\\udc00");
  });
});
