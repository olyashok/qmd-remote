import { createHmac } from "node:crypto";
import { describe, expect, test } from "vitest";
import { ScopedAuthError, verifyScopedSearchToken } from "../src/scoped-auth.js";

const secret = Buffer.from("scoped-test-secret-with-at-least-thirty-two-bytes");
const config = { secret, collections: ["rooms-shape-24-bright"] };
const now = 1_788_000_000;

function token(overrides: Record<string, unknown> = {}, header: Record<string, unknown> = {}) {
  const encodedHeader = Buffer.from(JSON.stringify({ alg: "HS256", typ: "JWT", ...header })).toString("base64url");
  const encodedPayload = Buffer.from(JSON.stringify({
    sub: "user@example.com",
    iss: "cellect-rooms",
    aud: "cellect-qmd-private",
    iat: now,
    exp: now + 90,
    tenant: "shape",
    scopes: ["project:24-bright-street"],
    access: ["documents"],
    mode: "user",
    ...overrides,
  })).toString("base64url");
  const signature = createHmac("sha256", secret)
    .update(`${encodedHeader}.${encodedPayload}`)
    .digest("base64url");
  return `${encodedHeader}.${encodedPayload}.${signature}`;
}

describe("scoped QMD token verification", () => {
  test("accepts the rooms issuer, private audience, and explicit ACL", () => {
    expect(verifyScopedSearchToken(token(), config, now + 1)).toMatchObject({
      tenant: "shape",
      scopes: ["project:24-bright-street"],
      access: ["documents"],
    });
  });

  test.each([
    ["wrong audience", { aud: "cellect-qmd-public" }, {}],
    ["expired", { exp: now }, {}],
    ["overlong", { exp: now + 121 }, {}],
    ["wildcard", { scopes: ["project:*"] }, {}],
    ["wrong algorithm", {}, { alg: "none" }],
  ])("rejects %s tokens", (_name, overrides, header) => {
    expect(() => verifyScopedSearchToken(token(overrides, header), config, now + 1)).toThrow(ScopedAuthError);
  });

  test("rejects a modified signature", () => {
    const value = token();
    const parts = value.split(".");
    parts[2] = `${parts[2]![0] === "A" ? "B" : "A"}${parts[2]!.slice(1)}`;
    expect(() => verifyScopedSearchToken(parts.join("."), config, now + 1)).toThrow("signature");
  });
});
