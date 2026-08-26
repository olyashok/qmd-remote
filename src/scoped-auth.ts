import { createHmac, timingSafeEqual } from "node:crypto";
import { readFileSync } from "node:fs";

export type ScopedSearchClaims = {
  sub: string;
  tenant: string;
  scopes: string[];
  access: string[];
  mode: "user" | "service";
};

export type ScopedSearchConfig = {
  secret: Buffer;
  collections: string[];
};

export class ScopedAuthError extends Error {}

function stringArray(value: unknown, name: string, maximum: number): string[] {
  if (!Array.isArray(value) || value.length < 1 || value.length > maximum) {
    throw new ScopedAuthError(`${name} must be a non-empty array`);
  }
  const values = [...new Set(value)];
  if (values.length !== value.length || values.some(item => typeof item !== "string" || item.length < 1 || item.length > 200)) {
    throw new ScopedAuthError(`${name} contains an invalid value`);
  }
  return values as string[];
}

function decodePart(part: string): unknown {
  try {
    return JSON.parse(Buffer.from(part, "base64url").toString("utf8"));
  } catch {
    throw new ScopedAuthError("Malformed scoped token");
  }
}

export function loadScopedSearchConfig(env: NodeJS.ProcessEnv = process.env): ScopedSearchConfig | null {
  const secretFile = env.QMD_SCOPED_TOKEN_SECRET_FILE?.trim();
  const collections = [...new Set(
    (env.QMD_SCOPED_COLLECTIONS ?? "").split(",").map(value => value.trim()).filter(Boolean),
  )];
  if (!secretFile && collections.length === 0) return null;
  if (!secretFile || collections.length === 0) {
    throw new Error("QMD_SCOPED_TOKEN_SECRET_FILE and QMD_SCOPED_COLLECTIONS must be configured together");
  }
  const secret = Buffer.from(readFileSync(secretFile, "utf8").trim(), "utf8");
  if (secret.length < 32) throw new Error("QMD scoped token secret must be at least 32 bytes");
  if (collections.length > 50 || collections.some(value => value.length > 200)) {
    throw new Error("QMD_SCOPED_COLLECTIONS is invalid");
  }
  return { secret, collections };
}

export function bearerToken(authorization: string | undefined): string {
  const match = authorization?.match(/^Bearer ([^\s]+)$/);
  if (!match) throw new ScopedAuthError("Missing scoped bearer token");
  return match[1]!;
}

export function verifyScopedSearchToken(
  token: string,
  config: ScopedSearchConfig,
  nowSeconds = Math.floor(Date.now() / 1000),
): ScopedSearchClaims {
  const parts = token.split(".");
  if (parts.length !== 3 || parts.some(part => !part)) throw new ScopedAuthError("Malformed scoped token");
  const header = decodePart(parts[0]!) as Record<string, unknown>;
  const payload = decodePart(parts[1]!) as Record<string, unknown>;
  if (header.alg !== "HS256" || (header.typ !== undefined && header.typ !== "JWT")) {
    throw new ScopedAuthError("Unsupported scoped token algorithm");
  }
  const expected = createHmac("sha256", config.secret).update(`${parts[0]}.${parts[1]}`).digest();
  let actual: Buffer;
  try {
    actual = Buffer.from(parts[2]!, "base64url");
  } catch {
    throw new ScopedAuthError("Malformed scoped token signature");
  }
  if (actual.length !== expected.length || !timingSafeEqual(actual, expected)) {
    throw new ScopedAuthError("Invalid scoped token signature");
  }

  const audience = Array.isArray(payload.aud) ? payload.aud : [payload.aud];
  const issuedAt = Number(payload.iat);
  const expiresAt = Number(payload.exp);
  const notBefore = payload.nbf === undefined ? issuedAt : Number(payload.nbf);
  if (payload.iss !== "cellect-rooms" || !audience.includes("cellect-qmd-private")) {
    throw new ScopedAuthError("Invalid scoped token issuer or audience");
  }
  if (
    !Number.isSafeInteger(issuedAt) || !Number.isSafeInteger(expiresAt) || !Number.isSafeInteger(notBefore)
    || issuedAt > nowSeconds + 5 || nowSeconds < notBefore - 5 || expiresAt <= nowSeconds
    || expiresAt - issuedAt > 120 || nowSeconds - issuedAt > 120
  ) {
    throw new ScopedAuthError("Scoped token is expired or outside its allowed lifetime");
  }
  if (typeof payload.sub !== "string" || payload.sub.length < 1 || payload.sub.length > 200) {
    throw new ScopedAuthError("Scoped token subject is invalid");
  }
  if (typeof payload.tenant !== "string" || !/^[a-z0-9][a-z0-9-]{0,62}$/.test(payload.tenant)) {
    throw new ScopedAuthError("Scoped token tenant is invalid");
  }
  const scopes = stringArray(payload.scopes, "scopes", 100);
  if (scopes.some(scope => !/^(project|company|fund|data-room):[^:*?]+$/.test(scope))) {
    throw new ScopedAuthError("Scoped token contains an invalid or wildcard scope");
  }
  const access = stringArray(payload.access, "access", 10);
  if (access.some(value => !["documents", "construction", "money"].includes(value))) {
    throw new ScopedAuthError("Scoped token contains an invalid access class");
  }
  if (payload.mode !== "user" && payload.mode !== "service") {
    throw new ScopedAuthError("Scoped token mode is invalid");
  }
  return {
    sub: payload.sub,
    tenant: payload.tenant,
    scopes,
    access,
    mode: payload.mode,
  };
}
