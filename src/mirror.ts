/**
 * mirror.ts - Remote index mirroring for QMD
 *
 * Allows maintaining a fast local copy of a remote .qmd index (e.g. on a
 * GPU server) while tracking the source and enabling incremental re-sync.
 *
 * Usage:
 *   qmd mirror <ssh-source> [local-path]   # set up + initial sync
 *   qmd mirror sync                         # re-sync from recorded source
 *   qmd mirror status                       # show source, last sync, staleness
 *   qmd mirror clear                        # remove tracking (keep local copy)
 */

import { join } from "path";
import { existsSync, readFileSync, writeFileSync, unlinkSync } from "fs";

// =============================================================================
// Types
// =============================================================================

export type MirrorConfig = {
  source: string;       // e.g. "root@192.168.5.163:/root/.openclaw/agents/shape-main/.qmd"
  localPath: string;    // absolute path to local .qmd dir
  lastSync: string | null;  // ISO 8601 timestamp
};

// =============================================================================
// Config storage (inside the local .qmd dir)
// =============================================================================

const MIRROR_FILE = ".mirror.json";

export function getMirrorConfigPath(qmdDir: string): string {
  return join(qmdDir, MIRROR_FILE);
}

export function loadMirrorConfig(qmdDir: string): MirrorConfig | null {
  const path = getMirrorConfigPath(qmdDir);
  if (!existsSync(path)) return null;
  try {
    return JSON.parse(readFileSync(path, "utf-8")) as MirrorConfig;
  } catch {
    return null;
  }
}

export function saveMirrorConfig(qmdDir: string, config: MirrorConfig): void {
  writeFileSync(getMirrorConfigPath(qmdDir), JSON.stringify(config, null, 2));
}

export function clearMirrorConfig(qmdDir: string): void {
  const path = getMirrorConfigPath(qmdDir);
  if (existsSync(path)) unlinkSync(path);
}

// =============================================================================
// Staleness check
// =============================================================================

/**
 * Returns true if the mirror has never synced or was last synced more than
 * maxAgeHours ago (default: 23h, so a daily-changing index stays fresh).
 */
export function isMirrorStale(config: MirrorConfig, maxAgeHours = 23): boolean {
  if (!config.lastSync) return true;
  const ageMs = Date.now() - new Date(config.lastSync).getTime();
  return ageMs > maxAgeHours * 3600 * 1000;
}

export function mirrorAgeString(config: MirrorConfig): string {
  if (!config.lastSync) return "never synced";
  const ageMs = Date.now() - new Date(config.lastSync).getTime();
  const mins = Math.floor(ageMs / 60000);
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

// =============================================================================
// Sync
// =============================================================================

export type SyncOptions = {
  onProgress?: (line: string) => void;
};

/**
 * rsync the remote source into qmdDir.
 * Uses --no-whole-file so rsync uses block-level diffs (fast for large SQLite
 * files where only the WAL changes). Excludes backup files.
 */
export async function syncMirror(
  qmdDir: string,
  config: MirrorConfig,
  opts: SyncOptions = {}
): Promise<void> {
  // Ensure trailing slash on source so rsync copies contents, not the dir itself
  const source = config.source.endsWith("/") ? config.source : config.source + "/";
  const dest = qmdDir.endsWith("/") ? qmdDir : qmdDir + "/";

  const proc = Bun.spawn(
    [
      "rsync",
      "-az",
      "--no-whole-file",          // use block diffs (essential for large SQLite)
      "--inplace",                  // update files in-place (also helps rsync diffs)
      "--exclude=*.backup-*",       // skip large backup snapshots
      "--exclude=.mirror.json",     // don't overwrite our tracking file
      source,
      dest,
    ],
    {
      stdout: "pipe",
      stderr: "pipe",
    }
  );

  // Stream stderr (rsync progress/errors) to caller
  if (opts.onProgress && proc.stderr) {
    const reader = proc.stderr.getReader();
    const decoder = new TextDecoder();
    (async () => {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        opts.onProgress!(decoder.decode(value));
      }
    })();
  }

  const exitCode = await proc.exited;
  if (exitCode !== 0) {
    const errText = proc.stderr ? await new Response(proc.stderr).text() : "";
    throw new Error(`rsync failed (exit ${exitCode}): ${errText}`);
  }

  // Record successful sync time
  config.lastSync = new Date().toISOString();
  saveMirrorConfig(qmdDir, config);
}
