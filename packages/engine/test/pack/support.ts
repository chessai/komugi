// Shared helpers for the npm-pack tests (AC2's test/pack/pack.test.ts,
// design doc section 7.1/9 unit 6), which needs a real `npm pack` of a
// built packages/engine -- which is why this lives in test/pack/ rather
// than alongside the rest of test/ (the full-regression glob only covers
// test/*.test.ts).
//
// This file carries only the pack/build helpers; the scan of the packed
// tarball's contents against pack_denylist.json lives in
// pack_contents.test.ts, which is the only caller that needs it.

import { execFileSync } from 'node:child_process';
import { existsSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

export interface NpmPackFileEntry {
  path: string;
}

export interface NpmPackResult {
  filename: string;
  files: NpmPackFileEntry[];
}

export function requireNpmOnPath(): void {
  try {
    execFileSync('npm', ['--version'], { stdio: 'ignore' });
  } catch (error) {
    throw new Error(`npm is not on PATH (needed to run 'npm pack'): ${(error as Error).message}`);
  }
}

export function requireDistBuilt(engineDir: string): void {
  const indexJs = join(engineDir, 'dist', 'index.js');
  if (!existsSync(indexJs)) {
    throw new Error(
      `${indexJs} is missing -- build packages/engine first ` +
        `(tsc -p packages/engine/tsconfig.json) before running the pack tests`,
    );
  }
}

export interface PackHandle {
  result: NpmPackResult;
  destDir: string;
  tarballPath: string;
  cleanup: () => void;
}

// Runs a real `npm pack --json` of `engineDir` into a fresh temp directory,
// with a fresh npm cache so the run never touches the user's real cache.
// Caller must call `cleanup()` (e.g. in a `finally`).
export function npmPack(engineDir: string): PackHandle {
  requireNpmOnPath();
  requireDistBuilt(engineDir);

  const cacheDir = mkdtempSync(join(tmpdir(), 'komugi-npm-cache-'));
  const destDir = mkdtempSync(join(tmpdir(), 'komugi-pack-dest-'));
  const cleanup = (): void => {
    rmSync(cacheDir, { recursive: true, force: true });
    rmSync(destDir, { recursive: true, force: true });
  };

  try {
    const stdout = execFileSync(
      'npm',
      ['pack', '--json', '--pack-destination', destDir],
      {
        cwd: engineDir,
        env: { ...process.env, npm_config_cache: cacheDir },
        encoding: 'utf8',
      },
    );
    const parsed = JSON.parse(stdout) as NpmPackResult[];
    const [result] = parsed;
    if (!result) throw new Error("npm pack --json produced no entries");
    return { result, destDir, tarballPath: join(destDir, result.filename), cleanup };
  } catch (error) {
    cleanup();
    throw error;
  }
}
