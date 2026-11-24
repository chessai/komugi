// KOM-1199 unit U1 (design docs/design/kom1199-client.md section 4/7): the
// client's typed failure shape plus the classify/validate functions that
// build it. Zero value imports from protocol.ts -- `actionSize` comes in as
// a parameter (section 7's "so the module has zero value imports") and the
// `WORKER_ERROR_CODES` members are re-spelled below, exhaustiveness-checked
// against the real `WorkerErrorCode` type instead of importing the array.

import type { WorkerErrorCode, WorkerReplyErr, WorkerReplyOk, WorkerInitReplyOk } from './protocol.ts';

export const CLIENT_ERROR_CODES = ['timeout', 'worker-crash', 'illegal-reply', 'backend-unavailable', 'incompatible-model', 'model-not-found'] as const;
export type ClientErrorCode = (typeof CLIENT_ERROR_CODES)[number];
export type KomugiFailureCode = WorkerErrorCode | ClientErrorCode;

// A missing or extra key here is a compile error against `WorkerErrorCode`,
// so this stays in sync with protocol.ts without a value import of it.
const WORKER_ERROR_CODE_MEMBERS: { [K in WorkerErrorCode]: true } = {
  'not-initialized': true,
  'model-fetch': true,
  'sha-mismatch': true,
  'init-failed': true,
  'threads-timeout': true,
};

function isWorkerErrorCode(code: string): code is WorkerErrorCode {
  return Object.prototype.hasOwnProperty.call(WORKER_ERROR_CODE_MEMBERS, code);
}

export const KOMUGI_FAILURE_CODES: readonly KomugiFailureCode[] = [
  ...(Object.keys(WORKER_ERROR_CODE_MEMBERS) as WorkerErrorCode[]),
  ...CLIENT_ERROR_CODES,
];

export interface KomugiFailure {
  code: KomugiFailureCode;
  stage: 'init' | 'search';
  message: string;
  workerCode?: string;
}

function withWorkerCode(code: KomugiFailureCode, stage: 'init' | 'search', message: string, workerCode: string | undefined): KomugiFailure {
  return workerCode === undefined ? { code, stage, message } : { code, stage, message, workerCode };
}

export function classifyErrorReply(stage: 'init' | 'search', reply: WorkerReplyErr): KomugiFailure {
  const workerCode = reply.code;
  if (workerCode !== undefined && isWorkerErrorCode(workerCode)) {
    return withWorkerCode(workerCode, stage, reply.err, workerCode);
  }
  return withWorkerCode(stage === 'init' ? 'init-failed' : 'backend-unavailable', stage, reply.err, workerCode);
}

function illegalReply(stage: 'init' | 'search', message: string): { ok: false; failure: KomugiFailure } {
  return { ok: false, failure: { code: 'illegal-reply', stage, message } };
}

export function validateSearchReply(
  raw: unknown,
  id: string,
  actionSize: number,
): { ok: true; reply: WorkerReplyOk } | { ok: false; failure: KomugiFailure } {
  if (typeof raw !== 'object' || raw === null) return illegalReply('search', 'search reply is not an object');
  const r = raw as Record<string, unknown>;
  if (r.id !== id) return illegalReply('search', 'search reply id does not match the pending request');
  if (typeof r.san !== 'string' || r.san.length === 0) return illegalReply('search', 'search reply san is not a non-empty string');
  if (typeof r.action !== 'number' || !Number.isInteger(r.action) || r.action < 0 || r.action >= actionSize) {
    return illegalReply('search', 'search reply action is not an integer in range');
  }
  if (typeof r.stats !== 'object' || r.stats === null) return illegalReply('search', 'search reply stats is not an object');
  // `stamp === 'mismatch'` is not a validation failure (review F2, section 4):
  // the sent history did not replay, but the reply is still a valid move.
  return { ok: true, reply: raw as WorkerReplyOk };
}

export function validateInitReply(
  raw: unknown,
  id: string,
): { ok: true; reply: WorkerInitReplyOk } | { ok: false; failure: KomugiFailure } {
  if (typeof raw !== 'object' || raw === null) return illegalReply('init', 'init reply is not an object');
  const r = raw as Record<string, unknown>;
  if (r.id !== id) return illegalReply('init', 'init reply id does not match the pending request');
  if (r.backend !== 'webgpu' && r.backend !== 'wasm') return illegalReply('init', 'init reply backend is not webgpu/wasm');
  if (typeof r.threads !== 'number' || !Number.isInteger(r.threads) || r.threads <= 0) {
    return illegalReply('init', 'init reply threads is not a positive integer');
  }
  if (typeof r.batch !== 'number' || !Number.isInteger(r.batch) || r.batch <= 0) {
    return illegalReply('init', 'init reply batch is not a positive integer');
  }
  if (typeof r.probeRecord !== 'object' || r.probeRecord === null) return illegalReply('init', 'init reply probeRecord is not an object');
  return { ok: true, reply: raw as WorkerInitReplyOk };
}
