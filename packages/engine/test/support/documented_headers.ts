// KOM-1201 unit B (design docs/design/kom1201-asset-kit.md section 6 AC3/AC4,
// N8): the single set of header values the static-server smoke test serves
// and the integration guide documents. `DOCUMENTED_HEADERS` drives the smoke
// server; `parseGuideHeaders` re-derives the same shape from the guide's own
// markdown tables, so a test comparing a response against the parsed guide
// (not against this constant) catches a drift between the two.

export interface ExtensionHeaders {
  contentType: string;
  cacheControl: string;
}

export type DocumentedExtension = '.mjs' | '.wasm' | '.onnx' | '.json';

export interface DocumentedHeaders {
  extensions: Record<DocumentedExtension, ExtensionHeaders>;
  coop: string;
  coep: string;
  csp: string;
}

export const DOCUMENTED_HEADERS: DocumentedHeaders = {
  extensions: {
    '.mjs': { contentType: 'text/javascript', cacheControl: 'public, max-age=31536000, immutable' },
    '.wasm': { contentType: 'application/wasm', cacheControl: 'public, max-age=31536000, immutable' },
    '.onnx': { contentType: 'application/octet-stream', cacheControl: 'no-cache' },
    '.json': { contentType: 'application/json', cacheControl: 'no-cache' },
  },
  coop: 'same-origin',
  coep: 'require-corp',
  csp: "script-src 'self' 'wasm-unsafe-eval'; connect-src 'self' blob:",
};

const EXTENSIONS: readonly DocumentedExtension[] = ['.mjs', '.wasm', '.onnx', '.json'];

// A 3-column `| `a` | `b` | `c` |` row, every cell backticked (section 4
// item 1's fixed format).
const ROW3 = /^\|\s*`([^`]*)`\s*\|\s*`([^`]*)`\s*\|\s*`([^`]*)`\s*\|\s*$/;
// A 2-column `| `a` | `b` |` row.
const ROW2 = /^\|\s*`([^`]*)`\s*\|\s*`([^`]*)`\s*\|\s*$/;

// Parses the two fixed tables out of the integration guide's markdown text:
// the `| Extension | Content-Type | Cache-Control |` table, and the
// `| Header | Value |` table carrying COOP, COEP and the example CSP.
// Throws if a required row is missing, rather than returning a partial
// result that would silently pass a comparison.
export function parseGuideHeaders(markdown: string): DocumentedHeaders {
  const extensions = {} as Record<DocumentedExtension, ExtensionHeaders>;
  const headerValues = new Map<string, string>();
  for (const line of markdown.split('\n')) {
    const row3 = ROW3.exec(line);
    if (row3) {
      const [, ext, contentType, cacheControl] = row3;
      if ((EXTENSIONS as readonly string[]).includes(ext as string)) {
        extensions[ext as DocumentedExtension] = { contentType: contentType as string, cacheControl: cacheControl as string };
      }
      continue;
    }
    const row2 = ROW2.exec(line);
    if (row2) {
      const [, header, value] = row2;
      headerValues.set(header as string, value as string);
    }
  }
  for (const ext of EXTENSIONS) {
    if (!extensions[ext]) throw new Error(`guide's header table is missing a row for ${ext}`);
  }
  const coop = headerValues.get('Cross-Origin-Opener-Policy');
  const coep = headerValues.get('Cross-Origin-Embedder-Policy');
  const csp = headerValues.get('Content-Security-Policy');
  if (coop === undefined) throw new Error("guide's header/value table is missing Cross-Origin-Opener-Policy");
  if (coep === undefined) throw new Error("guide's header/value table is missing Cross-Origin-Embedder-Policy");
  if (csp === undefined) throw new Error("guide's header/value table is missing Content-Security-Policy");
  return { extensions, coop, coep, csp };
}
