#!/usr/bin/env node
// komugi release CI: the mock site's static file server (KOM-1221 design
// docs/design/kom1221-release-ci.md section 4, section 9 unit 3). Plain
// node:http, no dependencies. Run as:
//   node ci/release/mock-site/server.mjs --root <dir> --port-file <file>
//
// Listens on 127.0.0.1:0 and writes the resolved port (digits only) to
// <file> the moment it is listening (F17/G11a) -- a caller removes any
// stale <file> first, then waits on it instead of guessing the port.

import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';

function parseArgs(argv) {
  let root;
  let portFile;
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--root') {
      root = argv[i + 1];
      i += 1;
    } else if (argv[i] === '--port-file') {
      portFile = argv[i + 1];
      i += 1;
    } else {
      throw new Error(`server.mjs: unknown argument ${argv[i]}`);
    }
  }
  if (root === undefined || portFile === undefined) {
    throw new Error('server.mjs: usage: --root <dir> --port-file <file>');
  }
  return { root: path.resolve(root), portFile: path.resolve(portFile) };
}

// integration.md section 1's table, plus `.js` (esbuild's own bundle
// output -- page.bundle.js/worker.js/crash_worker.js, G3e) and `.html`
// (index.html; Chromium refuses a module script/navigation document not
// served with these content types). `.mjs`/`.wasm` are the only long-lived,
// immutable entries (served from the CLI's versioned ort-<version>/ dir).
const CONTENT_TYPES = {
  '.mjs': 'text/javascript',
  '.wasm': 'application/wasm',
  '.onnx': 'application/octet-stream',
  '.json': 'application/json',
  '.js': 'text/javascript',
  '.html': 'text/html',
};
const IMMUTABLE_EXTENSIONS = new Set(['.mjs', '.wasm']);

// Resolves strictly under `root`; anything that would escape it (`..`
// traversal) is treated the same as a missing file, below.
function resolveUnderRoot(root, urlPath) {
  const decoded = decodeURIComponent(urlPath.split('?')[0].split('#')[0]);
  const resolved = path.normalize(path.join(root, decoded));
  if (resolved !== root && !resolved.startsWith(root + path.sep)) return null;
  return resolved;
}

function main() {
  const { root, portFile } = parseArgs(process.argv.slice(2));

  const server = http.createServer((req, res) => {
    // Section 2's optional pair, set on every response -- including HEAD
    // and 404s -- never conditionally.
    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');

    const urlPath = req.url === '/' ? '/index.html' : (req.url ?? '/index.html');
    const filePath = resolveUnderRoot(root, urlPath);
    const ext = filePath !== null ? path.extname(filePath) : '';
    const contentType = CONTENT_TYPES[ext];

    if (filePath === null || contentType === undefined) {
      res.statusCode = 404;
      res.end();
      return;
    }

    fs.readFile(filePath, (err, data) => {
      if (err) {
        res.statusCode = 404;
        res.end();
        return;
      }
      res.setHeader('Content-Type', contentType);
      res.setHeader('Cache-Control', IMMUTABLE_EXTENSIONS.has(ext) ? 'public, max-age=31536000, immutable' : 'no-cache');
      res.statusCode = 200;
      if (req.method === 'HEAD') {
        res.end();
        return;
      }
      res.end(data);
    });
  });

  server.listen(0, '127.0.0.1', () => {
    const address = server.address();
    fs.writeFileSync(portFile, String(address.port));
  });
}

main();
