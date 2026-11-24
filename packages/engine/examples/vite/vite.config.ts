// vite.config.ts (host-side, not this package)
import { defaultClientConditions, defineConfig } from 'vite';

export default defineConfig({
  resolve: {
    conditions: ['onnxruntime-web-use-extern-wasm', ...defaultClientConditions],
  },
  optimizeDeps: {
    exclude: ['onnxruntime-web', '@komugi-ai/engine'],
  },
  worker: {
    format: 'es',
  },
});
