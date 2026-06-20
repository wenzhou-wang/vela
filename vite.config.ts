import { defineConfig } from 'vite';
import { resolve } from 'node:path';

export default defineConfig({
  root: '.',
  resolve: {
    alias: {
      vela: resolve(__dirname, 'src/index.ts'),
    },
  },
  server: {
    open: '/examples/gltf-viewer/index.html',
  },
  build: {
    target: 'esnext',
    sourcemap: true,
    rollupOptions: {
      input: {
        viewer: resolve(__dirname, 'examples/gltf-viewer/index.html'),
        sunsetRidge: resolve(__dirname, 'examples/sunset-ridge/index.html'),
      },
    },
  },
  optimizeDeps: {
    esbuildOptions: { target: 'esnext' },
  },
});
