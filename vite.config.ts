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
    open: '/examples/index.html',
  },
  build: {
    target: 'esnext',
    sourcemap: true,
    rollupOptions: {
      input: {
        examples: resolve(__dirname, 'examples/index.html'),
        viewer: resolve(__dirname, 'examples/gltf-viewer/index.html'),
        waveField: resolve(__dirname, 'examples/wave-field/index.html'),
        sunsetRidge: resolve(__dirname, 'examples/sunset-ridge/index.html'),
        neonForge: resolve(__dirname, 'examples/neon-forge/index.html'),
        agentLab: resolve(__dirname, 'examples/agent-lab/index.html'),
      },
    },
  },
  optimizeDeps: {
    esbuildOptions: { target: 'esnext' },
  },
});
