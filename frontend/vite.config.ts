import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Same-origin in the deployed single container; proxied to the local API in dev.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/health': 'http://127.0.0.1:8040',
      '/workloads': 'http://127.0.0.1:8040',
      '/demand': 'http://127.0.0.1:8040',
      '/findings': 'http://127.0.0.1:8040',
      '/assumptions': 'http://127.0.0.1:8040',
      '/diagnostic': 'http://127.0.0.1:8040',
      '/geoint': 'http://127.0.0.1:8040',
    },
  },
  build: { outDir: 'dist', sourcemap: false },
})
