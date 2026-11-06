import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import { resolve } from 'node:path';

export default defineConfig({
  resolve: { tsconfigPaths: true },
  plugins: [tailwindcss(), react()],
  define: {
    'import.meta.env.BASE_URL': 'new URL("./", import.meta.url).href',
    'process.env.NODE_ENV': '"production"',
  },
  build: {
    emptyOutDir: false,
    copyPublicDir: false,
    minify: true,
    lib: {
      entry: resolve(import.meta.dirname, 'src/embedded.tsx'),
      formats: ['es'],
      fileName: () => 'embedded.js',
      cssFileName: 'embedded',
    },
    rolldownOptions: { output: { codeSplitting: false } },
  },
});
