import { defineConfig } from 'vite';
import { tanstackStart } from '@tanstack/react-start/plugin/vite';
import viteReact from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import { nitroV2Plugin } from '@tanstack/nitro-v2-vite-plugin';

export default defineConfig({
  resolve: { tsconfigPaths: true },
  plugins: [
    nitroV2Plugin({
      preset: 'node-server',
      compatibilityDate: '2025-11-07',
      routeRules: {
        '/ntc/**': { headers: { 'cache-control': 'public, max-age=31536000, immutable' } },
        '/materialx/**': { headers: { 'cache-control': 'public, max-age=31536000, immutable' } },
        '/textures/**': { headers: { 'cache-control': 'public, max-age=31536000, immutable' } },
      },
    }),
    tailwindcss(),
    tanstackStart(),
    viteReact(),
  ],
});
