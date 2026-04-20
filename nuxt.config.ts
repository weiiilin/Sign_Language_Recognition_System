export default defineNuxtConfig({
  future: { compatibilityVersion: 4 },
  compatibilityDate: '2026-04-05',
  modules: ['@vite-pwa/nuxt', '@pinia/nuxt'],
  devServer: {
    host: '0.0.0.0',
    port: 3001,
  },
  vite: {
    server: {
      allowedHosts: true,
    },
    plugins: [
      // require('@vitejs/plugin-basic-ssl')() // 使用 ngrok 時請關閉本地 SSL，交由 ngrok 處理 HTTPS
    ],
    optimizeDeps: {
      include: [
        '@vue/devtools-core',
        '@vue/devtools-kit',
        'comlink',
        '@tensorflow/tfjs',
        '@mediapipe/hands',
      ]
    }
  },
  pwa: {
    registerType: 'autoUpdate',
    manifest: {
      name: 'SignFlow AI',
      short_name: 'SignFlow',
      theme_color: '#4f46e5',
      icons: [
        { src: 'pwa-192x192.png', sizes: '192x192', type: 'image/png' },
        { src: 'pwa-512x512.png', sizes: '512x512', type: 'image/png' }
      ]
    },
    workbox: {
      globPatterns: ['**/*.{js,css,html,png,svg,wasm,onnx,onnx.data}'],

      runtimeCaching: [
        {
          urlPattern: /^https:\/\/cdn\.jsdelivr\.net\/.*/i,
          handler: 'CacheFirst',
          options: {
            cacheName: 'external-resources',
            expiration: { maxEntries: 10, maxAgeSeconds: 60 * 60 * 24 * 30 }
          }
        }
      ]
    }
  }
})