import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Hosted at ibiraheel.com/demos/yield-root-cause-lens/. Every asset and data fetch
// goes through import.meta.env.BASE_URL, so the build works under that sub-path.
export default defineConfig({
  base: '/demos/yield-root-cause-lens/',
  plugins: [react()],
})
