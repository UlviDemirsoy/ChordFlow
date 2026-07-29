import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { datasetWriterPlugin } from './vite-plugins/datasetWriter.ts'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), datasetWriterPlugin()],
})
