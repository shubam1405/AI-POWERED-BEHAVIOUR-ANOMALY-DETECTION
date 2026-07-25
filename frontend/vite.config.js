import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Proxy config to direct all API requests to the FastAPI backend running on port 8000
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/health': 'http://localhost:8000',
      '/sessions': 'http://localhost:8000',
      '/session': 'http://localhost:8000',
      '/report': 'http://localhost:8000',
      '/summary': 'http://localhost:8000',
      '/recommendations': 'http://localhost:8000',
      '/chat': 'http://localhost:8000',
      '/dashboard': 'http://localhost:8000',
      '/campaigns': 'http://localhost:8000',
      '/campaign': 'http://localhost:8000',
      '/simulate': 'http://localhost:8000',
    }
  }
})
