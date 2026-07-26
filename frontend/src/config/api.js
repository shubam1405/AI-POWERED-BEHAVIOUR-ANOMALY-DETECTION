/**
 * src/config/api.js
 *
 * Single source of truth for the backend API base URL.
 *
 * In development the Vite dev-server proxy handles relative paths, but in
 * production (Vercel) there is no proxy — every request must include the
 * full backend URL.  Set VITE_API_URL in your Vercel project environment
 * variables to point at the deployed FastAPI service.
 *
 * Example (.env.local or Vercel dashboard):
 *   VITE_API_URL=https://your-fastapi-backend.onrender.com
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || '';

export default API_BASE_URL;
