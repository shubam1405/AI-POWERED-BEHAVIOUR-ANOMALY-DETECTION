/**
 * src/config/axiosClient.js
 *
 * Pre-configured Axios instance with the correct baseURL for the environment.
 *
 * Import this instead of importing axios directly so every request
 * automatically targets the right backend regardless of environment.
 *
 * Usage:
 *   import api from '../config/axiosClient';
 *   const res = await api.get('/health');
 */

import axios from 'axios';
import API_BASE_URL from './api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export default api;
