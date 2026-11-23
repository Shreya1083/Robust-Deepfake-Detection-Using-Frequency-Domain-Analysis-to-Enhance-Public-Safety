import axios from 'axios'

const API_BASE = '/api'

export const api = axios.create({
  baseURL: API_BASE,
  withCredentials: true, // Automatically sends and receives HttpOnly cookies
})

// Detection APIs
export const checkHealth   = ()            => api.get('/health')
export const checkStatus   = ()            => api.get('/status')
export const detectImage   = (formData)    => api.post('/detect/image',  formData, { headers: { 'Content-Type': 'multipart/form-data' } })
export const detectVideo   = (formData)    => api.post('/detect/video',  formData, { headers: { 'Content-Type': 'multipart/form-data' } })
export const analyzeFFT    = (formData)    => api.post('/frequency/fft', formData, { headers: { 'Content-Type': 'multipart/form-data' } })

// Authentication APIs
export const apiSignUp     = (userData)    => api.post('/auth/signup', userData)
export const apiLogin      = (credentials) => api.post('/auth/login', credentials)
export const apiLogout     = ()            => api.post('/auth/logout')
export const apiGetMe      = ()            => api.get('/auth/me')
export const apiAuthStatus = ()            => api.get('/auth/status')
