import axios from 'axios'

const API_BASE = '/api'

export const api = axios.create({ baseURL: API_BASE })

export const checkHealth   = ()            => api.get('/health')
export const checkStatus   = ()            => api.get('/status')
export const detectImage   = (formData)    => api.post('/detect/image',  formData, { headers: { 'Content-Type': 'multipart/form-data' } })
export const detectVideo   = (formData)    => api.post('/detect/video',  formData, { headers: { 'Content-Type': 'multipart/form-data' } })
export const analyzeFFT    = (formData)    => api.post('/frequency/fft', formData, { headers: { 'Content-Type': 'multipart/form-data' } })
