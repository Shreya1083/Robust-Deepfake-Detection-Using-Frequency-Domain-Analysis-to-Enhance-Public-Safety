import { useState } from 'react'
import { detectVideo } from '../services/api'
import './VideoDetection.css'

export default function VideoDetection() {
  const [file, setFile]       = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult]   = useState(null)
  const [error, setError]     = useState(null)

  const handleFile = e => {
    const f = e.target.files[0]
    if (!f) return
    setFile(f); setResult(null); setError(null)
  }

  const analyze = async () => {
    if (!file) return
    setLoading(true); setError(null); setResult(null)
    try {
      const fd = new FormData()
      fd.append('file', file)
      const { data } = await detectVideo(fd)
      setResult(data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally { setLoading(false) }
  }

  const reset = () => { setFile(null); setResult(null); setError(null) }

  return (
    <div className="page-container">
      <div className="section-header">
        <div className="label">Phase 5 · Coming Soon</div>
        <h1>Video Detection</h1>
        <p style={{ marginTop: '0.5rem' }}>
          Frame-level deepfake detection across video sequences. Aggregated temporal prediction.
        </p>
      </div>

      <div className="alert alert-warning" style={{ marginBottom: '2rem' }}>
        🚧 Video detection will be fully implemented in <strong>Phase 5</strong>.
        Upload is functional but inference is not yet active.
      </div>

      <div className="video-layout">
        <div className="card upload-card">
          <h3>Upload Video</h3>
          <p style={{ marginTop: '0.5rem', marginBottom: '1.25rem' }}>
            Supported formats: MP4 · AVI · MOV
          </p>
          <input type="file" accept=".mp4,.avi,.mov" className="video-file-input" onChange={handleFile} />
          {file && (
            <div className="file-info">
              <span>📹 {file.name}</span>
              <span className="text-muted">{(file.size / (1024*1024)).toFixed(2)} MB</span>
            </div>
          )}
          <div className="upload-actions" style={{ marginTop: '1rem' }}>
            {file && (
              <button className="btn btn-primary" onClick={analyze} disabled={loading}>
                {loading ? 'Sending…' : '🔍 Analyze Video'}
              </button>
            )}
            {file && <button className="btn btn-outline" onClick={reset}>✕ Clear</button>}
          </div>
          {error && <div className="alert alert-danger" style={{ marginTop: '1rem' }}>⚠️ {error}</div>}
        </div>

        <div className="card result-info-card">
          <h3>What to expect in Phase 5</h3>
          <ul className="phase5-list">
            {[
              'Frame extraction with configurable sampling rate',
              'Per-frame face detection and CNN inference',
              'Temporal aggregation of frame-level predictions',
              'Overall video REAL / DEEPFAKE verdict',
              'Frame-level heatmap timeline',
            ].map((item, i) => (
              <li key={i}><span className="text-accent">→</span> {item}</li>
            ))}
          </ul>

          {result && (
            <div className="alert alert-info" style={{ marginTop: '1.5rem' }}>
              {result.error || result.phase}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
