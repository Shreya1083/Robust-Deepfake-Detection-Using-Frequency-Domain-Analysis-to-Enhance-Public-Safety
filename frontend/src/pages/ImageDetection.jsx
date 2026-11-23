import { useState, useCallback } from 'react'
import { detectImage } from '../services/api'
import './ImageDetection.css'

const ACCEPT = '.jpg,.jpeg,.png'
const MAX_MB = 20

function ProbBar({ label, value, colorVar }) {
  return (
    <div className="prob-row">
      <div className="prob-label">
        <span>{label}</span>
        <span className="prob-pct">{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="progress-bar">
        <div
          className="progress-fill"
          style={{ width: `${value * 100}%`, background: `var(${colorVar})` }}
        />
      </div>
    </div>
  )
}

export default function ImageDetection() {
  const [file, setFile]       = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult]   = useState(null)
  const [error, setError]     = useState(null)
  const [dragging, setDragging] = useState(false)

  const handleFile = (f) => {
    if (!f) return
    if (f.size > MAX_MB * 1024 * 1024) { setError(`File exceeds ${MAX_MB} MB.`); return }
    setFile(f)
    setResult(null)
    setError(null)
    const reader = new FileReader()
    reader.onload = e => setPreview(e.target.result)
    reader.readAsDataURL(f)
  }

  const onInputChange = e => handleFile(e.target.files[0])

  const onDrop = useCallback(e => {
    e.preventDefault(); setDragging(false)
    handleFile(e.dataTransfer.files[0])
  }, [])
  const onDragOver = e => { e.preventDefault(); setDragging(true) }
  const onDragLeave = () => setDragging(false)

  const analyze = async () => {
    if (!file) return
    setLoading(true); setError(null); setResult(null)
    try {
      const fd = new FormData()
      fd.append('file', file)
      const { data } = await detectImage(fd)
      setResult(data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Request failed.')
    } finally {
      setLoading(false)
    }
  }

  const reset = () => { setFile(null); setPreview(null); setResult(null); setError(null) }

  const verdict = result?.prediction
  const isReal  = verdict === 'REAL'
  const isFake  = verdict === 'DEEPFAKE'

  return (
    <div className="page-container">
      <div className="section-header">
        <div className="label">Phase 3 · CNN Spatial Detection</div>
        <h1>Image Detection</h1>
        <p style={{ marginTop: '0.5rem' }}>
          Upload a face image. The ResNet18 CNN predicts whether it is REAL or a DEEPFAKE.
        </p>
      </div>

      <div className="detection-layout">
        {/* Upload Panel */}
        <div className="upload-panel">
          <div
            className={`drop-zone card ${dragging ? 'dragging' : ''} ${preview ? 'has-preview' : ''}`}
            onDrop={onDrop}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
          >
            {preview ? (
              <div className="preview-wrapper">
                <img src={preview} alt="Preview" className="preview-img" />
                <div className="preview-meta">
                  <span className="text-muted">{file?.name}</span>
                  <span className="text-muted">{(file?.size / 1024).toFixed(1)} KB</span>
                </div>
              </div>
            ) : (
              <div className="drop-placeholder">
                <div className="drop-icon">🖼️</div>
                <p>Drag &amp; drop an image here</p>
                <p className="text-muted" style={{ fontSize: '0.85rem' }}>or click to browse</p>
                <p className="text-muted" style={{ fontSize: '0.8rem', marginTop: '0.25rem' }}>JPG · JPEG · PNG · max {MAX_MB} MB</p>
              </div>
            )}
            <input
              type="file" accept={ACCEPT}
              className="file-input"
              onChange={onInputChange}
            />
          </div>

          <div className="upload-actions">
            {file && !loading && (
              <button className="btn btn-primary" onClick={analyze}>
                🔍 Analyze Image
              </button>
            )}
            {loading && (
              <button className="btn btn-primary" disabled>
                <span className="spinner" style={{ width: 18, height: 18, borderWidth: 2 }} />
                Analyzing…
              </button>
            )}
            {(file || result) && (
              <button className="btn btn-outline" onClick={reset}>✕ Clear</button>
            )}
          </div>

          {error && (
            <div className="alert alert-danger" style={{ marginTop: '1rem' }}>
              ⚠️ {error}
            </div>
          )}
        </div>

        {/* Results Panel */}
        <div className="results-panel">
          {!result && !loading && (
            <div className="card result-placeholder">
              <div className="placeholder-icon">🛡️</div>
              <h3>Results will appear here</h3>
              <p>Upload an image and click Analyze Image.</p>
            </div>
          )}

          {loading && (
            <div className="card result-placeholder">
              <div className="spinner" style={{ marginBottom: '1rem' }} />
              <h3>Running CNN Inference…</h3>
              <p>Detecting face, preprocessing and running ResNet18.</p>
            </div>
          )}

          {result && (
            <div className="results-content">
              {/* Model not ready */}
              {!result.model_ready && (
                <div className="alert alert-warning" style={{ marginBottom: '1rem' }}>
                  ⚠️ Model checkpoint not found. Train the model first. The result below is a placeholder.
                </div>
              )}

              {/* Verdict card */}
              <div className={`card verdict-card ${isReal ? 'verdict-real' : isFake ? 'verdict-fake' : ''}`}>
                <div className="verdict-emoji">{isReal ? '✅' : isFake ? '🚨' : '❓'}</div>
                <div>
                  <div className={`badge ${isReal ? 'badge-real' : isFake ? 'badge-fake' : 'badge-warning'}`} style={{ fontSize: '1rem', padding: '0.4rem 1.2rem' }}>
                    {verdict || 'No prediction'}
                  </div>
                  <div className="confidence-line">
                    Confidence: <strong>{result.confidence != null ? `${(result.confidence * 100).toFixed(1)}%` : '—'}</strong>
                  </div>
                </div>
              </div>

              {/* Probabilities */}
              {result.real_prob != null && (
                <div className="card">
                  <h4 style={{ marginBottom: '1rem' }}>Class Probabilities</h4>
                  <ProbBar label="✅ REAL"     value={result.real_prob}     colorVar="--success" />
                  <ProbBar label="🚨 DEEPFAKE" value={result.deepfake_prob} colorVar="--danger"  />
                </div>
              )}

              {/* Face detection */}
              <div className="card">
                <h4>Face Detection</h4>
                <p style={{ marginTop: '0.5rem' }}>
                  {result.face_detected
                    ? '✅ Face detected and used for inference.'
                    : '⚠️ No face detected — full image used as fallback.'}
                </p>
                {result.cropped_face_b64 && (
                  <img
                    src={`data:image/jpeg;base64,${result.cropped_face_b64}`}
                    alt="Detected face"
                    className="cropped-face"
                  />
                )}
              </div>

              {/* Meta */}
              <div className="card meta-card">
                <div className="meta-row"><span>File</span><span>{result.filename}</span></div>
                <div className="meta-row"><span>Device</span><span>{result.device || '—'}</span></div>
                <div className="meta-row"><span>Model</span><span>ResNet18 (Phase 3)</span></div>
              </div>

              {/* FFT/DWT placeholders */}
              <div className="card placeholder-future">
                <h4>📡 Frequency Analysis <span className="badge badge-warning">Phase 4</span></h4>
                <p>FFT and DWT visualisations will appear here after Phase 4 is implemented.</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
