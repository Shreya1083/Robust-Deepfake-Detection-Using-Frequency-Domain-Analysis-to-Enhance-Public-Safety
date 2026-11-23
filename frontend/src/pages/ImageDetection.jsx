import { useState, useCallback } from 'react'
import { Link } from 'react-router-dom'
import { detectImage } from '../services/api'
import './ImageDetection.css'

const ACCEPT = '.jpg,.jpeg,.png'
const MAX_MB = 20

function ProbBar({ label, value, colorVar }) {
  const pct = (value * 100).toFixed(1)
  return (
    <div className="prob-row">
      <div className="prob-label">
        <span className="prob-name">{label}</span>
        <span className="prob-pct">{pct}%</span>
      </div>
      <div className="progress-bar">
        <div
          className="progress-fill"
          style={{ width: `${pct}%`, background: `var(${colorVar})` }}
        />
      </div>
    </div>
  )
}

export default function ImageDetection() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [dragging, setDragging] = useState(false)
  const [showReport, setShowReport] = useState(false)

  const handleFile = (f) => {
    if (!f) return
    if (f.size > MAX_MB * 1024 * 1024) {
      setError(`File exceeds maximum size of ${MAX_MB} MB.`)
      return
    }
    setFile(f)
    setResult(null)
    setError(null)
    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target.result)
    reader.readAsDataURL(f)
  }

  const onInputChange = (e) => handleFile(e.target.files[0])

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }, [])

  const onDragOver = (e) => {
    e.preventDefault()
    setDragging(true)
  }
  const onDragLeave = () => setDragging(false)

  const analyze = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const fd = new FormData()
      fd.append('file', file)
      const { data } = await detectImage(fd)
      setResult(data)
    } catch (err) {
      const respData = err.response?.data
      let errText = 'Image analysis failed.'
      if (typeof respData?.detail === 'string') {
        errText = respData.detail
      } else if (err.message) {
        errText = err.message
      }
      setError(errText)
    } finally {
      setLoading(false)
    }
  }

  const reset = () => {
    setFile(null)
    setPreview(null)
    setResult(null)
    setError(null)
    setShowReport(false)
  }

  const verdict = result?.prediction
  const isReal = verdict === 'REAL'
  const isFake = verdict === 'DEEPFAKE'

  return (
    <div className="page-container image-detection-page">
      <div className="section-header">
        <span className="badge badge-teal">Spatial Forensic Analysis</span>
        <h1>Image Deepfake Detection</h1>
        <p>
          Upload a digital image or portrait. VeriFrame extracts facial landmarks and runs deep
          convolutional neural network inference to detect synthetic generation, face swapping, and manipulation artifacts.
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
                <img src={preview} alt="Inspection target" className="preview-img" />
                <div className="preview-meta">
                  <span className="file-name">{file?.name}</span>
                  <span className="file-size">{(file?.size / 1024).toFixed(1)} KB</span>
                </div>
              </div>
            ) : (
              <div className="drop-placeholder">
                <div className="drop-icon">🖼️</div>
                <h3>Drag &amp; drop your image here</h3>
                <p className="text-muted">or click anywhere to browse files</p>
                <div className="upload-chips">
                  <span>JPG</span>
                  <span>JPEG</span>
                  <span>PNG</span>
                  <span>Max {MAX_MB} MB</span>
                </div>
              </div>
            )}
            <input
              type="file"
              accept={ACCEPT}
              className="file-input"
              onChange={onInputChange}
            />
          </div>

          <div className="upload-actions">
            {file && !loading && (
              <button className="btn btn-primary btn-analyze" onClick={analyze}>
                🔍 Run Deepfake Analysis
              </button>
            )}
            {loading && (
              <button className="btn btn-primary" disabled>
                <span className="spinner" style={{ width: 18, height: 18, borderWidth: 2 }} />
                Extracting Features &amp; Analyzing…
              </button>
            )}
            {(file || result) && (
              <button className="btn btn-outline" onClick={reset}>
                ✕ Reset
              </button>
            )}
          </div>

          {error && (
            <div className="alert alert-danger" style={{ marginTop: '1rem' }}>
              ⚠️ {error}
            </div>
          )}

          {/* Quick Guide Card */}
          <div className="card guide-card">
            <h4>💡 How Spatial Analysis Works</h4>
            <ul className="guide-list">
              <li>
                <strong>Face Localization:</strong> Detects and aligns the bounding box around primary facial structures.
              </li>
              <li>
                <strong>ResNet18 Feature Extraction:</strong> Deep convolutional layers analyze spatial blend boundaries, irregular eye reflections, and dermal texture anomalies.
              </li>
              <li>
                <strong>Probability Scoring:</strong> Computes calibrated probabilities representing genuine vs manipulated likelihood.
              </li>
            </ul>
          </div>
        </div>

        {/* Results Panel */}
        <div className="results-panel">
          {!result && !loading && (
            <div className="card result-placeholder">
              <div className="placeholder-icon">🛡️</div>
              <h3>Awaiting Image Analysis</h3>
              <p>Upload a target image on the left and click "Run Deepfake Analysis" to view forensic inspection results.</p>
              <div className="placeholder-tips">
                <span className="tip-item">✓ High-resolution face crops supported</span>
                <span className="tip-item">✓ Automatic fallback if no face detected</span>
                <span className="tip-item">✓ Results securely logged in your history</span>
              </div>
            </div>
          )}

          {loading && (
            <div className="card result-placeholder">
              <div className="spinner" style={{ width: 36, height: 36, marginBottom: '1.25rem' }} />
              <h3>Running CNN Spatial Inference…</h3>
              <p>Preprocessing image, checking facial landmarks, and evaluating convolutional feature maps.</p>
            </div>
          )}

          {result && (
            <div className="results-content">
              {/* Model status warning if checkpoint missing */}
              {!result.model_ready && (
                <div className="alert alert-warning" style={{ marginBottom: '1.25rem' }}>
                  ⚠️ Model checkpoint is currently running in baseline evaluation mode.
                </div>
              )}

              {/* Primary Verdict Card */}
              <div className={`card verdict-card ${isReal ? 'verdict-real' : isFake ? 'verdict-fake' : ''}`}>
                <div className="verdict-header">
                  <div className="verdict-emoji-badge">
                    {isReal ? '✅' : isFake ? '🚨' : '❓'}
                  </div>
                  <div className="verdict-details">
                    <span className="verdict-title-label">Classification Verdict</span>
                    <div className="verdict-badge-row">
                      <span className={`badge ${isReal ? 'badge-real' : isFake ? 'badge-fake' : 'badge-neutral'}`}>
                        {verdict || 'INCONCLUSIVE'}
                      </span>
                      <span className="confidence-text">
                        Confidence: <strong>{result.confidence != null ? `${(result.confidence * 100).toFixed(1)}%` : '—'}</strong>
                      </span>
                    </div>
                  </div>
                </div>

                <p className="verdict-desc">
                  {isReal
                    ? 'Natural spatial characteristics detected. No significant synthetic generation patterns or facial blending boundaries found.'
                    : isFake
                    ? 'High probability of digital manipulation detected. Spatial artifacts, blending boundaries, or generative synthesis identified.'
                    : 'The analysis returned an inconclusive verdict. Consider running Frequency Analysis for supplementary spectral checks.'}
                </p>
              </div>

              {/* Class Probabilities Card */}
              {result.real_prob != null && (
                <div className="card">
                  <h4 style={{ marginBottom: '1rem', color: 'var(--text-primary)' }}>Calibrated Class Probabilities</h4>
                  <ProbBar label="Authentic (REAL)" value={result.real_prob} colorVar="--success" />
                  <ProbBar label="Manipulated (DEEPFAKE)" value={result.deepfake_prob} colorVar="--danger" />
                </div>
              )}

              {/* Face Detection Details */}
              <div className="card">
                <h4 style={{ marginBottom: '0.75rem', color: 'var(--text-primary)' }}>Facial Landmark Inspection</h4>
                <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>
                  {result.face_detected
                    ? '✅ Primary face detected and normalized for convolutional inference.'
                    : '⚠️ No face detected with primary detector — full frame analyzed as fallback.'}
                </p>
                {result.cropped_face_b64 && (
                  <div className="face-crop-preview">
                    <img
                      src={`data:image/jpeg;base64,${result.cropped_face_b64}`}
                      alt="Cropped face inspection"
                      className="cropped-face-img"
                    />
                    <span className="crop-label">Extracted Face Region (224×224)</span>
                  </div>
                )}
              </div>

              {/* Technical Inspection Metadata */}
              <div className="card meta-card">
                <h4 style={{ marginBottom: '0.75rem', color: 'var(--text-primary)' }}>Technical Specifications</h4>
                <div className="meta-grid">
                  <div className="meta-item">
                    <span className="meta-label">File Name</span>
                    <span className="meta-value">{result.filename || file?.name}</span>
                  </div>
                  <div className="meta-item">
                    <span className="meta-label">Inference Device</span>
                    <span className="meta-value">{result.device || 'CPU'}</span>
                  </div>
                  <div className="meta-item">
                    <span className="meta-label">Neural Architecture</span>
                    <span className="meta-value">ResNet18 (PyTorch)</span>
                  </div>
                  <div className="meta-item">
                    <span className="meta-label">History Tracking</span>
                    <span className="meta-value text-success">
                      {result.history_id ? '✓ Saved to Database' : 'Session Active'}
                    </span>
                  </div>
                </div>
              </div>

              {/* Action Buttons: View Report & Frequency Shortcut */}
              <div className="card action-card">
                <div className="action-buttons-grid">
                  <button className="btn btn-primary" onClick={() => setShowReport(true)}>
                    📄 View Forensic Report
                  </button>
                  <Link to="/frequency-analysis" className="btn btn-outline">
                    📡 Run Frequency Analysis
                  </Link>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Forensic Report Modal */}
      {showReport && result && (
        <div className="modal-backdrop" onClick={() => setShowReport(false)}>
          <div className="modal-card print-report" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div className="modal-brand">
                <span className="modal-brand-icon">🛡️</span>
                <div>
                  <h3>VeriFrame Forensic Inspection Report</h3>
                  <p className="text-muted" style={{ fontSize: '0.8rem' }}>AI-Powered Deepfake Detection System</p>
                </div>
              </div>
              <button className="modal-close" onClick={() => setShowReport(false)}>✕</button>
            </div>

            <div className="modal-body">
              <div className="report-verdict-banner" style={{
                background: isReal ? 'rgba(16, 185, 129, 0.08)' : isFake ? 'rgba(239, 68, 68, 0.08)' : 'rgba(245, 158, 11, 0.08)',
                border: `1px solid ${isReal ? 'var(--success)' : isFake ? 'var(--danger)' : 'var(--warning)'}`,
                padding: '1.25rem',
                borderRadius: '8px',
                marginBottom: '1.5rem',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between'
              }}>
                <div>
                  <span style={{ fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)' }}>
                    Primary Determination
                  </span>
                  <div style={{ fontSize: '1.6rem', fontWeight: 800, color: isReal ? 'var(--success)' : 'var(--danger)' }}>
                    {verdict}
                  </div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Confidence Rating</span>
                  <div style={{ fontSize: '1.4rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                    {result.confidence != null ? `${(result.confidence * 100).toFixed(1)}%` : '—'}
                  </div>
                </div>
              </div>

              <div className="report-section">
                <h4>Inspection Metadata</h4>
                <table className="report-table">
                  <tbody>
                    <tr>
                      <td>File Inspected</td>
                      <td><strong>{result.filename || file?.name}</strong></td>
                    </tr>
                    <tr>
                      <td>Media Format</td>
                      <td>Image ({file?.type || 'Standard Bitmap'})</td>
                    </tr>
                    <tr>
                      <td>File Size</td>
                      <td>{(file?.size ? (file.size / 1024).toFixed(1) + ' KB' : '—')}</td>
                    </tr>
                    <tr>
                      <td>Inspection Date &amp; Time</td>
                      <td>{new Date().toLocaleString()}</td>
                    </tr>
                    <tr>
                      <td>Record ID</td>
                      <td><code>{result.history_id || 'LOCAL-SCAN'}</code></td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <div className="report-section">
                <h4>Quantitative Classification Probabilities</h4>
                <table className="report-table">
                  <thead>
                    <tr>
                      <th>Class Target</th>
                      <th>Calculated Likelihood</th>
                      <th>Risk Evaluation</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>Authentic Content (REAL)</td>
                      <td>{(result.real_prob != null ? (result.real_prob * 100).toFixed(2) : '—')}%</td>
                      <td>
                        <span className={`badge ${result.real_prob > 0.5 ? 'badge-real' : 'badge-neutral'}`}>
                          {result.real_prob > 0.5 ? 'Primary Match' : 'Unmatched'}
                        </span>
                      </td>
                    </tr>
                    <tr>
                      <td>Synthetic / Manipulated (DEEPFAKE)</td>
                      <td>{(result.deepfake_prob != null ? (result.deepfake_prob * 100).toFixed(2) : '—')}%</td>
                      <td>
                        <span className={`badge ${result.deepfake_prob > 0.5 ? 'badge-fake' : 'badge-neutral'}`}>
                          {result.deepfake_prob > 0.5 ? 'Elevated Risk' : 'Low Risk'}
                        </span>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <div className="report-section">
                <h4>Forensic Findings &amp; Summary</h4>
                <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                  {isReal
                    ? 'The neural spatial classifier found consistent dermal gradient distributions, symmetric specular highlights, and natural boundary transitions characteristic of authentic photography. No signs of generative upsampling or deepfake tampering.'
                    : isFake
                    ? 'The neural spatial classifier detected anomalous feature patterns typically induced by generative adversarial networks (GANs), diffusion decoders, or boundary blending artifacts common in facial swap synthesis.'
                    : 'The analysis yielded borderline confidence values. Recommended next step: run Frequency Domain Analysis (FFT) to examine high-frequency spectral energy.'}
                </p>
              </div>
            </div>

            <div className="modal-footer">
              <button className="btn btn-primary" onClick={() => window.print()}>
                🖨️ Print / Save PDF
              </button>
              <button className="btn btn-outline" onClick={() => setShowReport(false)}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
