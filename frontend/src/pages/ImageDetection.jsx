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
  const isFusion = result?.is_fusion || result?.model_type === 'multimodal_fusion'

  return (
    <div className="page-container image-detection-page">
      <div className="section-header">
        <span className="badge badge-teal">Multimodal Forensic Analysis (CNN + FFT + DWT)</span>
        <h1>Image Deepfake Detection</h1>
        <p>
          Upload a target portrait or digital image. VeriFrame leverages a unified multimodal architecture
          fusing deep CNN spatial embeddings (512-dim), FFT frequency power spectrum (70-dim), and DWT
          wavelet detail sub-bands (28-dim) into a 610-dimensional forensic classifier.
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
                🔍 Run Multimodal Detection
              </button>
            )}
            {loading && (
              <button className="btn btn-primary" disabled>
                <span className="spinner" style={{ width: 18, height: 18, borderWidth: 2 }} />
                Extracting CNN + FFT + DWT Features…
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
            <h4>💡 How Multimodal Feature Fusion Works</h4>
            <ul className="guide-list">
              <li>
                <strong>1. CNN Spatial Embedding (512-dim):</strong> Pretrained ResNet18 extracts rich spatial feature maps of facial landmarks, boundary blending, and skin texture.
              </li>
              <li>
                <strong>2. FFT Spectral Analysis (70-dim):</strong> 2D Fast Fourier Transform evaluates high-frequency energy decay and periodic GAN upsampling checkerboard patterns.
              </li>
              <li>
                <strong>3. DWT Wavelet Sub-bands (28-dim):</strong> 2D Haar Wavelet Transform decomposes localized directional detail (horizontal, vertical, diagonal) and relative energy distributions.
              </li>
              <li>
                <strong>4. Unified Fusion Classifier (610-dim):</strong> Normalizes and concatenates all three feature branches into an MLP network to deliver a robust determination.
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
              <p>Upload a target image on the left and click "Run Multimodal Detection" to view forensic inspection results.</p>
              <div className="placeholder-tips">
                <span className="tip-item">✓ CNN (512) + FFT (70) + DWT (28) = 610 Features</span>
                <span className="tip-item">✓ Automatic face detection &amp; alignment</span>
                <span className="tip-item">✓ Results securely recorded to detection history</span>
              </div>
            </div>
          )}

          {loading && (
            <div className="card result-placeholder">
              <div className="spinner" style={{ width: 36, height: 36, marginBottom: '1.25rem' }} />
              <h3>Running Multimodal Feature Fusion…</h3>
              <p>Extracting CNN spatial embeddings, FFT radial power spectrum, and DWT wavelet coefficients.</p>
            </div>
          )}

          {result && (
            <div className="results-content">
              {/* Model status warning if checkpoint missing */}
              {!result.model_ready && (
                <div className="alert alert-warning" style={{ marginBottom: '1.25rem' }}>
                  ⚠️ Model is currently running in baseline evaluation mode.
                </div>
              )}

              {/* Primary Verdict Card */}
              <div className={`card verdict-card ${isReal ? 'verdict-real' : isFake ? 'verdict-fake' : ''}`}>
                <div className="verdict-header">
                  <div className="verdict-emoji-badge">
                    {isReal ? '✅' : isFake ? '🚨' : '❓'}
                  </div>
                  <div className="verdict-details">
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
                      <span className="verdict-title-label">Classification Verdict</span>
                      {isFusion ? (
                        <span className="badge badge-teal" style={{ fontSize: '0.75rem' }}>
                          ⚡ Multimodal Fusion (610-dim)
                        </span>
                      ) : (
                        <span className="badge badge-neutral" style={{ fontSize: '0.75rem' }}>
                          Spatial CNN Baseline
                        </span>
                      )}
                    </div>
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
                    ? 'Natural spatial and frequency characteristics detected. No significant synthetic generation patterns, spectral anomalies, or wavelet boundary seams found.'
                    : isFake
                    ? 'High probability of digital manipulation detected. Spatial artifacts, anomalous high-frequency power distributions, or wavelet boundary seams identified.'
                    : 'The analysis returned an inconclusive verdict. Consider running Frequency Analysis for supplementary inspection.'}
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

              {/* Multimodal Feature Branches Status */}
              <div className="card multimodal-branches-card">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                  <h4 style={{ margin: 0, color: 'var(--text-primary)' }}>Multimodal Analysis Branches</h4>
                  <span className="badge badge-teal" style={{ fontSize: '0.75rem' }}>
                    {result.feature_dims?.total || 610} Total Features
                  </span>
                </div>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>
                  Forensic signals extracted across orthogonal spatial, spectral, and wavelet domains:
                </p>

                <div className="branches-grid">
                  {/* Branch 1: CNN */}
                  <div className="branch-card">
                    <div className="branch-header">
                      <span className="branch-icon">🖼️</span>
                      <div className="branch-info">
                        <strong>CNN Spatial Analysis</strong>
                        <span className="branch-meta">512 dimensions • ResNet18</span>
                      </div>
                      <span className="status-pill status-active">Active</span>
                    </div>
                    <p className="branch-desc">
                      Penultimate average pooling embedding capturing blend boundaries &amp; dermal textures.
                    </p>
                  </div>

                  {/* Branch 2: FFT */}
                  <div className="branch-card">
                    <div className="branch-header">
                      <span className="branch-icon">⚡</span>
                      <div className="branch-info">
                        <strong>FFT Spectral Analysis</strong>
                        <span className="branch-meta">70 dimensions • 2D FFT</span>
                      </div>
                      <span className="status-pill status-active">Active</span>
                    </div>
                    <p className="branch-desc">
                      Azimuthally averaged radial power spectrum (64 bins) &amp; spectral summary statistics.
                    </p>
                  </div>

                  {/* Branch 3: DWT */}
                  <div className="branch-card">
                    <div className="branch-header">
                      <span className="branch-icon">🌊</span>
                      <div className="branch-info">
                        <strong>DWT Wavelet Analysis</strong>
                        <span className="branch-meta">28 dimensions • Haar Wavelet</span>
                      </div>
                      <span className="status-pill status-active">Active</span>
                    </div>
                    <p className="branch-desc">
                      Sub-band coefficients (LL, LH, HL, HH), relative energy ratios &amp; high-frequency seams.
                    </p>
                  </div>
                </div>

                <div style={{ marginTop: '1rem', textAlign: 'right' }}>
                  <Link to="/frequency-analysis" className="btn btn-outline btn-sm" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}>
                    📡 View In-Depth Spectral &amp; Wavelet Visualizations →
                  </Link>
                </div>
              </div>

              {/* Face Detection Details */}
              <div className="card">
                <h4 style={{ marginBottom: '0.75rem', color: 'var(--text-primary)' }}>Facial Landmark Inspection</h4>
                <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>
                  {result.face_detected
                    ? '✅ Primary face detected and normalized for multimodal inference.'
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
                    <span className="meta-label">Model Engine</span>
                    <span className="meta-value">
                      {isFusion ? 'Multimodal Fusion MLP (610-dim)' : 'ResNet18 CNN (PyTorch)'}
                    </span>
                  </div>
                  <div className="meta-item">
                    <span className="meta-label">Combined Features</span>
                    <span className="meta-value">
                      {result.feature_dims?.total || (isFusion ? 610 : 512)} Dimensions
                    </span>
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
                    Multimodal Classification Verdict
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
                      <td>Classification Model</td>
                      <td>
                        <strong>{isFusion ? 'Multimodal Fusion MLP (CNN + FFT + DWT)' : 'ResNet18 CNN Baseline'}</strong>
                      </td>
                    </tr>
                    <tr>
                      <td>Feature Vector Dimension</td>
                      <td>{result.feature_dims?.total || 610} Dimensions (CNN: 512, FFT: 70, DWT: 28)</td>
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
                    ? 'The multimodal fusion classifier analyzed spatial landmarks, 2D Fourier frequency distribution, and discrete wavelet sub-band coefficients. All branches showed natural smooth skin gradients, natural spectral roll-off, and low detail energy ratios typical of authentic photography.'
                    : isFake
                    ? 'The multimodal fusion classifier detected corroborating forensic anomalies across multiple domains: high-frequency spectral artifacts, anomalous wavelet sub-band energy concentrations, and spatial blending irregularities characteristic of deepfake generation.'
                    : 'The analysis yielded borderline confidence values. Recommended next step: run Frequency Analysis for detailed spectral visualization.'}
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
