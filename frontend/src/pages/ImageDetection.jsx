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

function RadialChart({ spectrum }) {
  if (!spectrum || spectrum.length === 0) return null
  const W = 360, H = 130, N = spectrum.length
  const pts = spectrum.map((v, i) => {
    const x = (i / (N - 1)) * W
    const y = H - v * (H - 15) - 8
    return `${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')

  return (
    <div className="radial-chart-container">
      <div className="chart-header">
        <span className="chart-title">Azimuthally Averaged Radial Power Spectrum (64 Frequency Bins)</span>
        <span className="chart-subtitle">Energy decay from DC Center (0) to Nyquist Frequency (64)</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" height="130" className="chart-svg">
        <defs>
          <linearGradient id="radialSpecGrad" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#10b981" />
            <stop offset="35%" stopColor="#0d9488" />
            <stop offset="100%" stopColor="#ef4444" />
          </linearGradient>
        </defs>
        {/* Horizontal grid lines */}
        {[0.25, 0.5, 0.75].map(v => (
          <line
            key={v}
            x1="0"
            y1={H - v * (H - 15) - 8}
            x2={W}
            y2={H - v * (H - 15) - 8}
            stroke="var(--border)"
            strokeWidth="1"
            strokeDasharray="4,4"
          />
        ))}
        {/* Spectral energy curve */}
        <polyline points={pts} fill="none" stroke="url(#radialSpecGrad)" strokeWidth="2.5" />
        {/* Band indicators */}
        <line x1={W * 0.10} y1="0" x2={W * 0.10} y2={H} stroke="var(--border)" strokeWidth="1" strokeDasharray="2,3" />
        <line x1={W * 0.40} y1="0" x2={W * 0.40} y2={H} stroke="var(--border)" strokeWidth="1" strokeDasharray="2,3" />
        <text x="6" y={H - 4} fontSize="9" fill="var(--success)" fontWeight="700">LOW (0-10%)</text>
        <text x={W * 0.15} y={H - 4} fontSize="9" fill="var(--accent)" fontWeight="700">MID (10-40%)</text>
        <text x={W * 0.55} y={H - 4} fontSize="9" fill="var(--danger)" fontWeight="700">HIGH (40-100%)</text>
      </svg>
      <div className="chart-legend-row">
        <span className="text-success">← Low Freq (Global Illumination)</span>
        <span className="text-accent">Mid Freq (Facial Texture)</span>
        <span className="text-danger">High Freq (Edges &amp; GAN Grids) →</span>
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
  const [zoomImage, setZoomImage] = useState(null)

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
    setZoomImage(null)
  }

  const verdict = result?.prediction
  const isReal = verdict === 'REAL'
  const isFake = verdict === 'DEEPFAKE'

  const spatial = result?.spatial_analysis
  const freq = result?.frequency_analysis
  const dwt = result?.dwt_analysis
  const summary = result?.analysis_summary

  return (
    <div className="page-container image-detection-page">
      <div className="section-header">
        <span className="badge badge-teal">Forensic Multimodal Inspection</span>
        <h1>Image Deepfake Detection</h1>
        <p>
          Upload an image to perform deep spatial feature analysis with convolutional Grad-CAM heatmaps,
          2D Fast Fourier Transform (FFT) frequency analysis, and Discrete Wavelet Transform (DWT) multi-scale inspection.
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
                🔍 Run Forensic Analysis
              </button>
            )}
            {loading && (
              <button className="btn btn-primary" disabled>
                <span className="spinner" style={{ width: 18, height: 18, borderWidth: 2 }} />
                Extracting Spatial &amp; Frequency Signals…
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
            <h4>💡 How VeriFrame Analyzes Media</h4>
            <ul className="guide-list">
              <li>
                <strong>Spatial Feature Analysis (Grad-CAM):</strong> ResNet18 convolutional layer gradients highlight the anatomical face regions most influential to the model's prediction.
              </li>
              <li>
                <strong>FFT Frequency Analysis:</strong> 2D Fast Fourier Transform uncovers high-frequency energy anomalies and synthetic GAN upsampling patterns.
              </li>
              <li>
                <strong>DWT Wavelet Decomposition:</strong> 2D Haar wavelets isolate horizontal, vertical, and diagonal detail boundaries to inspect boundary blending seams.
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
              <p>Upload a target image on the left and click "Run Forensic Analysis" to view complete spatial and frequency results.</p>
              <div className="placeholder-tips">
                <span className="tip-item">✓ Spatial ResNet18 CNN inference with layer4 Grad-CAM</span>
                <span className="tip-item">✓ 2D FFT magnitude spectrum &amp; radial frequency profile</span>
                <span className="tip-item">✓ 2D Haar DWT multi-scale subband analysis</span>
                <span className="tip-item">✓ Results securely recorded to detection history</span>
              </div>
            </div>
          )}

          {loading && (
            <div className="card result-placeholder">
              <div className="spinner" style={{ width: 36, height: 36, marginBottom: '1.25rem' }} />
              <h3>Running Forensic Pipeline…</h3>
              <p>Detecting facial landmarks, computing ResNet18 layer4 Grad-CAM gradients, calculating 2D FFT spectrum, and evaluating 2D DWT subbands.</p>
            </div>
          )}

          {result && (
            <div className="results-content">
              {/* SECTION A: FINAL DETECTION RESULT */}
              <div className={`card verdict-card ${isReal ? 'verdict-real' : isFake ? 'verdict-fake' : ''}`}>
                <div className="verdict-header">
                  <div className="verdict-emoji-badge">
                    {isReal ? '✅' : isFake ? '🚨' : '❓'}
                  </div>
                  <div className="verdict-details">
                    <span className="verdict-title-label">Primary Determination</span>
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

                {/* Class Probabilities Bar */}
                {result.real_prob != null && (
                  <div className="verdict-probs-wrap">
                    <ProbBar label="Authentic (REAL)" value={result.real_prob} colorVar="--success" />
                    <ProbBar label="Manipulated (DEEPFAKE)" value={result.deepfake_prob} colorVar="--danger" />
                  </div>
                )}
              </div>

              {/* SECTION B: SPATIAL FEATURE ANALYSIS (CNN – GRAD-CAM) */}
              {spatial && spatial.status === 'active' && (
                <div className="card forensic-section-card">
                  <div className="section-card-header">
                    <div>
                      <span className="badge badge-teal">Spatial Feature Analysis (CNN – Grad-CAM)</span>
                      <h3>CNN Spatial Activation &amp; Saliency</h3>
                    </div>
                    <span className="badge badge-neutral">{spatial.architecture || 'ResNet18'}</span>
                  </div>

                  <p className="section-desc">
                    {spatial.explanation || 'The CNN spatial analysis highlights facial regions that contributed most to the classification.'}
                  </p>

                  {/* 3-View Spatial Visualizations Grid */}
                  <div className="visuals-trio-grid">
                    <div className="visual-col" onClick={() => result.cropped_face_b64 && setZoomImage({ src: `data:image/jpeg;base64,${result.cropped_face_b64}`, title: 'Preprocessed Face Input' })}>
                      <span className="visual-caption">1. Input Face Crop</span>
                      {result.cropped_face_b64 ? (
                        <img
                          src={`data:image/jpeg;base64,${result.cropped_face_b64}`}
                          alt="Input Face Crop"
                          className="forensic-thumb"
                        />
                      ) : (
                        <div className="thumb-placeholder">Face Not Cropped</div>
                      )}
                      <span className="click-to-zoom">🔍 Click to enlarge</span>
                    </div>

                    <div className="visual-col" onClick={() => spatial.heatmap_b64 && setZoomImage({ src: `data:image/jpeg;base64,${spatial.heatmap_b64}`, title: 'Grad-CAM Activation Heatmap' })}>
                      <span className="visual-caption">2. Grad-CAM Heatmap</span>
                      {spatial.heatmap_b64 ? (
                        <img
                          src={`data:image/jpeg;base64,${spatial.heatmap_b64}`}
                          alt="Grad-CAM Heatmap"
                          className="forensic-thumb"
                        />
                      ) : (
                        <div className="thumb-placeholder">Heatmap Unavailable</div>
                      )}
                      <span className="click-to-zoom">🔍 Click to enlarge</span>
                    </div>

                    <div className="visual-col" onClick={() => spatial.overlay_b64 && setZoomImage({ src: `data:image/jpeg;base64,${spatial.overlay_b64}`, title: 'Grad-CAM Saliency Overlay' })}>
                      <span className="visual-caption">3. Spatial Overlay</span>
                      {spatial.overlay_b64 ? (
                        <img
                          src={`data:image/jpeg;base64,${spatial.overlay_b64}`}
                          alt="Heatmap Overlay"
                          className="forensic-thumb"
                        />
                      ) : (
                        <div className="thumb-placeholder">Overlay Unavailable</div>
                      )}
                      <span className="click-to-zoom">🔍 Click to enlarge</span>
                    </div>
                  </div>

                  {/* Spatial Technical Specs */}
                  <div className="spatial-specs-grid">
                    <div className="spec-item">
                      <span className="spec-label">Feature Extraction Layer</span>
                      <span className="spec-val">{spatial.target_layer || 'layer4 (Conv2d: 512 channels, 7×7)'}</span>
                    </div>
                    <div className="spec-item">
                      <span className="spec-label">Extraction Status</span>
                      <span className="spec-val text-success">● Active (Computed from Gradients)</span>
                    </div>
                    <div className="spec-item">
                      <span className="spec-label">Influential Salient Regions</span>
                      <span className="spec-val text-accent">{spatial.salient_regions || 'Facial contours'}</span>
                    </div>
                  </div>
                </div>
              )}

              {/* SECTION C: FFT FREQUENCY ANALYSIS */}
              {freq && freq.status === 'active' && (
                <div className="card forensic-section-card">
                  <div className="section-card-header">
                    <div>
                      <span className="badge badge-teal">Frequency Domain Analysis (2D FFT)</span>
                      <h3>Spectral Magnitude &amp; Radial Distribution</h3>
                    </div>
                    <span className="badge badge-neutral">2D Fast Fourier Transform</span>
                  </div>

                  <p className="section-desc">
                    {freq.explanation || 'Frequency analysis examines image patterns and frequency distributions that may reveal inconsistencies associated with image manipulation.'}
                  </p>

                  {/* FFT Visualizations */}
                  <div className="visuals-trio-grid">
                    <div className="visual-col" onClick={() => result.cropped_face_b64 && setZoomImage({ src: `data:image/jpeg;base64,${result.cropped_face_b64}`, title: 'Grayscale Spatial Input' })}>
                      <span className="visual-caption">1. Grayscale Spatial Input</span>
                      {result.cropped_face_b64 && (
                        <img src={`data:image/jpeg;base64,${result.cropped_face_b64}`} alt="Grayscale Input" className="forensic-thumb" />
                      )}
                      <span className="click-to-zoom">🔍 Click to enlarge</span>
                    </div>

                    <div className="visual-col" onClick={() => freq.magnitude_b64 && setZoomImage({ src: `data:image/jpeg;base64,${freq.magnitude_b64}`, title: 'Log-Magnitude FFT Spectrum (BONE)' })}>
                      <span className="visual-caption">2. Magnitude Spectrum (Log)</span>
                      {freq.magnitude_b64 && (
                        <img src={`data:image/jpeg;base64,${freq.magnitude_b64}`} alt="Magnitude Spectrum" className="forensic-thumb" />
                      )}
                      <span className="click-to-zoom">🔍 Click to enlarge</span>
                    </div>

                    <div className="visual-col" onClick={() => freq.log_spectrum_b64 && setZoomImage({ src: `data:image/jpeg;base64,${freq.log_spectrum_b64}`, title: 'Colorized Power Spectrum (JET)' })}>
                      <span className="visual-caption">3. Power Distribution (JET)</span>
                      {freq.log_spectrum_b64 && (
                        <img src={`data:image/jpeg;base64,${freq.log_spectrum_b64}`} alt="Colorized Spectrum" className="forensic-thumb" />
                      )}
                      <span className="click-to-zoom">🔍 Click to enlarge</span>
                    </div>
                  </div>

                  {/* Computed FFT Metrics */}
                  {freq.metrics && (
                    <div className="fft-metrics-container">
                      <div className="metrics-chip-row">
                        <div className="metric-chip">
                          <span className="m-label">Low-Freq Energy (0–10%)</span>
                          <span className="m-val text-success">{freq.metrics.low_energy_pct}%</span>
                          <span className="m-sub">Global illumination</span>
                        </div>
                        <div className="metric-chip">
                          <span className="m-label">Mid-Freq Energy (10–40%)</span>
                          <span className="m-val text-accent">{freq.metrics.mid_energy_pct}%</span>
                          <span className="m-sub">Facial texture</span>
                        </div>
                        <div className="metric-chip">
                          <span className="m-label">High-Freq Energy (40–100%)</span>
                          <span className="m-val text-danger">{freq.metrics.high_energy_pct}%</span>
                          <span className="m-sub">Edges &amp; GAN artifacts</span>
                        </div>
                        <div className="metric-chip">
                          <span className="m-label">High Energy Ratio</span>
                          <span className="m-val">{freq.metrics.high_energy_ratio}</span>
                          <span className="m-sub">Relative decay ratio</span>
                        </div>
                      </div>

                      {/* Radial Power Profile Chart */}
                      {freq.metrics.radial_spectrum && (
                        <RadialChart spectrum={freq.metrics.radial_spectrum} />
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* SECTION D: DWT WAVELET ANALYSIS */}
              {dwt && dwt.status === 'active' && (
                <div className="card forensic-section-card">
                  <div className="section-card-header">
                    <div>
                      <span className="badge badge-teal">Discrete Wavelet Transform (2D Haar DWT)</span>
                      <h3>Multi-Scale Directional Subband Decomposition</h3>
                    </div>
                    <span className="badge badge-neutral">Haar Wavelet</span>
                  </div>

                  <p className="section-desc">
                    {dwt.explanation || '2D DWT separates spatial information into multi-scale approximation and directional detail coefficients.'}
                  </p>

                  <div className="dwt-layout-row">
                    {dwt.subbands_b64 && (
                      <div className="dwt-visual-wrap" onClick={() => setZoomImage({ src: `data:image/jpeg;base64,${dwt.subbands_b64}`, title: '2D Haar DWT Subbands (LL, LH, HL, HH)' })}>
                        <img src={`data:image/jpeg;base64,${dwt.subbands_b64}`} alt="DWT Subbands 2x2" className="dwt-subbands-thumb" />
                        <span className="click-to-zoom">🔍 Click to enlarge (LL · LH · HL · HH)</span>
                      </div>
                    )}

                    {dwt.features && (
                      <div className="dwt-metrics-table-wrap">
                        <table className="dwt-table">
                          <thead>
                            <tr>
                              <th>Subband</th>
                              <th>Component Type</th>
                              <th>Energy Ratio</th>
                            </tr>
                          </thead>
                          <tbody>
                            <tr>
                              <td><strong>LL</strong></td>
                              <td>Low-Low (Approximation)</td>
                              <td><span className="text-success">{(dwt.features.ll.energy_ratio * 100).toFixed(2)}%</span></td>
                            </tr>
                            <tr>
                              <td><strong>LH</strong></td>
                              <td>Low-High (Horizontal Details)</td>
                              <td><span className="text-accent">{(dwt.features.lh.energy_ratio * 100).toFixed(2)}%</span></td>
                            </tr>
                            <tr>
                              <td><strong>HL</strong></td>
                              <td>High-Low (Vertical Details)</td>
                              <td><span className="text-accent">{(dwt.features.hl.energy_ratio * 100).toFixed(2)}%</span></td>
                            </tr>
                            <tr>
                              <td><strong>HH</strong></td>
                              <td>High-High (Diagonal Seams)</td>
                              <td><span className="text-danger">{(dwt.features.hh.energy_ratio * 100).toFixed(2)}%</span></td>
                            </tr>
                          </tbody>
                        </table>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* SECTION E: ANALYSIS SUMMARY */}
              {summary && (
                <div className="card summary-card">
                  <div className="summary-card-header">
                    <span className="summary-icon">📋</span>
                    <div>
                      <h4>Holistic Forensic Analysis Summary</h4>
                      <p className="text-muted" style={{ fontSize: '0.8rem', margin: 0 }}>
                        Cross-domain synthesis of spatial activations, frequency spectra, and wavelet decompositions
                      </p>
                    </div>
                  </div>
                  <div className="summary-body">
                    <p className="summary-text">{summary}</p>
                  </div>
                </div>
              )}

              {/* Action Buttons: View Report Modal & JSON */}
              <div className="card action-card">
                <div className="action-buttons-grid">
                  <button className="btn btn-primary" onClick={() => setShowReport(true)}>
                    📄 View Complete Forensic Report
                  </button>
                  <Link to="/frequency-analysis" className="btn btn-outline">
                    📡 Open Standalone Frequency Studio
                  </Link>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Image Zoom Modal */}
      {zoomImage && (
        <div className="modal-backdrop" onClick={() => setZoomImage(null)}>
          <div className="zoom-modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h4>{zoomImage.title}</h4>
              <button className="modal-close" onClick={() => setZoomImage(null)}>✕</button>
            </div>
            <div className="zoom-modal-body">
              <img src={zoomImage.src} alt={zoomImage.title} className="zoom-large-img" />
            </div>
          </div>
        </div>
      )}

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
                background: isReal ? 'rgba(16, 185, 129, 0.08)' : 'rgba(239, 68, 68, 0.08)',
                border: `1px solid ${isReal ? 'var(--success)' : 'var(--danger)'}`,
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
                      <td>Spatial Model</td>
                      <td>ResNet18 (PyTorch)</td>
                    </tr>
                    <tr>
                      <td>Inspection Date &amp; Time</td>
                      <td>{new Date().toLocaleString()}</td>
                    </tr>
                    <tr>
                      <td>Record ID</td>
                      <td><code>{result.history_id || 'SESSION-RECORD'}</code></td>
                    </tr>
                    <tr>
                      <td>Salient Region</td>
                      <td>{spatial?.salient_regions || 'Facial contours'}</td>
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

              {summary && (
                <div className="report-section">
                  <h4>Holistic Analytical Summary</h4>
                  <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                    {summary}
                  </p>
                </div>
              )}
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
