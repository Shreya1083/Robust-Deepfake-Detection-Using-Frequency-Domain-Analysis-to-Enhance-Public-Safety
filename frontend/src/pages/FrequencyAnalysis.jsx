import { useState, useCallback } from 'react'
import { Link } from 'react-router-dom'
import { analyzeFFT } from '../services/api'
import './FrequencyAnalysis.css'

const ACCEPT = '.jpg,.jpeg,.png,.bmp'
const MAX_MB = 20

function SpectrumImage({ b64, label, colorHint }) {
  if (!b64) return null
  return (
    <div className="spectrum-box">
      <div className="spectrum-label">{label}</div>
      <img
        src={`data:image/jpeg;base64,${b64}`}
        alt={label}
        className="spectrum-img"
        style={{ borderColor: colorHint }}
      />
    </div>
  )
}

function FeatureBar({ label, value, max = 1, color = 'var(--accent)', tooltip }) {
  const pct = Math.min(100, (value / max) * 100)
  return (
    <div className="feat-row" title={tooltip}>
      <div className="feat-label">
        <span>{label}</span>
        <span className="feat-val">{typeof value === 'number' ? value.toFixed(5) : value}</span>
      </div>
      <div className="progress-bar">
        <div className="progress-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  )
}

function RadialChart({ spectrum }) {
  if (!spectrum || spectrum.length === 0) return null
  const W = 320, H = 120, N = spectrum.length
  const pts = spectrum.map((v, i) => {
    const x = (i / (N - 1)) * W
    const y = H - v * H
    return `${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')

  return (
    <div className="radial-chart">
      <div className="chart-title">Azimuthally Averaged Radial Power Spectrum (64 Frequency Bins)</div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" height="120" className="chart-svg">
        <defs>
          <linearGradient id="specGrad" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%"   stopColor="#10b981" />
            <stop offset="40%"  stopColor="#0d9488" />
            <stop offset="100%" stopColor="#ef4444" />
          </linearGradient>
        </defs>
        {/* Grid lines */}
        {[0.25, 0.5, 0.75].map(v => (
          <line key={v} x1="0" y1={H - v * H} x2={W} y2={H - v * H}
                stroke="var(--border)" strokeWidth="1" strokeDasharray="4,4" />
        ))}
        {/* Spectrum line */}
        <polyline points={pts} fill="none" stroke="url(#specGrad)" strokeWidth="2.5" />
        {/* Band labels */}
        <text x="5"   y={H - 5} fontSize="9" fill="var(--success)" fontWeight="600">LOW</text>
        <text x={W * 0.12} y={H - 5} fontSize="9" fill="var(--accent)" fontWeight="600">MID</text>
        <text x={W * 0.44} y={H - 5} fontSize="9" fill="var(--danger)" fontWeight="600">HIGH</text>
        {/* Band boundary lines */}
        <line x1={W * 0.10} y1="0" x2={W * 0.10} y2={H} stroke="var(--border)" strokeWidth="1" strokeDasharray="2,3" />
        <line x1={W * 0.40} y1="0" x2={W * 0.40} y2={H} stroke="var(--border)" strokeWidth="1" strokeDasharray="2,3" />
      </svg>
      <div className="chart-axis">
        <span className="text-success">← Low Frequency</span>
        <span className="text-accent">Mid Frequency</span>
        <span className="text-danger">High Frequency →</span>
      </div>
    </div>
  )
}

export default function FrequencyAnalysis() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [dragging, setDragging] = useState(false)

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
    reader.onload = e => setPreview(e.target.result)
    reader.readAsDataURL(f)
  }

  const onInputChange = e => handleFile(e.target.files[0])

  const onDrop = useCallback(e => {
    e.preventDefault()
    setDragging(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }, [])

  const onDragOver  = e => { e.preventDefault(); setDragging(true) }
  const onDragLeave = () => setDragging(false)

  const analyze = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const fd = new FormData()
      fd.append('file', file)
      const { data } = await analyzeFFT(fd)
      setResult(data)
    } catch (err) {
      const respData = err.response?.data
      let errText = 'FFT frequency analysis failed.'
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
  }

  const f = result?.features

  return (
    <div className="page-container frequency-page">
      <div className="section-header">
        <span className="badge badge-teal">Spectral Forensic Analysis</span>
        <h1>Frequency Domain Analysis</h1>
        <p>
          Compute 2D Fast Fourier Transform (FFT) features and azimuthal radial power spectra.
          Generative models and face-swap algorithms often introduce subtle periodic artefacts
          and abnormal spectral energy concentrations in high-frequency bands.
        </p>
      </div>

      {/* Info Row */}
      <div className="info-row">
        {[
          { icon: '📡', label: '2D Fast Fourier Transform', desc: 'Real 2D NumPy FFT spectral decomposition' },
          { icon: '📊', label: '70-Dimensional Features', desc: '6 statistical scalars + 64 radial spectral bins' },
          { icon: '🛡️', label: 'Artifact Identification', desc: 'Uncovers GAN upsampling & diffusion anomalies' },
        ].map(c => (
          <div key={c.label} className="card info-chip">
            <span className="chip-icon">{c.icon}</span>
            <div>
              <div className="chip-label">{c.label}</div>
              <div className="chip-desc">{c.desc}</div>
            </div>
          </div>
        ))}
      </div>

      <div className="fft-layout">
        {/* Upload Panel */}
        <div className="fft-upload-panel">
          <div
            className={`drop-zone card ${dragging ? 'dragging' : ''} ${preview ? 'has-preview' : ''}`}
            onDrop={onDrop}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
          >
            {preview ? (
              <div className="preview-wrapper">
                <img src={preview} alt="Input" className="preview-img" />
                <div className="preview-meta">
                  <span className="text-muted">{file?.name}</span>
                  <span className="text-muted">{(file?.size / 1024).toFixed(1)} KB</span>
                </div>
              </div>
            ) : (
              <div className="drop-placeholder">
                <div className="drop-icon">📡</div>
                <h3>Drag &amp; drop image here</h3>
                <p className="text-muted">or click to select from files</p>
                <div className="upload-chips">
                  <span>JPG</span>
                  <span>PNG</span>
                  <span>BMP</span>
                  <span>Max {MAX_MB} MB</span>
                </div>
              </div>
            )}
            <input type="file" accept={ACCEPT} className="file-input" onChange={onInputChange} />
          </div>

          <div className="upload-actions">
            {file && !loading && (
              <button className="btn btn-primary" onClick={analyze}>
                📡 Compute Frequency Spectra
              </button>
            )}
            {loading && (
              <button className="btn btn-primary" disabled>
                <span className="spinner" style={{ width: 18, height: 18, borderWidth: 2 }} />
                Decomposing 2D Frequencies…
              </button>
            )}
            {(file || result) && (
              <button className="btn btn-outline" onClick={reset}>
                ✕ Reset
              </button>
            )}
          </div>

          {error && <div className="alert alert-danger" style={{ marginTop: '1rem' }}>⚠️ {error}</div>}

          {/* Educational Explainer */}
          <div className="card explainer-card" style={{ marginTop: '1.25rem' }}>
            <h4>Spectral Band Interpretation</h4>
            <ul className="explainer-list">
              <li>
                <span className="text-success">●</span> <strong>Low Frequency (0–10%):</strong> Captures gross illumination, global geometry, and coarse facial structure.
              </li>
              <li>
                <span className="text-accent">●</span> <strong>Mid Frequency (10–40%):</strong> Encodes facial contours, skin shading, and primary features.
              </li>
              <li>
                <span className="text-danger">●</span> <strong>High Frequency (40–100%):</strong> Contains fine dermal textures, noise floors, and transposed-convolution upsampling patterns.
              </li>
            </ul>
            <p className="explainer-note">
              Natural photographic sensors exhibit exponential falloff in high-frequency energy. Generative decoders often leave periodic peaks or abnormal high-frequency tails.
            </p>
          </div>
        </div>

        {/* Results Panel */}
        <div className="fft-results-panel">
          {!result && !loading && (
            <div className="card result-placeholder">
              <div className="placeholder-icon">📊</div>
              <h3>Awaiting Frequency Decomposition</h3>
              <p>Upload a facial image and click "Compute Frequency Spectra" to inspect its 2D Fourier transform and radial power distribution.</p>
            </div>
          )}

          {loading && (
            <div className="card result-placeholder">
              <div className="spinner" style={{ width: 36, height: 36, marginBottom: '1rem' }} />
              <h3>Calculating 2D Fast Fourier Transform…</h3>
              <p>Performing 2D FFT, shifting zero-frequency component to center, and extracting radial power bins.</p>
            </div>
          )}

          {result && !result.success && (
            <div className="alert alert-danger">⚠️ {result.error || 'Frequency analysis could not be completed.'}</div>
          )}

          {result?.success && (
            <div className="fft-results">
              {/* Spectral Visualizations */}
              <div className="card">
                <h4 style={{ marginBottom: '1.25rem', color: 'var(--text-primary)' }}>2D Spectral Decomposition</h4>
                <div className="spectra-row">
                  <SpectrumImage b64={result.original_b64} label="Grayscale Input (224×224)" colorHint="var(--border)" />
                  <SpectrumImage b64={result.magnitude_b64} label="FFT Magnitude Spectrum (BONE)" colorHint="var(--accent)" />
                  <SpectrumImage b64={result.log_spectrum_b64} label="Log Power Spectrum (JET)" colorHint="var(--danger)" />
                </div>
              </div>

              {/* Radial Spectrum Chart */}
              {f?.radial_spectrum && (
                <div className="card">
                  <RadialChart spectrum={f.radial_spectrum} />
                </div>
              )}

              {/* Extracted Features */}
              {f && (
                <div className="card">
                  <h4 style={{ marginBottom: '1rem', color: 'var(--text-primary)' }}>Quantitative Spectral Descriptors</h4>

                  <div className="feat-group-label">Summary Statistics</div>
                  <FeatureBar label="Mean Log Magnitude" value={f.mean_magnitude} max={20} color="var(--accent)" tooltip="Average log-magnitude across all frequency bins" />
                  <FeatureBar label="Std Log Magnitude" value={f.std_magnitude} max={10} color="var(--accent)" tooltip="Standard deviation of log-magnitude spectrum" />
                  <FeatureBar label="Spectral Entropy" value={f.spectral_entropy} max={1} color="var(--warning)" tooltip="Shannon entropy of normalized power (0=concentrated, 1=uniform)" />

                  <div className="feat-group-label" style={{ marginTop: '1.25rem' }}>Energy Band Distribution</div>
                  <FeatureBar label="Low-Frequency Energy (0–10%)" value={f.low_energy} max={1} color="var(--success)" tooltip="Fraction of spectral energy in low frequencies" />
                  <FeatureBar label="Mid-Frequency Energy (10–40%)" value={f.mid_energy} max={1} color="var(--accent)" tooltip="Fraction of spectral energy in mid frequencies" />
                  <FeatureBar label="High-Frequency Energy (40–100%)" value={f.high_energy} max={1} color="var(--danger)" tooltip="Fraction of spectral energy in high frequencies" />

                  {/* Energy Segmented Bar */}
                  <div className="energy-dist" style={{ marginTop: '1.25rem' }}>
                    <div className="energy-bar-wrap">
                      <div className="energy-seg" style={{ width: `${f.low_energy * 100}%`, background: 'var(--success)' }} title={`Low: ${(f.low_energy*100).toFixed(1)}%`} />
                      <div className="energy-seg" style={{ width: `${f.mid_energy * 100}%`, background: 'var(--accent)' }} title={`Mid: ${(f.mid_energy*100).toFixed(1)}%`} />
                      <div className="energy-seg" style={{ width: `${f.high_energy * 100}%`, background: 'var(--danger)' }} title={`High: ${(f.high_energy*100).toFixed(1)}%`} />
                    </div>
                    <div className="energy-labels">
                      <span className="text-success">Low: {(f.low_energy*100).toFixed(1)}%</span>
                      <span className="text-accent">Mid: {(f.mid_energy*100).toFixed(1)}%</span>
                      <span className="text-danger">High: {(f.high_energy*100).toFixed(1)}%</span>
                    </div>
                  </div>
                </div>
              )}

              {/* Technical Metadata */}
              {f && (
                <div className="card meta-card">
                  <div className="meta-row"><span>File Name</span><span>{result.filename}</span></div>
                  <div className="meta-row"><span>Input Matrix</span><span>{result.image_size?.join(' × ')} px</span></div>
                  <div className="meta-row"><span>Feature Dimensionality</span><span>{f.feature_dim} descriptors</span></div>
                  <div className="meta-row"><span>Azimuthal Resolution</span><span>64 radial bins</span></div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Multimodal Architecture Overview */}
      <div className="divider" style={{ margin: '3rem 0 2rem' }} />

      <section className="architecture-section">
        <h2>Multimodal Detection Architecture</h2>
        <p className="text-muted" style={{ maxWidth: 700, marginBottom: '1.5rem' }}>
          VeriFrame combines spatial CNN representations with frequency domain signatures to provide robust defense against modern generative models.
        </p>

        <div className="card architecture-card">
          <div className="arch-flow">
            <div className="arch-box">
              <span className="arch-icon">🧠</span>
              <strong>Spatial CNN</strong>
              <small>512-dim ResNet18 Embeddings</small>
            </div>
            <div className="arch-plus">+</div>
            <div className="arch-box arch-active">
              <span className="arch-icon">📡</span>
              <strong>2D FFT Spectrum</strong>
              <small>70-dim Frequency Vector</small>
            </div>
            <div className="arch-plus">+</div>
            <div className="arch-box">
              <span className="arch-icon">🌊</span>
              <strong>DWT Wavelets</strong>
              <small>Multi-scale Subband Features</small>
            </div>
            <div className="arch-arrow">→</div>
            <div className="arch-box arch-output">
              <span className="arch-icon">🛡️</span>
              <strong>Multimodal Classifier</strong>
              <small>Unified Real / Deepfake Verdict</small>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}
