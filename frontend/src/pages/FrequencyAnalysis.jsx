import { useState, useCallback } from 'react'
import { analyzeFFT } from '../services/api'
import './FrequencyAnalysis.css'

const ACCEPT = '.jpg,.jpeg,.png,.bmp'
const MAX_MB = 20

/* ---- sub-components ---- */

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
      <div className="chart-title">Azimuthally Averaged Radial Power Spectrum (64 bins)</div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" height="120" className="chart-svg">
        <defs>
          <linearGradient id="specGrad" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%"   stopColor="#10b981" />
            <stop offset="40%"  stopColor="#00d4ff" />
            <stop offset="100%" stopColor="#ef4444" />
          </linearGradient>
        </defs>
        {/* Grid lines */}
        {[0.25, 0.5, 0.75].map(v => (
          <line key={v} x1="0" y1={H - v * H} x2={W} y2={H - v * H}
                stroke="var(--border)" strokeWidth="1" strokeDasharray="4,4" />
        ))}
        {/* Spectrum line */}
        <polyline points={pts} fill="none" stroke="url(#specGrad)" strokeWidth="2" />
        {/* Band labels */}
        <text x="5"   y={H - 5} fontSize="9" fill="var(--success)">LOW</text>
        <text x={W * 0.12} y={H - 5} fontSize="9" fill="var(--accent)">MID</text>
        <text x={W * 0.44} y={H - 5} fontSize="9" fill="var(--danger)">HIGH</text>
        {/* Band boundary lines */}
        <line x1={W * 0.10} y1="0" x2={W * 0.10} y2={H} stroke="var(--border)" strokeWidth="1" strokeDasharray="2,3" />
        <line x1={W * 0.40} y1="0" x2={W * 0.40} y2={H} stroke="var(--border)" strokeWidth="1" strokeDasharray="2,3" />
      </svg>
      <div className="chart-axis">
        <span className="text-success">← Low freq</span>
        <span className="text-accent">Mid freq</span>
        <span className="text-danger">High freq →</span>
      </div>
    </div>
  )
}

/* ---- Main Page ---- */

export default function FrequencyAnalysis() {
  const [file, setFile]       = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult]   = useState(null)
  const [error, setError]     = useState(null)
  const [dragging, setDragging] = useState(false)

  const handleFile = (f) => {
    if (!f) return
    if (f.size > MAX_MB * 1024 * 1024) { setError(`File exceeds ${MAX_MB} MB.`); return }
    setFile(f); setResult(null); setError(null)
    const reader = new FileReader()
    reader.onload = e => setPreview(e.target.result)
    reader.readAsDataURL(f)
  }

  const onInputChange = e => handleFile(e.target.files[0])
  const onDrop = useCallback(e => {
    e.preventDefault(); setDragging(false)
    handleFile(e.dataTransfer.files[0])
  }, [])
  const onDragOver  = e => { e.preventDefault(); setDragging(true) }
  const onDragLeave = () => setDragging(false)

  const analyze = async () => {
    if (!file) return
    setLoading(true); setError(null); setResult(null)
    try {
      const fd = new FormData()
      fd.append('file', file)
      const { data } = await analyzeFFT(fd)
      setResult(data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Request failed.')
    } finally { setLoading(false) }
  }

  const reset = () => { setFile(null); setPreview(null); setResult(null); setError(null) }

  const f = result?.features

  return (
    <div className="page-container">
      {/* Header */}
      <div className="section-header">
        <div className="label">Phase 4 · FFT Analysis</div>
        <h1>Frequency Analysis</h1>
        <p style={{ marginTop: '0.5rem', maxWidth: 680 }}>
          Upload a face image to compute its real 2D Fast Fourier Transform.
          Deepfake images often show characteristic spectral artefacts in the high-frequency bands.
        </p>
      </div>

      {/* Info row */}
      <div className="info-row">
        {[
          { icon: '📡', label: 'FFT Analysis', desc: 'Real 2D NumPy FFT — no simulated values' },
          { icon: '📊', label: '70-dim Features', desc: '6 scalars + 64-bin radial spectrum' },
          { icon: '🔗', label: 'Phase 6 Ready', desc: 'Feature vector reusable for fusion' },
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

      {/* Upload + Results layout */}
      <div className="fft-layout">
        {/* Upload panel */}
        <div className="fft-upload-panel">
          <div
            className={`drop-zone card ${dragging ? 'dragging' : ''} ${preview ? 'has-preview' : ''}`}
            onDrop={onDrop} onDragOver={onDragOver} onDragLeave={onDragLeave}
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
                <div className="drop-icon">📡</div>
                <p>Drag &amp; drop an image here</p>
                <p className="text-muted" style={{ fontSize: '0.85rem' }}>or click to browse</p>
                <p className="text-muted" style={{ fontSize: '0.8rem' }}>JPG · PNG · BMP · max {MAX_MB} MB</p>
              </div>
            )}
            <input type="file" accept={ACCEPT} className="file-input" onChange={onInputChange} />
          </div>

          <div className="upload-actions">
            {file && !loading && (
              <button className="btn btn-primary" onClick={analyze}>📡 Run FFT Analysis</button>
            )}
            {loading && (
              <button className="btn btn-primary" disabled>
                <span className="spinner" style={{ width: 18, height: 18, borderWidth: 2 }} />
                Computing FFT…
              </button>
            )}
            {(file || result) && (
              <button className="btn btn-outline" onClick={reset}>✕ Clear</button>
            )}
          </div>

          {error && <div className="alert alert-danger" style={{ marginTop: '1rem' }}>⚠️ {error}</div>}

          {/* What FFT shows */}
          <div className="card explainer-card" style={{ marginTop: '1.25rem' }}>
            <h4 className="text-accent">What the spectrum reveals</h4>
            <ul className="explainer-list">
              <li><span className="text-success">●</span> <strong>Low band (0–10%)</strong> — global illumination and face structure</li>
              <li><span className="text-accent">●</span> <strong>Mid band (10–40%)</strong> — coarse textures, facial features</li>
              <li><span className="text-danger">●</span> <strong>High band (40–100%)</strong> — fine edges, GAN upsampling artefacts</li>
            </ul>
            <p style={{ fontSize: '0.8rem', marginTop: '0.75rem' }}>
              GAN-generated deepfakes often show elevated or anomalous high-frequency energy due to 
              transposed-convolution upsampling patterns. FFT alone does not classify REAL/DEEPFAKE — 
              it contributes one of three feature modalities combined in Phase 6.
            </p>
          </div>
        </div>

        {/* Results panel */}
        <div className="fft-results-panel">
          {!result && !loading && (
            <div className="card result-placeholder">
              <div className="placeholder-icon">📊</div>
              <h3>FFT results will appear here</h3>
              <p>Upload an image and click Run FFT Analysis.</p>
            </div>
          )}

          {loading && (
            <div className="card result-placeholder">
              <div className="spinner" style={{ marginBottom: '1rem' }} />
              <h3>Computing 2D FFT…</h3>
              <p>Running NumPy FFT, extracting 70-dim feature vector.</p>
            </div>
          )}

          {result && !result.success && (
            <div className="alert alert-danger">⚠️ {result.error || 'FFT analysis failed.'}</div>
          )}

          {result?.success && (
            <div className="fft-results">
              {/* Spectrum visualisations */}
              <div className="card">
                <h4 style={{ marginBottom: '1rem' }}>Frequency Spectra</h4>
                <div className="spectra-row">
                  <SpectrumImage b64={result.original_b64}     label="Grayscale Input (224×224)" colorHint="var(--border)" />
                  <SpectrumImage b64={result.magnitude_b64}    label="FFT Magnitude Spectrum (BONE)" colorHint="var(--accent)" />
                  <SpectrumImage b64={result.log_spectrum_b64} label="Log Spectrum (JET)" colorHint="var(--danger)" />
                </div>
              </div>

              {/* Radial spectrum chart */}
              {f?.radial_spectrum && (
                <div className="card">
                  <RadialChart spectrum={f.radial_spectrum} />
                </div>
              )}

              {/* Numerical features */}
              {f && (
                <div className="card">
                  <h4 style={{ marginBottom: '1rem' }}>Extracted FFT Features</h4>

                  <div className="feat-group-label">Summary Statistics</div>
                  <FeatureBar label="Mean Log Magnitude"   value={f.mean_magnitude}   max={20}   color="var(--accent)"  tooltip="Average log-magnitude across all frequency components" />
                  <FeatureBar label="Std Log Magnitude"    value={f.std_magnitude}    max={10}   color="var(--accent)"  tooltip="Standard deviation of log-magnitude spectrum" />
                  <FeatureBar label="Spectral Entropy"     value={f.spectral_entropy} max={1}    color="var(--warning)" tooltip="Shannon entropy of normalised power (0=concentrated, 1=uniform)" />

                  <div className="feat-group-label" style={{ marginTop: '1rem' }}>Energy Band Distribution</div>
                  <FeatureBar label="🟢 Low-frequency energy  (0–10%)"  value={f.low_energy}  max={1} color="var(--success)" tooltip="Fraction of total spectral energy in low frequencies (structure)" />
                  <FeatureBar label="🔵 Mid-frequency energy  (10–40%)" value={f.mid_energy}  max={1} color="var(--accent)"  tooltip="Fraction of total spectral energy in mid frequencies (textures)" />
                  <FeatureBar label="🔴 High-frequency energy (40–100%)" value={f.high_energy} max={1} color="var(--danger)"  tooltip="Fraction of total spectral energy in high frequencies (edges/GAN)" />

                  {/* Energy distribution mini-chart */}
                  <div className="energy-dist">
                    <div className="energy-bar-wrap">
                      <div className="energy-seg" style={{ width: `${f.low_energy * 100}%`,  background: 'var(--success)' }} title={`Low: ${(f.low_energy*100).toFixed(1)}%`} />
                      <div className="energy-seg" style={{ width: `${f.mid_energy * 100}%`,  background: 'var(--accent)'  }} title={`Mid: ${(f.mid_energy*100).toFixed(1)}%`} />
                      <div className="energy-seg" style={{ width: `${f.high_energy * 100}%`, background: 'var(--danger)'  }} title={`High: ${(f.high_energy*100).toFixed(1)}%`} />
                    </div>
                    <div className="energy-labels">
                      <span className="text-success">Low {(f.low_energy*100).toFixed(1)}%</span>
                      <span className="text-accent">Mid {(f.mid_energy*100).toFixed(1)}%</span>
                      <span className="text-danger">High {(f.high_energy*100).toFixed(1)}%</span>
                    </div>
                  </div>
                </div>
              )}

              {/* Meta */}
              {f && (
                <div className="card meta-card">
                  <div className="meta-row"><span>File</span><span>{result.filename}</span></div>
                  <div className="meta-row"><span>FFT Input Size</span><span>{result.image_size?.join(' × ')} px</span></div>
                  <div className="meta-row"><span>Feature Vector Dim</span><span>{f.feature_dim}</span></div>
                  <div className="meta-row"><span>Radial Bins</span><span>64</span></div>
                  <div className="meta-row"><span>Phase 6 Ready</span><span className="text-success">✓ Vector exportable</span></div>
                </div>
              )}

              {/* DWT placeholder */}
              <div className="card placeholder-future">
                <h4>🌊 DWT Wavelet Analysis <span className="badge badge-warning">Phase 5</span></h4>
                <p>Wavelet sub-band decomposition (LL · LH · HL · HH) will appear here after Phase 5.</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* FFT Explanation + Fusion section below */}
      <div className="divider" />

      <section className="freq-section">
        <h2>🔗 Phase 6 — Feature Fusion (Preview)</h2>
        <div className="card fusion-card">
          <div className="fusion-diagram">
            <div className="fusion-box">🧠 CNN<br/><small>512-dim</small></div>
            <div className="fusion-plus">+</div>
            <div className="fusion-box fusion-active">📡 FFT<br/><small>70-dim ✓</small></div>
            <div className="fusion-plus">+</div>
            <div className="fusion-box">🌊 DWT<br/><small>128-dim</small></div>
            <div className="fusion-arrow">→</div>
            <div className="fusion-box fusion-output">🛡️ Fused<br/><small>MLP → REAL/FAKE</small></div>
          </div>
          <p style={{ marginTop: '1rem' }}>
            FFT features (70-dim) are now implemented and will be concatenated with
            CNN spatial embeddings (512-dim) and DWT wavelet features (Phase 5) in Phase 6.
          </p>
        </div>
      </section>
    </div>
  )
}
