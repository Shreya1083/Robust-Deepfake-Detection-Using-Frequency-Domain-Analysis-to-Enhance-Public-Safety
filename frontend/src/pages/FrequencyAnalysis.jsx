import { useState, useCallback } from 'react'
import { Link } from 'react-router-dom'
import { analyzeFFT, analyzeDWT } from '../services/api'
import './FrequencyAnalysis.css'

const ACCEPT = '.jpg,.jpeg,.png,.bmp'
const MAX_MB = 20

function SubbandCard({ b64, label, sublabel, desc, colorHint, badge }) {
  if (!b64) return null
  return (
    <div className="subband-card card">
      <div className="subband-header">
        <span className="subband-badge" style={{ backgroundColor: colorHint }}>
          {badge}
        </span>
        <div className="subband-titles">
          <div className="subband-label">{label}</div>
          <div className="subband-sublabel">{sublabel}</div>
        </div>
      </div>
      <div className="subband-img-box">
        <img
          src={`data:image/jpeg;base64,${b64}`}
          alt={label}
          className="subband-img"
          style={{ borderColor: colorHint }}
        />
      </div>
      <p className="subband-desc">{desc}</p>
    </div>
  )
}

function FeatureBar({ label, value, max = 1, color = 'var(--accent)', tooltip }) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100))
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
  const [loadingMode, setLoadingMode] = useState('')
  const [activeTab, setActiveTab] = useState('dwt') // 'dwt' | 'fft' | 'both'
  const [fftResult, setFftResult] = useState(null)
  const [dwtResult, setDwtResult] = useState(null)
  const [error, setError] = useState(null)
  const [dragging, setDragging] = useState(false)

  const handleFile = (f) => {
    if (!f) return
    if (f.size > MAX_MB * 1024 * 1024) {
      setError(`File exceeds maximum size of ${MAX_MB} MB.`)
      return
    }
    setFile(f)
    setFftResult(null)
    setDwtResult(null)
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

  // Run DWT Analysis
  const runDWT = async () => {
    if (!file) return
    setLoading(true)
    setLoadingMode('DWT')
    setError(null)
    try {
      const fd = new FormData()
      fd.append('file', file)
      const { data } = await analyzeDWT(fd)
      setDwtResult(data)
      setActiveTab('dwt')
    } catch (err) {
      const respData = err.response?.data
      let errText = 'DWT wavelet analysis failed.'
      if (typeof respData?.detail === 'string') {
        errText = respData.detail
      } else if (err.message) {
        errText = err.message
      }
      setError(errText)
    } finally {
      setLoading(false)
      setLoadingMode('')
    }
  }

  // Run FFT Analysis
  const runFFT = async () => {
    if (!file) return
    setLoading(true)
    setLoadingMode('FFT')
    setError(null)
    try {
      const fd = new FormData()
      fd.append('file', file)
      const { data } = await analyzeFFT(fd)
      setFftResult(data)
      setActiveTab('fft')
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
      setLoadingMode('')
    }
  }

  // Run Both FFT and DWT
  const runAllAnalyses = async () => {
    if (!file) return
    setLoading(true)
    setLoadingMode('ALL')
    setError(null)
    try {
      const fd1 = new FormData()
      fd1.append('file', file)
      const fd2 = new FormData()
      fd2.append('file', file)

      const [dwtRes, fftRes] = await Promise.all([
        analyzeDWT(fd1),
        analyzeFFT(fd2),
      ])
      setDwtResult(dwtRes.data)
      setFftResult(fftRes.data)
      setActiveTab('dwt')
    } catch (err) {
      const respData = err.response?.data
      let errText = 'Frequency domain analysis failed.'
      if (typeof respData?.detail === 'string') {
        errText = respData.detail
      } else if (err.message) {
        errText = err.message
      }
      setError(errText)
    } finally {
      setLoading(false)
      setLoadingMode('')
    }
  }

  const reset = () => {
    setFile(null)
    setPreview(null)
    setFftResult(null)
    setDwtResult(null)
    setError(null)
  }

  const dwtF = dwtResult?.features
  const fftF = fftResult?.features

  return (
    <div className="page-container frequency-page">
      <div className="section-header">
        <span className="badge badge-teal">Spectral &amp; Wavelet Forensics</span>
        <h1>Frequency Domain Analysis</h1>
        <p>
          Analyze face images using 2D Discrete Wavelet Transform (DWT) and 2D Fast Fourier Transform (FFT).
          Synthesized faces generated by GANs and diffusion decoders exhibit characteristic energy concentrations
          and boundary artifacts in sub-band frequency representations.
        </p>
      </div>

      {/* Info Row */}
      <div className="info-row">
        {[
          { icon: '🌊', label: '2D Discrete Wavelet Transform', desc: 'Haar multi-scale sub-band decomposition (LL, LH, HL, HH)' },
          { icon: '📡', label: '2D Fast Fourier Transform', desc: 'Centered magnitude spectrum & 64 radial frequency bins' },
          { icon: '🛡️', label: 'Artifact Identification', desc: 'Uncovers GAN transposed convolutions & diffusion high-freq anomalies' },
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
                <div className="drop-icon">🌊</div>
                <h3>Drag &amp; drop face image here</h3>
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

          {/* Action Buttons */}
          <div className="upload-actions-grid">
            {file && !loading && (
              <>
                <button className="btn btn-primary btn-block" onClick={runAllAnalyses}>
                  ✨ Analyze All (DWT + FFT)
                </button>
                <div className="upload-actions-sub">
                  <button className="btn btn-outline" onClick={runDWT}>
                    🌊 Compute DWT
                  </button>
                  <button className="btn btn-outline" onClick={runFFT}>
                    📡 Compute FFT
                  </button>
                </div>
              </>
            )}
            {loading && (
              <button className="btn btn-primary btn-block" disabled>
                <span className="spinner" style={{ width: 18, height: 18, borderWidth: 2 }} />
                {loadingMode === 'DWT' && 'Computing 2D Discrete Wavelet Transform…'}
                {loadingMode === 'FFT' && 'Decomposing 2D Fourier Frequencies…'}
                {loadingMode === 'ALL' && 'Extracting DWT Wavelets & FFT Spectra…'}
              </button>
            )}
            {(file || dwtResult || fftResult) && (
              <button className="btn btn-ghost" onClick={reset}>
                ✕ Reset Image
              </button>
            )}
          </div>

          {error && <div className="alert alert-danger" style={{ marginTop: '0.75rem' }}>⚠️ {error}</div>}

          {/* Educational Explainer */}
          <div className="card explainer-card" style={{ marginTop: '1rem' }}>
            <h4>DWT Sub-band Decomposition Interpretation</h4>
            <ul className="explainer-list">
              <li>
                <span className="text-success">●</span> <strong>LL (Approximation):</strong> Coarse, low-pass facial structure, lighting, and skin tones.
              </li>
              <li>
                <span className="text-accent">●</span> <strong>LH (Horizontal Details):</strong> Horizontal edges, eye contours, mouth creases, and vertical frequency transitions.
              </li>
              <li>
                <span className="text-warning">●</span> <strong>HL (Vertical Details):</strong> Vertical contours, nose bridge, jawline boundaries, and horizontal frequency transitions.
              </li>
              <li>
                <span className="text-danger">●</span> <strong>HH (Diagonal Details):</strong> High-frequency corners, fine dermal texture, sensor noise floor, and synthetic synthesis artifacts.
              </li>
            </ul>
            <p className="explainer-note">
              Wavelet decomposition separates spatial information by frequency and orientation simultaneously, making it sensitive to synthetic boundary irregularities and grid checkerboard artifacts that appear during neural image generation.
            </p>
          </div>
        </div>

        {/* Results Panel */}
        <div className="fft-results-panel">
          {!dwtResult && !fftResult && !loading && (
            <div className="card result-placeholder">
              <div className="placeholder-icon">🌊</div>
              <h3>Awaiting Frequency Decomposition</h3>
              <p>Upload a facial image and select an analysis method to inspect its 2D Wavelet sub-bands and Fourier power distribution.</p>
            </div>
          )}

          {loading && (
            <div className="card result-placeholder">
              <div className="spinner" style={{ width: 36, height: 36, marginBottom: '1rem' }} />
              <h3>Calculating Multi-Scale Frequency Decomposition…</h3>
              <p>Applying 2D Haar Wavelet transform, extracting sub-band coefficient statistics, and computing energy distributions.</p>
            </div>
          )}

          {(dwtResult || fftResult) && (
            <div className="frequency-results-container">
              {/* Navigation Tabs */}
              <div className="analysis-tabs">
                <button
                  className={`analysis-tab ${activeTab === 'dwt' ? 'active' : ''}`}
                  onClick={() => setActiveTab('dwt')}
                >
                  🌊 DWT Wavelet Analysis {dwtResult ? '✓' : ''}
                </button>
                <button
                  className={`analysis-tab ${activeTab === 'fft' ? 'active' : ''}`}
                  onClick={() => setActiveTab('fft')}
                >
                  📡 FFT Spectral Analysis {fftResult ? '✓' : ''}
                </button>
                <button
                  className={`analysis-tab ${activeTab === 'both' ? 'active' : ''}`}
                  onClick={() => setActiveTab('both')}
                >
                  📊 Multi-Domain Comparison
                </button>
              </div>

              {/* ======================================================== */}
              {/* DWT ANALYSIS SECTION                                     */}
              {/* ======================================================== */}
              {(activeTab === 'dwt' || activeTab === 'both') && (
                <div className="dwt-section">
                  {!dwtResult ? (
                    <div className="card sub-placeholder">
                      <p>DWT has not been computed yet for this image.</p>
                      <button className="btn btn-primary btn-sm" onClick={runDWT}>Run DWT Analysis</button>
                    </div>
                  ) : !dwtResult.success ? (
                    <div className="alert alert-danger">⚠️ {dwtResult.error || 'DWT analysis failed.'}</div>
                  ) : (
                    <div className="dwt-content">
                      {/* Sub-bands Grid View */}
                      <div className="card">
                        <div className="card-header-flex">
                          <div>
                            <h3 className="section-subtitle">2D Discrete Wavelet Decomposition</h3>
                            <p className="text-muted" style={{ fontSize: '0.8rem', margin: 0 }}>
                              Single-level 2D Haar Wavelet decomposition yielding 4 spatial-frequency sub-bands (112×112 px each)
                            </p>
                          </div>
                          <span className="badge badge-teal">Haar Wavelet</span>
                        </div>

                        {/* 5-Image Showcase: Original + 4 Sub-bands */}
                        <div className="subbands-grid">
                          <SubbandCard
                            b64={dwtResult.original_b64}
                            label="Original Input"
                            sublabel="224×224 px"
                            desc="Standardized grayscale facial frame fed into the 2D wavelet transform."
                            colorHint="var(--primary)"
                            badge="ORIG"
                          />
                          <SubbandCard
                            b64={dwtResult.ll_b64}
                            label="LL Approximation"
                            sublabel="Low-Low Sub-band"
                            desc="Low-pass filtered approximation capturing primary facial anatomy and illumination."
                            colorHint="var(--success)"
                            badge="LL"
                          />
                          <SubbandCard
                            b64={dwtResult.lh_b64}
                            label="LH Horizontal Details"
                            sublabel="Low-High Sub-band"
                            desc="Horizontal detail features highlighting horizontal edges, eyelids, and lips."
                            colorHint="var(--accent)"
                            badge="LH"
                          />
                          <SubbandCard
                            b64={dwtResult.hl_b64}
                            label="HL Vertical Details"
                            sublabel="High-Low Sub-band"
                            desc="Vertical detail features highlighting facial borders, nose ridge, and vertical lines."
                            colorHint="var(--warning)"
                            badge="HL"
                          />
                          <SubbandCard
                            b64={dwtResult.hh_b64}
                            label="HH Diagonal Details"
                            sublabel="High-High Sub-band"
                            desc="Diagonal high-frequency textures, noise floor, and synthesized checkerboard artifacts."
                            colorHint="var(--danger)"
                            badge="HH"
                          />
                        </div>

                        {/* Combined 2x2 Decomposition Grid */}
                        {dwtResult.combined_b64 && (
                          <div className="combined-grid-box" style={{ marginTop: '1.25rem' }}>
                            <div className="combined-grid-header">
                              <strong>Standard 2×2 Wavelet Decomposition Quad-Grid</strong>
                              <span className="text-muted" style={{ fontSize: '0.75rem' }}>[ LL | HL ] on top, [ LH | HH ] on bottom</span>
                            </div>
                            <div className="combined-img-wrap">
                              <img
                                src={`data:image/jpeg;base64,${dwtResult.combined_b64}`}
                                alt="2x2 Wavelet Grid"
                                className="combined-img"
                              />
                            </div>
                          </div>
                        )}
                      </div>

                      {/* Wavelet Energy Measurements & Distribution */}
                      {dwtF && (
                        <div className="card">
                          <h4 style={{ marginBottom: '1rem', color: 'var(--text-primary)' }}>Wavelet Energy Measurements</h4>

                          <div className="energy-summary-grid">
                            <div className="energy-stat-card">
                              <span className="energy-stat-num">{(dwtF.energy_distribution.ll_ratio * 100).toFixed(2)}%</span>
                              <span className="energy-stat-label">LL Approx Energy</span>
                            </div>
                            <div className="energy-stat-card">
                              <span className="energy-stat-num">{(dwtF.high_freq_energy_ratio * 100).toFixed(2)}%</span>
                              <span className="energy-stat-label">Detail Bands Energy</span>
                            </div>
                            <div className="energy-stat-card">
                              <span className="energy-stat-num">{dwtF.wavelet_entropy.toFixed(4)}</span>
                              <span className="energy-stat-label">Wavelet Shannon Entropy</span>
                            </div>
                            <div className="energy-stat-card">
                              <span className="energy-stat-num">{dwtF.detail_to_approx_ratio.toFixed(5)}</span>
                              <span className="energy-stat-label">Detail-to-Approx Ratio</span>
                            </div>
                          </div>

                          <div className="feat-group-label" style={{ marginTop: '1.25rem' }}>Sub-Band Relative Energy Distribution</div>
                          <FeatureBar
                            label="LL Approximation Energy (0 Hz to Nyquist/2)"
                            value={dwtF.energy_distribution.ll_ratio}
                            max={1}
                            color="var(--success)"
                            tooltip="Ratio of energy in low-frequency approximation"
                          />
                          <FeatureBar
                            label="LH Horizontal Detail Energy"
                            value={dwtF.energy_distribution.lh_ratio}
                            max={0.2}
                            color="var(--accent)"
                            tooltip="Ratio of energy in horizontal details"
                          />
                          <FeatureBar
                            label="HL Vertical Detail Energy"
                            value={dwtF.energy_distribution.hl_ratio}
                            max={0.2}
                            color="var(--warning)"
                            tooltip="Ratio of energy in vertical details"
                          />
                          <FeatureBar
                            label="HH Diagonal Detail Energy"
                            value={dwtF.energy_distribution.hh_ratio}
                            max={0.2}
                            color="var(--danger)"
                            tooltip="Ratio of energy in diagonal details"
                          />

                          {/* Relative Distribution Segmented Bar */}
                          <div className="energy-dist" style={{ marginTop: '1.25rem' }}>
                            <div className="energy-bar-wrap">
                              <div
                                className="energy-seg"
                                style={{ width: `${Math.max(1, dwtF.energy_distribution.ll_ratio * 100)}%`, background: 'var(--success)' }}
                                title={`LL: ${(dwtF.energy_distribution.ll_ratio * 100).toFixed(2)}%`}
                              />
                              <div
                                className="energy-seg"
                                style={{ width: `${dwtF.energy_distribution.lh_ratio * 100}%`, background: 'var(--accent)' }}
                                title={`LH: ${(dwtF.energy_distribution.lh_ratio * 100).toFixed(2)}%`}
                              />
                              <div
                                className="energy-seg"
                                style={{ width: `${dwtF.energy_distribution.hl_ratio * 100}%`, background: 'var(--warning)' }}
                                title={`HL: ${(dwtF.energy_distribution.hl_ratio * 100).toFixed(2)}%`}
                              />
                              <div
                                className="energy-seg"
                                style={{ width: `${dwtF.energy_distribution.hh_ratio * 100}%`, background: 'var(--danger)' }}
                                title={`HH: ${(dwtF.energy_distribution.hh_ratio * 100).toFixed(2)}%`}
                              />
                            </div>
                            <div className="energy-labels">
                              <span className="text-success">LL: {(dwtF.energy_distribution.ll_ratio * 100).toFixed(1)}%</span>
                              <span className="text-accent">LH: {(dwtF.energy_distribution.lh_ratio * 100).toFixed(2)}%</span>
                              <span className="text-warning">HL: {(dwtF.energy_distribution.hl_ratio * 100).toFixed(2)}%</span>
                              <span className="text-danger">HH: {(dwtF.energy_distribution.hh_ratio * 100).toFixed(2)}%</span>
                            </div>
                          </div>
                        </div>
                      )}

                      {/* Sub-band Numerical Statistics Table */}
                      {dwtF && (
                        <div className="card">
                          <h4 style={{ marginBottom: '1rem', color: 'var(--text-primary)' }}>Sub-Band Quantitative Descriptors</h4>
                          <div className="table-responsive">
                            <table className="dwt-table">
                              <thead>
                                <tr>
                                  <th>Sub-Band</th>
                                  <th>Mean</th>
                                  <th>Std Dev</th>
                                  <th>Mean Abs</th>
                                  <th>Max Abs</th>
                                  <th>Energy</th>
                                  <th>Energy Ratio</th>
                                </tr>
                              </thead>
                              <tbody>
                                <tr>
                                  <td><span className="badge badge-success">LL</span> Approx</td>
                                  <td>{dwtF.subbands.LL.mean.toFixed(3)}</td>
                                  <td>{dwtF.subbands.LL.std.toFixed(3)}</td>
                                  <td>{dwtF.subbands.LL.mean_abs.toFixed(3)}</td>
                                  <td>{dwtF.subbands.LL.max_abs.toFixed(2)}</td>
                                  <td>{dwtF.subbands.LL.energy.toExponential(2)}</td>
                                  <td><strong>{(dwtF.subbands.LL.energy_ratio * 100).toFixed(2)}%</strong></td>
                                </tr>
                                <tr>
                                  <td><span className="badge badge-teal">LH</span> Horiz</td>
                                  <td>{dwtF.subbands.LH.mean.toFixed(3)}</td>
                                  <td>{dwtF.subbands.LH.std.toFixed(3)}</td>
                                  <td>{dwtF.subbands.LH.mean_abs.toFixed(3)}</td>
                                  <td>{dwtF.subbands.LH.max_abs.toFixed(2)}</td>
                                  <td>{dwtF.subbands.LH.energy.toExponential(2)}</td>
                                  <td><strong>{(dwtF.subbands.LH.energy_ratio * 100).toFixed(2)}%</strong></td>
                                </tr>
                                <tr>
                                  <td><span className="badge badge-amber">HL</span> Vert</td>
                                  <td>{dwtF.subbands.HL.mean.toFixed(3)}</td>
                                  <td>{dwtF.subbands.HL.std.toFixed(3)}</td>
                                  <td>{dwtF.subbands.HL.mean_abs.toFixed(3)}</td>
                                  <td>{dwtF.subbands.HL.max_abs.toFixed(2)}</td>
                                  <td>{dwtF.subbands.HL.energy.toExponential(2)}</td>
                                  <td><strong>{(dwtF.subbands.HL.energy_ratio * 100).toFixed(2)}%</strong></td>
                                </tr>
                                <tr>
                                  <td><span className="badge badge-danger">HH</span> Diag</td>
                                  <td>{dwtF.subbands.HH.mean.toFixed(3)}</td>
                                  <td>{dwtF.subbands.HH.std.toFixed(3)}</td>
                                  <td>{dwtF.subbands.HH.mean_abs.toFixed(3)}</td>
                                  <td>{dwtF.subbands.HH.max_abs.toFixed(2)}</td>
                                  <td>{dwtF.subbands.HH.energy.toExponential(2)}</td>
                                  <td><strong>{(dwtF.subbands.HH.energy_ratio * 100).toFixed(2)}%</strong></td>
                                </tr>
                              </tbody>
                            </table>
                          </div>
                        </div>
                      )}

                      {/* Technical Metadata */}
                      {dwtF && (
                        <div className="card meta-card">
                          <div className="meta-row"><span>File Name</span><span>{dwtResult.filename}</span></div>
                          <div className="meta-row"><span>Input Matrix</span><span>{dwtResult.image_size?.join(' × ')} px</span></div>
                          <div className="meta-row"><span>Sub-Band Dimensions</span><span>{dwtResult.subband_shape?.join(' × ')} px (1-level)</span></div>
                          <div className="meta-row"><span>Wavelet Family</span><span>Haar (pywt.dwt2)</span></div>
                          <div className="meta-row"><span>Feature Dimensionality</span><span>{dwtF.feature_dim} scalar descriptors</span></div>
                          <div className="meta-row"><span>Total Wavelet Energy</span><span>{dwtF.total_energy.toExponential(4)}</span></div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* ======================================================== */}
              {/* FFT ANALYSIS SECTION                                     */}
              {/* ======================================================== */}
              {(activeTab === 'fft' || activeTab === 'both') && (
                <div className="fft-section" style={{ marginTop: activeTab === 'both' ? '2rem' : '0' }}>
                  {!fftResult ? (
                    <div className="card sub-placeholder">
                      <p>FFT has not been computed yet for this image.</p>
                      <button className="btn btn-primary btn-sm" onClick={runFFT}>Run FFT Analysis</button>
                    </div>
                  ) : !fftResult.success ? (
                    <div className="alert alert-danger">⚠️ {fftResult.error || 'FFT frequency analysis failed.'}</div>
                  ) : (
                    <div className="fft-results">
                      {/* Spectral Visualizations */}
                      <div className="card">
                        <div className="card-header-flex">
                          <div>
                            <h3 className="section-subtitle">2D Fast Fourier Spectral Decomposition</h3>
                            <p className="text-muted" style={{ fontSize: '0.8rem', margin: 0 }}>
                              2D FFT with zero-frequency DC component shifted to matrix center
                            </p>
                          </div>
                          <span className="badge badge-teal">NumPy 2D FFT</span>
                        </div>

                        <div className="spectra-row" style={{ marginTop: '1.25rem' }}>
                          <div className="spectrum-box">
                            <div className="spectrum-label">Grayscale Input (224×224)</div>
                            <img
                              src={`data:image/jpeg;base64,${fftResult.original_b64}`}
                              alt="Grayscale Input"
                              className="spectrum-img"
                              style={{ borderColor: 'var(--border)' }}
                            />
                          </div>
                          <div className="spectrum-box">
                            <div className="spectrum-label">FFT Magnitude Spectrum (BONE)</div>
                            <img
                              src={`data:image/jpeg;base64,${fftResult.magnitude_b64}`}
                              alt="FFT Magnitude Spectrum"
                              className="spectrum-img"
                              style={{ borderColor: 'var(--accent)' }}
                            />
                          </div>
                          <div className="spectrum-box">
                            <div className="spectrum-label">Log Power Spectrum (JET)</div>
                            <img
                              src={`data:image/jpeg;base64,${fftResult.log_spectrum_b64}`}
                              alt="Log Power Spectrum"
                              className="spectrum-img"
                              style={{ borderColor: 'var(--danger)' }}
                            />
                          </div>
                        </div>
                      </div>

                      {/* Radial Spectrum Chart */}
                      {fftF?.radial_spectrum && (
                        <div className="card">
                          <RadialChart spectrum={fftF.radial_spectrum} />
                        </div>
                      )}

                      {/* Extracted Features */}
                      {fftF && (
                        <div className="card">
                          <h4 style={{ marginBottom: '1rem', color: 'var(--text-primary)' }}>Quantitative Spectral Descriptors</h4>

                          <div className="feat-group-label">Summary Statistics</div>
                          <FeatureBar label="Mean Log Magnitude" value={fftF.mean_magnitude} max={20} color="var(--accent)" tooltip="Average log-magnitude across all frequency bins" />
                          <FeatureBar label="Std Log Magnitude" value={fftF.std_magnitude} max={10} color="var(--accent)" tooltip="Standard deviation of log-magnitude spectrum" />
                          <FeatureBar label="Spectral Entropy" value={fftF.spectral_entropy} max={1} color="var(--warning)" tooltip="Shannon entropy of normalized power (0=concentrated, 1=uniform)" />

                          <div className="feat-group-label" style={{ marginTop: '1.25rem' }}>Energy Band Distribution</div>
                          <FeatureBar label="Low-Frequency Energy (0–10%)" value={fftF.low_energy} max={1} color="var(--success)" tooltip="Fraction of spectral energy in low frequencies" />
                          <FeatureBar label="Mid-Frequency Energy (10–40%)" value={fftF.mid_energy} max={1} color="var(--accent)" tooltip="Fraction of spectral energy in mid frequencies" />
                          <FeatureBar label="High-Frequency Energy (40–100%)" value={fftF.high_energy} max={1} color="var(--danger)" tooltip="Fraction of spectral energy in high frequencies" />

                          {/* Energy Segmented Bar */}
                          <div className="energy-dist" style={{ marginTop: '1.25rem' }}>
                            <div className="energy-bar-wrap">
                              <div className="energy-seg" style={{ width: `${fftF.low_energy * 100}%`, background: 'var(--success)' }} title={`Low: ${(fftF.low_energy*100).toFixed(1)}%`} />
                              <div className="energy-seg" style={{ width: `${fftF.mid_energy * 100}%`, background: 'var(--accent)' }} title={`Mid: ${(fftF.mid_energy*100).toFixed(1)}%`} />
                              <div className="energy-seg" style={{ width: `${fftF.high_energy * 100}%`, background: 'var(--danger)' }} title={`High: ${(fftF.high_energy*100).toFixed(1)}%`} />
                            </div>
                            <div className="energy-labels">
                              <span className="text-success">Low: {(fftF.low_energy*100).toFixed(1)}%</span>
                              <span className="text-accent">Mid: {(fftF.mid_energy*100).toFixed(1)}%</span>
                              <span className="text-danger">High: {(fftF.high_energy*100).toFixed(1)}%</span>
                            </div>
                          </div>
                        </div>
                      )}

                      {/* Technical Metadata */}
                      {fftF && (
                        <div className="card meta-card">
                          <div className="meta-row"><span>File Name</span><span>{fftResult.filename}</span></div>
                          <div className="meta-row"><span>Input Matrix</span><span>{fftResult.image_size?.join(' × ')} px</span></div>
                          <div className="meta-row"><span>Feature Dimensionality</span><span>{fftF.feature_dim} descriptors</span></div>
                          <div className="meta-row"><span>Azimuthal Resolution</span><span>64 radial bins</span></div>
                        </div>
                      )}
                    </div>
                  )}
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
          VeriFrame combines spatial CNN representations with dual frequency-domain features (FFT and DWT) to establish a comprehensive defense against modern generative models.
        </p>

        <div className="card architecture-card">
          <div className="arch-flow">
            <div className="arch-box">
              <span className="arch-icon">🧠</span>
              <strong>Spatial CNN</strong>
              <small>512-dim ResNet18 Embeddings</small>
            </div>
            <div className="arch-plus">+</div>
            <div className={`arch-box ${fftResult ? 'arch-active' : ''}`}>
              <span className="arch-icon">📡</span>
              <strong>2D FFT Spectrum</strong>
              <small>70-dim Frequency Vector</small>
            </div>
            <div className="arch-plus">+</div>
            <div className={`arch-box ${dwtResult ? 'arch-active' : ''}`}>
              <span className="arch-icon">🌊</span>
              <strong>DWT Wavelets</strong>
              <small>28-dim Multi-scale Sub-bands</small>
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
