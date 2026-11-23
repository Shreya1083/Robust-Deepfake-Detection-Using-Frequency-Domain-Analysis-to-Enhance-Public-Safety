import './FrequencyAnalysis.css'

const fftPoints = [
  { title: 'What is FFT?', body: 'The Fast Fourier Transform decomposes an image into its frequency components. High-frequency components correspond to edges and fine textures; low-frequency components correspond to overall structure.' },
  { title: 'Why FFT for Deepfakes?', body: 'GAN generators upsample features using transposed convolutions or nearest-neighbour resizing, which introduces regular spectral artefacts — grid-like patterns in the frequency domain that are absent in real images.' },
  { title: 'What we extract', body: 'Log-scaled magnitude spectrum of the 2D FFT. The azimuthally averaged radial spectrum is converted into a 1D feature vector and concatenated with the CNN spatial embedding for Phase 6 fusion.' },
]

const dwtPoints = [
  { title: 'What is DWT?', body: 'The Discrete Wavelet Transform decomposes an image into approximation (LL) and detail sub-bands (LH, HL, HH) at multiple scales, capturing edge, texture and structural information.' },
  { title: 'Why DWT for Deepfakes?', body: 'Deepfake faces often show blending seams, boundary artefacts, and inconsistent texture at different scales. DWT detail sub-bands (especially HH) make these visible even when spatial inspection fails.' },
  { title: 'What we extract', body: "Energy statistics (mean, variance, kurtosis) from each wavelet sub-band at multiple decomposition levels. These form a compact feature vector that complements both CNN and FFT representations." },
]

export default function FrequencyAnalysis() {
  return (
    <div className="page-container">
      <div className="section-header">
        <div className="label">Phase 4 & 5 — Coming Soon</div>
        <h1>Frequency Analysis</h1>
        <p style={{ marginTop: '0.5rem' }}>
          Understanding FFT and DWT — the frequency-domain backbone of robust deepfake detection.
        </p>
      </div>

      <div className="alert alert-warning" style={{ marginBottom: '2rem' }}>
        🚧 Frequency visualisations will be integrated in Phase 4 (FFT) and Phase 5 (DWT).
        Upload an image on the Image Detection page to see spatial CNN results now.
      </div>

      {/* FFT */}
      <section className="freq-section">
        <div className="freq-header">
          <div className="freq-icon">📡</div>
          <div>
            <div className="label">Fast Fourier Transform</div>
            <h2>FFT Frequency Analysis</h2>
          </div>
        </div>
        <div className="grid-3" style={{ marginTop: '1.25rem' }}>
          {fftPoints.map(p => (
            <div key={p.title} className="card">
              <h4 className="text-accent">{p.title}</h4>
              <p style={{ marginTop: '0.5rem' }}>{p.body}</p>
            </div>
          ))}
        </div>
        <div className="card placeholder-viz" style={{ marginTop: '1.25rem' }}>
          <div className="viz-placeholder">
            <span className="viz-icon">📊</span>
            <p>FFT magnitude spectrum will appear here (Phase 4)</p>
          </div>
        </div>
      </section>

      {/* DWT */}
      <section className="freq-section">
        <div className="freq-header">
          <div className="freq-icon">🌊</div>
          <div>
            <div className="label">Discrete Wavelet Transform</div>
            <h2>DWT Wavelet Analysis</h2>
          </div>
        </div>
        <div className="grid-3" style={{ marginTop: '1.25rem' }}>
          {dwtPoints.map(p => (
            <div key={p.title} className="card">
              <h4 className="text-accent">{p.title}</h4>
              <p style={{ marginTop: '0.5rem' }}>{p.body}</p>
            </div>
          ))}
        </div>
        <div className="card placeholder-viz" style={{ marginTop: '1.25rem' }}>
          <div className="viz-placeholder">
            <span className="viz-icon">🌊</span>
            <p>Wavelet sub-band decomposition (LL · LH · HL · HH) will appear here (Phase 5)</p>
          </div>
        </div>
      </section>

      {/* Fusion preview */}
      <section className="freq-section">
        <h2>🔗 Phase 6 — Feature Fusion</h2>
        <div className="card fusion-card">
          <div className="fusion-diagram">
            <div className="fusion-box">🧠 CNN<br/><small>512-dim</small></div>
            <div className="fusion-plus">+</div>
            <div className="fusion-box">📡 FFT<br/><small>256-dim</small></div>
            <div className="fusion-plus">+</div>
            <div className="fusion-box">🌊 DWT<br/><small>128-dim</small></div>
            <div className="fusion-arrow">→</div>
            <div className="fusion-box fusion-output">🛡️ Fused<br/><small>MLP → REAL/FAKE</small></div>
          </div>
          <p style={{ marginTop: '1rem' }}>
            Phase 6 will concatenate all three feature vectors and train a lightweight MLP fusion
            head for the final robust prediction.
          </p>
        </div>
      </section>
    </div>
  )
}
