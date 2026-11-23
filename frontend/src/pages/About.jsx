import './About.css'

const objectives = [
  'Develop a CNN-based spatial deepfake detector using pretrained ResNet18.',
  'Implement FFT-based frequency-domain feature extraction for deepfake signatures.',
  'Apply Discrete Wavelet Transform (DWT) for multi-scale texture analysis.',
  'Fuse spatial, frequency, and wavelet features for improved accuracy.',
  'Evaluate on FaceForensics++ and Celeb-DF v2 benchmarks.',
  'Build a deployable web application for real-time deepfake detection.',
]

const limitations = [
  { title: 'Visual-Only Methods', desc: 'Traditional visual inspection fails as GANs improve realism.' },
  { title: 'Single-Domain Detection', desc: 'Most models only use spatial domain, missing frequency artifacts.' },
  { title: 'Poor Generalisation', desc: 'Models trained on one dataset often fail on unseen manipulation types.' },
  { title: 'No Explainability', desc: 'Black-box detectors provide no insight into why a prediction was made.' },
]

const methodology = [
  { phase: 'Phase 0', title: 'Project Setup', desc: 'Environment, directory structure, config and version control.' },
  { phase: 'Phase 1', title: 'Streamlit Prototype', desc: 'Rapid UI prototyping with all 6 application pages.' },
  { phase: 'Phase 2', title: 'Dataset Preparation', desc: 'FaceForensics++ frame extraction, face detection, preprocessing and splits.' },
  { phase: 'Phase 3', title: 'CNN Spatial Detection', desc: 'ResNet18 binary classifier — REAL vs DEEPFAKE (current phase).' },
  { phase: 'Phase 3A', title: 'Web Application', desc: 'React + FastAPI production-ready web app replacing Streamlit.' },
  { phase: 'Phase 4', title: 'FFT Analysis', desc: 'Frequency domain feature extraction using Fast Fourier Transform.' },
  { phase: 'Phase 5', title: 'DWT Analysis', desc: 'Wavelet decomposition for multi-scale texture analysis.' },
  { phase: 'Phase 6', title: 'Feature Fusion', desc: 'Combine CNN + FFT + DWT features for final robust detection.' },
]

export default function About() {
  return (
    <div className="page-container">
      <div className="section-header">
        <div className="label">About</div>
        <h1>Project Overview</h1>
        <p style={{ marginTop: '0.75rem', maxWidth: 700 }}>
          A research-grade deepfake detection system that combines spatial, frequency,
          and wavelet domain analysis for robust, explainable predictions.
        </p>
      </div>

      {/* Objectives */}
      <section className="about-section">
        <h2>🎯 Project Objectives</h2>
        <div className="card">
          <ul className="objective-list">
            {objectives.map((o, i) => (
              <li key={i}>
                <span className="obj-num">{i + 1}</span>
                <span>{o}</span>
              </li>
            ))}
          </ul>
        </div>
      </section>

      {/* Limitations */}
      <section className="about-section">
        <h2>⚠️ Existing System Limitations</h2>
        <div className="grid-2">
          {limitations.map(l => (
            <div key={l.title} className="card">
              <h4 className="text-danger">{l.title}</h4>
              <p style={{ marginTop: '0.5rem' }}>{l.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Proposed System */}
      <section className="about-section">
        <h2>🚀 Proposed System</h2>
        <div className="card proposed-card">
          <p>
            Our system addresses each limitation by combining <strong className="text-accent">three complementary detection modalities</strong>:
          </p>
          <div className="modality-grid" style={{ marginTop: '1.25rem' }}>
            <div className="modality">
              <div className="modality-icon">🧠</div>
              <strong>Spatial (CNN)</strong>
              <p>ResNet18 detects texture and structural anomalies in pixel space.</p>
            </div>
            <div className="modality">
              <div className="modality-icon">📡</div>
              <strong>Frequency (FFT)</strong>
              <p>Fourier analysis identifies spectral inconsistencies introduced by GAN upsampling.</p>
            </div>
            <div className="modality">
              <div className="modality-icon">🌊</div>
              <strong>Wavelet (DWT)</strong>
              <p>Wavelet decomposition captures multi-scale edge and detail artifacts.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Methodology */}
      <section className="about-section">
        <h2>📋 Project Methodology</h2>
        <div className="timeline">
          {methodology.map((m, i) => (
            <div key={m.phase} className="timeline-item">
              <div className="timeline-dot" />
              <div className="timeline-content">
                <div className="badge badge-info" style={{ marginBottom: '0.4rem' }}>{m.phase}</div>
                <h4>{m.title}</h4>
                <p>{m.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Expected Outcomes */}
      <section className="about-section">
        <h2>✅ Expected Outcomes</h2>
        <div className="card">
          <ul className="outcome-list">
            {[
              'Binary deepfake classifier with >90% accuracy on FaceForensics++ test split.',
              'Multi-modal feature fusion improving robustness over single-modality baselines.',
              'Generalisation evaluation on Celeb-DF v2 (independent, never used for training).',
              'Production web application with React frontend and FastAPI backend.',
              'Full explainability via frequency visualisations (FFT magnitude maps, wavelet bands).',
            ].map((o, i) => (
              <li key={i}><span className="text-accent">✓</span> {o}</li>
            ))}
          </ul>
        </div>
      </section>
    </div>
  )
}
