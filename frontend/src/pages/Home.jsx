import { Link } from 'react-router-dom'
import './Home.css'

const features = [
  { icon: '🧠', title: 'CNN Spatial Analysis', desc: 'ResNet18 extracts deep spatial patterns from face images to detect subtle manipulation artifacts.' },
  { icon: '📡', title: 'FFT Frequency Analysis', desc: 'Fast Fourier Transform reveals unnatural frequency signatures that GAN-generated images introduce.' },
  { icon: '🌊', title: 'DWT Wavelet Analysis', desc: 'Discrete Wavelet Transform captures multi-scale texture anomalies invisible in pixel space.' },
]

const stats = [
  { label: 'Model Architecture', value: 'ResNet18' },
  { label: 'Input Resolution', value: '224 × 224' },
  { label: 'Classes', value: 'Real / Deepfake' },
  { label: 'Framework', value: 'PyTorch' },
]

export default function Home() {
  return (
    <div className="home">
      {/* Hero */}
      <section className="hero">
        <div className="hero-inner page-container">
          <div className="hero-badge badge badge-info">Research Project · 2024–25</div>
          <h1 className="hero-title">
            Robust Deepfake Detection<br />
            <span className="text-accent">Using Frequency Domain Analysis</span>
          </h1>
          <p className="hero-sub">
            A multi-modal AI system combining CNN spatial features, FFT frequency analysis,
            and DWT wavelet decomposition to detect AI-generated face manipulations with
            high accuracy — enhancing public safety and media integrity.
          </p>
          <div className="hero-actions">
            <Link to="/image-detection" className="btn btn-primary">🖼️ Detect Image</Link>
            <Link to="/about" className="btn btn-outline">Learn More</Link>
          </div>
        </div>
      </section>

      <div className="page-container">
        {/* Problem Statement */}
        <section className="section-block">
          <div className="section-header">
            <div className="label">Problem Statement</div>
            <h2>Why Deepfake Detection Matters</h2>
          </div>
          <div className="card problem-card">
            <p>
              Deepfakes — hyper-realistic AI-generated videos and images — are increasingly used
              for identity fraud, misinformation campaigns, and non-consensual intimate imagery.
              As generative models advance, visual inspection alone becomes unreliable.
              Automated detection is now a critical tool for law enforcement, journalism,
              and platform safety teams.
            </p>
            <div className="problem-stats grid-3" style={{ marginTop: '1.5rem' }}>
              {stats.map(s => (
                <div key={s.label} className="stat-item">
                  <div className="stat-value">{s.value}</div>
                  <div className="stat-label">{s.label}</div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Detection Approach */}
        <section className="section-block">
          <div className="section-header">
            <div className="label">Detection Approach</div>
            <h2>Multi-Modal Feature Fusion</h2>
          </div>
          <div className="grid-3">
            {features.map(f => (
              <div key={f.title} className="card feature-card">
                <div className="feature-icon">{f.icon}</div>
                <h3>{f.title}</h3>
                <p>{f.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Pipeline */}
        <section className="section-block">
          <div className="section-header">
            <div className="label">Detection Pipeline</div>
            <h2>How It Works</h2>
          </div>
          <div className="pipeline">
            {['Upload Image', 'Face Detection', 'CNN Inference', 'Frequency Analysis', 'Fusion & Result'].map((step, i) => (
              <div key={step} className="pipeline-step">
                <div className="pipeline-num">{i + 1}</div>
                <div className="pipeline-label">{step}</div>
                {i < 4 && <div className="pipeline-arrow">→</div>}
              </div>
            ))}
          </div>
        </section>

        {/* CTA */}
        <section className="cta-section">
          <div className="card cta-card">
            <h2>Ready to Detect?</h2>
            <p>Upload an image or video to run CNN-based deepfake detection.</p>
            <div className="hero-actions">
              <Link to="/image-detection" className="btn btn-primary">🖼️ Image Detection</Link>
              <Link to="/video-detection" className="btn btn-outline">🎬 Video Detection</Link>
            </div>
          </div>
        </section>
      </div>
    </div>
  )
}
