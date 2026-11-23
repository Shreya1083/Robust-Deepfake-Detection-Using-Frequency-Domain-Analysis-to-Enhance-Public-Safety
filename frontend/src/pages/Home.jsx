import { Link } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import './Home.css'

export default function Home() {
  const { user } = useAuth()

  return (
    <div className="home-page">
      {/* Hero Section */}
      <section className="hero-section">
        <div className="page-container hero-container">
          <div className="hero-badge badge badge-teal">
            🛡️ AI-Powered Media Integrity Platform
          </div>
          <h1 className="hero-title">
            Verify What's Real. <span className="hero-title-accent">Detect What's Fake.</span>
          </h1>
          <p className="hero-description">
            VeriFrame is an AI-powered deepfake detection system that analyzes images and videos to identify potential digital manipulation using spatial and frequency-domain features.
          </p>
          <div className="hero-buttons">
            <Link to={user ? "/dashboard" : "/signup"} className="btn btn-primary btn-hero">
              {user ? "Go to Dashboard" : "Get Started"}
            </Link>
            {!user ? (
              <Link to="/signin" className="btn btn-outline btn-hero">
                Sign In
              </Link>
            ) : (
              <Link to="/image-detection" className="btn btn-outline btn-hero">
                Analyze Media
              </Link>
            )}
          </div>
        </div>
      </section>

      <div className="page-container">
        {/* About VeriFrame Section */}
        <section className="home-section" id="about-veriframe">
          <div className="section-header text-center">
            <div className="label">Overview</div>
            <h2>About VeriFrame</h2>
            <p>Understanding digital manipulation and our multi-layered defense strategy.</p>
          </div>

          <div className="about-grid grid-2">
            <div className="card about-card">
              <div className="about-card-icon">🎭</div>
              <h3>What Are Deepfakes?</h3>
              <p>
                Deepfakes are synthetic media in which a person's likeness, facial features, or voice are digitally replaced or altered using deep generative neural networks. Modern generative techniques produce hyper-realistic alterations that easily deceive the human eye.
              </p>
            </div>

            <div className="card about-card">
              <div className="about-card-icon">⚖️</div>
              <h3>Why Deepfake Detection Matters</h3>
              <p>
                Unchecked manipulated media poses severe threats to personal identity, public discourse, legal evidence integrity, and democratic processes. Reliable automated detection provides a crucial safeguard for individuals, journalists, and security teams.
              </p>
            </div>

            <div className="card about-card">
              <div className="about-card-icon">🔬</div>
              <h3>How VeriFrame Analyzes Digital Media</h3>
              <p>
                VeriFrame inspects media beyond surface pixels. While visual manipulation can be blended seamlessly in the spatial image domain, the mathematical operations used by generative algorithms leave telltale signatures that are revealed through computational analysis.
              </p>
            </div>

            <div className="card about-card">
              <div className="about-card-icon">⚡</div>
              <h3>AI & Frequency-Domain Synergy</h3>
              <p>
                By combining convolutional neural networks with Fourier (FFT) and Wavelet (DWT) frequency decompositions, VeriFrame inspects spectral energy distribution, boundary blending gradients, and high-frequency synthesis noise across facial regions.
              </p>
            </div>
          </div>
        </section>

        {/* How It Works Section */}
        <section className="home-section" id="how-it-works">
          <div className="section-header text-center">
            <div className="label">Workflow</div>
            <h2>How It Works</h2>
            <p>Three straightforward steps to analyze media for signs of manipulation.</p>
          </div>

          <div className="steps-container">
            <div className="step-card">
              <div className="step-number">1</div>
              <div className="step-content">
                <h3>Upload Media</h3>
                <p>Upload a suspicious facial image or video directly through our secure interface.</p>
              </div>
            </div>

            <div className="step-connector" />

            <div className="step-card">
              <div className="step-number">2</div>
              <div className="step-content">
                <h3>Multi-Domain AI Analysis</h3>
                <p>Our pipeline inspects spatial facial features alongside frequency-domain spectral characteristics.</p>
              </div>
            </div>

            <div className="step-connector" />

            <div className="step-card">
              <div className="step-number">3</div>
              <div className="step-content">
                <h3>View Results & Reports</h3>
                <p>Receive clear REAL or DEEPFAKE classifications with exact model probabilities and diagnostic reports.</p>
              </div>
            </div>
          </div>
        </section>

        {/* Key Capabilities Section */}
        <section className="home-section" id="capabilities">
          <div className="section-header text-center">
            <div className="label">Core Architecture</div>
            <h2>Key Capabilities</h2>
            <p>Comprehensive inspection tools engineered for digital media forensics.</p>
          </div>

          <div className="capabilities-grid grid-3">
            <div className="card capability-card">
              <div className="capability-icon">🖼️</div>
              <h3>Image Deepfake Detection</h3>
              <p>Detect subtle alterations, face swaps, and full synthetic face generation in uploaded images.</p>
            </div>

            <div className="card capability-card">
              <div className="capability-icon">🎥</div>
              <h3>Video Deepfake Detection</h3>
              <p>Frame-by-frame extraction architecture designed to analyze temporal consistency and facial dynamics.</p>
            </div>

            <div className="card capability-card">
              <div className="capability-icon">🧠</div>
              <h3>CNN Spatial Feature Analysis</h3>
              <p>Deep residual convolutional network (ResNet18) trained to identify structural facial manipulation artifacts.</p>
            </div>

            <div className="card capability-card">
              <div className="capability-icon">📊</div>
              <h3>FFT Frequency Analysis</h3>
              <p>Fast Fourier Transform computes radial power spectrums and energy bands to expose unnatural GAN frequency signatures.</p>
            </div>

            <div className="card capability-card">
              <div className="capability-icon">🌊</div>
              <h3>DWT Multi-Scale Wavelets</h3>
              <p>Discrete Wavelet Transform isolates multi-resolution horizontal, vertical, and diagonal detail anomalies.</p>
            </div>

            <div className="card capability-card">
              <div className="capability-icon">📋</div>
              <h3>Forensic Reports & History</h3>
              <p>Audit trail of all previous detection activity, structured probability distributions, and exportable findings.</p>
            </div>
          </div>
        </section>

        {/* Call to Action Banner */}
        <section className="cta-section card">
          <div className="cta-content">
            <h2>Ready to Verify Media Authenticity?</h2>
            <p>Start analyzing images and videos with VeriFrame's multi-layered detection platform.</p>
          </div>
          <div className="cta-actions">
            <Link to={user ? "/image-detection" : "/signup"} className="btn btn-primary btn-hero">
              {user ? "Analyze Image Now" : "Create Free Account"}
            </Link>
          </div>
        </section>
      </div>
    </div>
  )
}
