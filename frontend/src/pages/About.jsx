import { Link } from 'react-router-dom'
import './About.css'

const corePillars = [
  {
    icon: '🧠',
    title: 'CNN Spatial Analysis',
    subtitle: 'Structural & Pixel-Level Forensics',
    desc: 'Deep residual neural network architectures (ResNet18) analyze subtle blending boundaries, unnatural skin smoothing, warping artifacts, and illumination inconsistencies across facial landmarks.',
  },
  {
    icon: '📡',
    title: 'FFT Frequency Analysis',
    subtitle: 'Global Spectral Domain Signatures',
    desc: 'Fast Fourier Transform converts facial crops into 2D magnitude spectrums. Azimuthally averaged radial power spectrums expose the unnatural high-frequency periodic patterns and power drop-offs characteristic of GAN generators.',
  },
  {
    icon: '🌊',
    title: 'DWT Wavelet Decomposition',
    subtitle: 'Multi-Resolution Spatial-Frequency Analysis',
    desc: '2D Discrete Wavelet Transform breaks media into approximation (LL) and detail sub-bands (LH, HL, HH), isolating horizontal lip borders, vertical nasal contours, and high-frequency diagonal synthesis noise.',
  },
]

const systemCapabilities = [
  {
    title: 'Comprehensive Image Verification',
    desc: 'Inspects user-uploaded digital portraits, headshots, and media stills with automated facial isolation and dual-domain inference.',
  },
  {
    title: 'Video Architecture & Temporal Coherence',
    desc: 'Designed around frame-level extraction to detect flickering, landmark drift, and inter-frame temporal anomalies.',
  },
  {
    title: 'Auditable Forensic History',
    desc: 'Maintains authenticated detection logs with exact model probabilities and exportable diagnostic summaries for compliance and forensic review.',
  },
  {
    title: 'Privacy-First Architecture',
    desc: 'Secure authentication, strict access control, and encrypted session management ensure that scans remain strictly private to the user.',
  },
]

export default function About() {
  return (
    <div className="page-container about-page">
      <div className="section-header">
        <div className="label">Platform Overview</div>
        <h1>About VeriFrame</h1>
        <p style={{ maxWidth: 720, marginTop: '0.5rem' }}>
          An advanced AI-powered forensic system designed to authenticate digital media and safeguard public information against sophisticated synthetic manipulations.
        </p>
      </div>

      {/* Mission & Problem Statement */}
      <section className="about-section">
        <div className="card mission-card">
          <div className="mission-content">
            <span className="badge badge-teal" style={{ marginBottom: '0.75rem' }}>Our Mission</span>
            <h2>Defending Media Trust in the Era of Generative AI</h2>
            <p style={{ marginTop: '0.75rem' }}>
              The rapid proliferation of generative adversarial networks (GANs) and diffusion models has made the creation of photorealistic fake media accessible at scale. Deepfakes are increasingly exploited for identity theft, financial extortion, disinformation campaigns, and non-consensual imagery.
            </p>
            <p style={{ marginTop: '0.75rem' }}>
              Standard visual review is no longer reliable because synthetic models are specifically trained to fool human observers. VeriFrame bridges this critical vulnerability through multi-domain computational analysis, inspecting underlying mathematical and frequency signatures that algorithms cannot hide.
            </p>
          </div>
        </div>
      </section>

      {/* Multi-Domain Forensic Architecture */}
      <section className="about-section">
        <div className="section-header">
          <div className="label">Forensic Approach</div>
          <h2>Multi-Modal Detection Architecture</h2>
          <p>VeriFrame combines complementary forensic techniques to achieve robust detection across varied manipulation types.</p>
        </div>

        <div className="grid-3">
          {corePillars.map(p => (
            <div key={p.title} className="card pillar-card">
              <div className="pillar-icon">{p.icon}</div>
              <h3>{p.title}</h3>
              <div className="pillar-sub">{p.subtitle}</div>
              <p>{p.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* System Capabilities Grid */}
      <section className="about-section">
        <div className="section-header">
          <div className="label">Capabilities</div>
          <h2>Key Platform Features</h2>
        </div>

        <div className="grid-2">
          {systemCapabilities.map(c => (
            <div key={c.title} className="card capability-item-card">
              <h4>{c.title}</h4>
              <p style={{ marginTop: '0.4rem' }}>{c.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Call to action */}
      <section className="about-cta card">
        <div>
          <h3>Start Verifying Media Authenticity</h3>
          <p style={{ marginTop: '0.25rem' }}>Experience multi-modal deepfake detection with VeriFrame today.</p>
        </div>
        <Link to="/image-detection" className="btn btn-primary">
          Launch Image Detection
        </Link>
      </section>
    </div>
  )
}
