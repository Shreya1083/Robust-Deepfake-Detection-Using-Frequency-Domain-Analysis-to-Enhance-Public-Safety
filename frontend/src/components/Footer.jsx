import { Link } from 'react-router-dom'
import './Footer.css'

export default function Footer() {
  return (
    <footer className="footer">
      <div className="footer-inner">
        <div className="footer-brand-section">
          <div className="footer-brand">
            <span className="footer-brand-icon">🛡️</span>
            <span className="footer-brand-name">VeriFrame</span>
          </div>
          <p className="footer-tagline">AI-Powered Deepfake Detection System</p>
          <p className="footer-desc">
            Advanced forensic verification combining convolutional spatial feature analysis and
            frequency-domain analysis to protect digital media integrity and enhance public safety.
          </p>
        </div>

        <div className="footer-links-grid">
          <div className="footer-col">
            <h4>Forensic Tools</h4>
            <ul>
              <li><Link to="/image-detection">Image Detection</Link></li>
              <li><Link to="/video-detection">Video Detection</Link></li>
              <li><Link to="/frequency-analysis">Frequency Analysis</Link></li>
              <li><Link to="/dashboard">User Dashboard</Link></li>
            </ul>
          </div>

          <div className="footer-col">
            <h4>Platform</h4>
            <ul>
              <li><Link to="/">Home</Link></li>
              <li><Link to="/about">System Architecture</Link></li>
              <li><Link to="/history">Inspection History</Link></li>
              <li><Link to="/profile">User Profile</Link></li>
            </ul>
          </div>

          <div className="footer-col">
            <h4>Technology</h4>
            <ul>
              <li><span>PyTorch &amp; ResNet18</span></li>
              <li><span>2D Fast Fourier Transform</span></li>
              <li><span>FastAPI Backend</span></li>
              <li><span>MongoDB Secure Storage</span></li>
            </ul>
          </div>
        </div>
      </div>

      <div className="footer-bottom">
        <p className="footer-copy">
          &copy; {new Date().getFullYear()} VeriFrame. All rights reserved. Built for digital trust and public safety.
        </p>
      </div>
    </footer>
  )
}
