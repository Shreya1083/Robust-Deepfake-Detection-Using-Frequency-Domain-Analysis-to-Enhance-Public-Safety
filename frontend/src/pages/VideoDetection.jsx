import { useState } from 'react'
import { Link } from 'react-router-dom'
import { detectVideo } from '../services/api'
import './VideoDetection.css'

export default function VideoDetection() {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleFile = (e) => {
    const f = e.target.files[0]
    if (!f) return
    setFile(f)
    setResult(null)
    setError(null)
    const url = URL.createObjectURL(f)
    setPreviewUrl(url)
  }

  const analyze = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const fd = new FormData()
      fd.append('file', file)
      const { data } = await detectVideo(fd)
      setResult(data)
    } catch (err) {
      const respData = err.response?.data
      let errText = 'Video analysis failed.'
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
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl)
    }
    setFile(null)
    setPreviewUrl(null)
    setResult(null)
    setError(null)
  }

  return (
    <div className="page-container video-detection-page">
      <div className="section-header">
        <span className="badge badge-teal">Temporal Sequence Analysis</span>
        <h1>Video Deepfake Detection</h1>
        <p>
          Analyze video sequences for temporal inconsistency, frame-level facial tampering,
          and artificial blending boundaries across multi-frame temporal clips.
        </p>
      </div>

      <div className="video-layout">
        {/* Upload & Preview Card */}
        <div className="card video-upload-card">
          <div className="card-header">
            <h3>Upload Video for Analysis</h3>
            <p className="text-muted">Supported formats: MP4, AVI, MOV, WebM</p>
          </div>

          <div className="video-dropzone">
            {previewUrl ? (
              <div className="video-player-container">
                <video src={previewUrl} controls className="video-preview-player" />
                <div className="video-meta-bar">
                  <span className="video-name">{file?.name}</span>
                  <span className="video-size">{(file?.size / (1024 * 1024)).toFixed(2)} MB</span>
                </div>
              </div>
            ) : (
              <div className="video-placeholder">
                <div className="video-icon">🎬</div>
                <h4>Select or drop a video file</h4>
                <p className="text-muted">High-frame-rate or portrait facial videos work best</p>
                <div className="upload-chips">
                  <span>MP4</span>
                  <span>AVI</span>
                  <span>MOV</span>
                  <span>WebM</span>
                </div>
              </div>
            )}
            <input
              type="file"
              accept=".mp4,.avi,.mov,.webm"
              className="video-file-input"
              onChange={handleFile}
            />
          </div>

          <div className="video-actions">
            {file && !loading && (
              <button className="btn btn-primary" onClick={analyze}>
                🔍 Run Video Verification
              </button>
            )}
            {loading && (
              <button className="btn btn-primary" disabled>
                <span className="spinner" style={{ width: 18, height: 18, borderWidth: 2 }} />
                Verifying Video Stream…
              </button>
            )}
            {file && (
              <button className="btn btn-outline" onClick={reset}>
                ✕ Clear
              </button>
            )}
          </div>

          {error && (
            <div className="alert alert-danger" style={{ marginTop: '1.25rem' }}>
              ⚠️ {error}
            </div>
          )}
        </div>

        {/* Status & Pipeline Card */}
        <div className="card video-status-card">
          <div className="card-header">
            <h3>Temporal Detection Pipeline</h3>
            <p className="text-muted">Multi-stage video verification architecture</p>
          </div>

          <div className="pipeline-steps">
            <div className="pipeline-step">
              <div className="step-num">1</div>
              <div className="step-info">
                <h4>Uniform Keyframe Extraction</h4>
                <p>Decomposes the video into structured temporal keyframe sequences (sampled at regular intervals).</p>
              </div>
            </div>

            <div className="pipeline-step">
              <div className="step-num">2</div>
              <div className="step-info">
                <h4>Facial Landmark Alignment &amp; Tracking</h4>
                <p>Tracks facial bounding boxes and landmarks continuously across consecutive frames to verify physical continuity.</p>
              </div>
            </div>

            <div className="pipeline-step">
              <div className="step-num">3</div>
              <div className="step-info">
                <h4>Spatial-Temporal Consistency Evaluation</h4>
                <p>Evaluates eye-blink dynamics, boundary jitter, and frequency spectrum variations over time.</p>
              </div>
            </div>

            <div className="pipeline-step">
              <div className="step-num">4</div>
              <div className="step-info">
                <h4>Temporal Sequence Scoring</h4>
                <p>Aggregates frame probabilities to generate an overall authentic vs manipulated classification.</p>
              </div>
            </div>
          </div>

          {result && (
            <div className="status-banner" style={{ marginTop: '1.5rem' }}>
              <div className="status-badge-row">
                <span className="badge badge-warning">Pipeline Status</span>
                <span className="status-file">{result.filename}</span>
              </div>
              <p className="status-text">
                {result.message || result.status || 'Video received. The temporal inference sequence engine is currently undergoing validation.'}
              </p>
              <div className="status-details">
                <span>Verification: File format and stream integrity verified.</span>
                <span>Active Model: Spatial-Temporal Sequence Aggregator (in development).</span>
              </div>
            </div>
          )}

          {!result && (
            <div className="system-note" style={{ marginTop: '1.5rem' }}>
              <span className="text-muted">
                ℹ️ Uploading a video performs stream validation. To run immediate face classification, you can extract a snapshot and use the{' '}
                <Link to="/image-detection" style={{ color: 'var(--accent)', fontWeight: 600 }}>
                  Image Detection Studio
                </Link>.
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
