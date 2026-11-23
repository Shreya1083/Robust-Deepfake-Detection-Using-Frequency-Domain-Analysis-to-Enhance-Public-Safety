import { useState, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import {
  apiGetHistory,
  apiGetHistoryStats,
  apiDeleteHistoryItem,
  detectImage,
  detectVideo,
} from '../services/api'
import './Dashboard.css'

export default function Dashboard() {
  const { user } = useAuth()
  const navigate = useNavigate()

  // Stats state
  const [stats, setStats] = useState({
    total_analyzed: 0,
    images_analyzed: 0,
    videos_analyzed: 0,
    real_count: 0,
    deepfake_count: 0,
    recent_activity_count: 0,
  })
  const [statsLoading, setStatsLoading] = useState(true)

  // History state
  const [historyItems, setHistoryItems] = useState([])
  const [historyLoading, setHistoryLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const [mediaFilter, setMediaFilter] = useState('all')
  const [resultFilter, setResultFilter] = useState('all')

  // Selected report modal state
  const [selectedReport, setSelectedReport] = useState(null)

  // In-card Image Quick Action State
  const [quickImageFile, setQuickImageFile] = useState(null)
  const [quickImagePreview, setQuickImagePreview] = useState(null)
  const [quickImageLoading, setQuickImageLoading] = useState(false)
  const [quickImageResult, setQuickImageResult] = useState(null)
  const [quickImageError, setQuickImageError] = useState(null)

  // In-card Video Quick Action State
  const [quickVideoFile, setQuickVideoFile] = useState(null)
  const [quickVideoLoading, setQuickVideoLoading] = useState(false)
  const [quickVideoResult, setQuickVideoResult] = useState(null)
  const [quickVideoError, setQuickVideoError] = useState(null)

  // Fetch summary metrics & recent history from user's actual MongoDB records
  const loadDashboardData = async () => {
    try {
      setStatsLoading(true)
      const statsRes = await apiGetHistoryStats()
      if (statsRes.data?.success) {
        setStats(statsRes.data)
      }
    } catch (err) {
      console.warn('Could not load user detection stats:', err)
    } finally {
      setStatsLoading(false)
    }

    try {
      setHistoryLoading(true)
      const histRes = await apiGetHistory({
        search: searchQuery || undefined,
        media_type: mediaFilter,
        prediction: resultFilter,
        limit: 10,
      })
      if (histRes.data?.items) {
        setHistoryItems(histRes.data.items)
      }
    } catch (err) {
      console.warn('Could not load detection history:', err)
    } finally {
      setHistoryLoading(false)
    }
  }

  useEffect(() => {
    loadDashboardData()
  }, [searchQuery, mediaFilter, resultFilter])

  // Delete a history record
  const handleDeleteHistory = async (id, e) => {
    e.stopPropagation()
    if (!window.confirm('Are you sure you want to delete this detection record?')) return
    try {
      await apiDeleteHistoryItem(id)
      setHistoryItems(prev => prev.filter(item => item.id !== id))
      if (selectedReport?.id === id) setSelectedReport(null)
      // refresh stats
      const statsRes = await apiGetHistoryStats()
      if (statsRes.data?.success) setStats(statsRes.data)
    } catch (err) {
      alert('Failed to delete history record: ' + (err.response?.data?.detail || err.message))
    }
  }

  // Quick image selection & upload in dashboard card
  const handleQuickImageSelect = (e) => {
    const f = e.target.files?.[0]
    if (!f) return
    setQuickImageFile(f)
    setQuickImageResult(null)
    setQuickImageError(null)
    const reader = new FileReader()
    reader.onload = ev => setQuickImagePreview(ev.target.result)
    reader.readAsDataURL(f)
  }

  const handleQuickImageAnalyze = async () => {
    if (!quickImageFile) return
    setQuickImageLoading(true)
    setQuickImageError(null)
    try {
      const fd = new FormData()
      fd.append('file', quickImageFile)
      const { data } = await detectImage(fd)
      setQuickImageResult(data)
      // Refresh user stats & history list
      loadDashboardData()
    } catch (err) {
      setQuickImageError(err.response?.data?.detail || err.message || 'Image detection failed.')
    } finally {
      setQuickImageLoading(false)
    }
  }

  // Quick video selection & upload in dashboard card
  const handleQuickVideoSelect = (e) => {
    const f = e.target.files?.[0]
    if (!f) return
    setQuickVideoFile(f)
    setQuickVideoResult(null)
    setQuickVideoError(null)
  }

  const handleQuickVideoAnalyze = async () => {
    if (!quickVideoFile) return
    setQuickVideoLoading(true)
    setQuickVideoError(null)
    try {
      const fd = new FormData()
      fd.append('file', quickVideoFile)
      const { data } = await detectVideo(fd)
      setQuickVideoResult(data)
    } catch (err) {
      setQuickVideoError(err.response?.data?.detail || err.message || 'Video analysis failed.')
    } finally {
      setQuickVideoLoading(false)
    }
  }

  return (
    <div className="page-container dashboard-page">
      {/* Welcome Section */}
      <section className="welcome-banner card">
        <div className="welcome-content">
          <div className="badge badge-teal" style={{ marginBottom: '0.5rem' }}>
            Authenticated Session
          </div>
          <h1>Welcome back, <span className="text-teal">{user?.full_name || user?.username || 'User'}</span></h1>
          <p className="welcome-subtitle">
            Analyze media and review your previous detection results.
          </p>
        </div>
        <div className="welcome-meta">
          <div className="user-email-tag">
            <span className="dot online" /> {user?.email}
          </div>
          <Link to="/profile" className="btn btn-outline btn-sm">
            View Profile
          </Link>
        </div>
      </section>

      {/* Main Dashboard Actions — Two Prominent Functional Cards */}
      <section className="dashboard-section">
        <div className="section-header">
          <div className="label">Forensic Actions</div>
          <h2>Media Analysis Tools</h2>
          <p>Directly analyze digital media or launch the dedicated detection environments.</p>
        </div>

        <div className="grid-2 dashboard-action-grid">
          {/* Functional Image Detection Card */}
          <div className="card action-card">
            <div className="action-card-header">
              <div className="action-card-icon">🖼️</div>
              <div>
                <h3>Image Detection</h3>
                <p>Analyze portraits for synthetic anomalies & GAN patterns</p>
              </div>
            </div>

            <div className="action-card-body">
              <div className="quick-upload-box">
                <input
                  type="file"
                  id="dashboard-image-upload"
                  accept="image/jpeg,image/png"
                  onChange={handleQuickImageSelect}
                  style={{ display: 'none' }}
                />
                <label htmlFor="dashboard-image-upload" className="quick-upload-label">
                  {quickImagePreview ? (
                    <div className="quick-preview-wrapper">
                      <img src={quickImagePreview} alt="Preview" className="quick-preview-img" />
                      <span className="quick-change-btn">Change Image</span>
                    </div>
                  ) : (
                    <div className="quick-placeholder">
                      <span className="upload-arrow">⬆️</span>
                      <span className="upload-text">Select Image to Analyze</span>
                      <span className="upload-types">JPG, PNG up to 20MB</span>
                    </div>
                  )}
                </label>
              </div>

              {quickImageFile && (
                <div className="quick-action-bar">
                  <span className="quick-filename" title={quickImageFile.name}>
                    📄 {quickImageFile.name}
                  </span>
                  <button
                    className="btn btn-teal btn-sm"
                    onClick={handleQuickImageAnalyze}
                    disabled={quickImageLoading}
                  >
                    {quickImageLoading ? (
                      <>
                        <span className="spinner spinner-sm" /> Analyzing…
                      </>
                    ) : (
                      'Start Detection'
                    )}
                  </button>
                </div>
              )}

              {quickImageError && (
                <div className="alert alert-danger" style={{ marginTop: '0.75rem', fontSize: '0.85rem' }}>
                  {quickImageError}
                </div>
              )}

              {quickImageResult && (
                <div className="quick-result-box" style={{ marginTop: '0.75rem' }}>
                  <div className="quick-result-header">
                    <span
                      className={`badge ${quickImageResult.prediction === 'REAL' ? 'badge-real' : 'badge-fake'}`}
                      style={{ fontSize: '0.85rem', padding: '0.35rem 0.85rem' }}
                    >
                      {quickImageResult.prediction === 'REAL' ? '✓ REAL' : '⚠️ DEEPFAKE'}
                    </span>
                    <span className="quick-conf">
                      Confidence: {(quickImageResult.confidence * 100).toFixed(1)}%
                    </span>
                  </div>

                  <div className="quick-probs">
                    <div className="prob-pill">
                      <span>Real:</span>
                      <strong>{quickImageResult.real_prob != null ? `${(quickImageResult.real_prob * 100).toFixed(1)}%` : '—'}</strong>
                    </div>
                    <div className="prob-pill">
                      <span>Deepfake:</span>
                      <strong>{quickImageResult.deepfake_prob != null ? `${(quickImageResult.deepfake_prob * 100).toFixed(1)}%` : '—'}</strong>
                    </div>
                  </div>

                  <div style={{ marginTop: '0.75rem', display: 'flex', gap: '0.5rem' }}>
                    <button
                      className="btn btn-outline btn-sm"
                      onClick={() => setSelectedReport({
                        filename: quickImageFile?.name || 'Uploaded Image',
                        media_type: 'image',
                        prediction: quickImageResult.prediction,
                        confidence: quickImageResult.confidence,
                        real_prob: quickImageResult.real_prob,
                        deepfake_prob: quickImageResult.deepfake_prob,
                        face_detected: quickImageResult.face_detected,
                        created_at: new Date().toISOString(),
                        details: {
                          model_ready: quickImageResult.model_ready,
                          device: quickImageResult.device,
                        }
                      })}
                    >
                      View Report
                    </button>
                    <Link to="/image-detection" className="btn btn-secondary btn-sm">
                      Open in Studio →
                    </Link>
                  </div>
                </div>
              )}
            </div>

            <div className="action-card-footer">
              <Link to="/image-detection" className="card-link">
                Launch Full Image Detection Studio →
              </Link>
            </div>
          </div>

          {/* Functional Video Detection Card */}
          <div className="card action-card">
            <div className="action-card-header">
              <div className="action-card-icon">🎥</div>
              <div>
                <h3>Video Detection</h3>
                <p>Temporal analysis & frame extraction forensics</p>
              </div>
            </div>

            <div className="action-card-body">
              <div className="quick-upload-box">
                <input
                  type="file"
                  id="dashboard-video-upload"
                  accept="video/mp4,video/avi,video/quicktime,video/webm"
                  onChange={handleQuickVideoSelect}
                  style={{ display: 'none' }}
                />
                <label htmlFor="dashboard-video-upload" className="quick-upload-label">
                  {quickVideoFile ? (
                    <div className="quick-placeholder">
                      <span className="upload-arrow">🎬</span>
                      <span className="upload-text">{quickVideoFile.name}</span>
                      <span className="upload-types">{(quickVideoFile.size / (1024 * 1024)).toFixed(1)} MB</span>
                    </div>
                  ) : (
                    <div className="quick-placeholder">
                      <span className="upload-arrow">⬆️</span>
                      <span className="upload-text">Select Video to Analyze</span>
                      <span className="upload-types">MP4, AVI, MOV, WEBM</span>
                    </div>
                  )}
                </label>
              </div>

              {quickVideoFile && (
                <div className="quick-action-bar">
                  <span className="quick-filename" title={quickVideoFile.name}>
                    🎥 {quickVideoFile.name}
                  </span>
                  <button
                    className="btn btn-teal btn-sm"
                    onClick={handleQuickVideoAnalyze}
                    disabled={quickVideoLoading}
                  >
                    {quickVideoLoading ? (
                      <>
                        <span className="spinner spinner-sm" /> Processing…
                      </>
                    ) : (
                      'Analyze Video'
                    )}
                  </button>
                </div>
              )}

              {quickVideoError && (
                <div className="alert alert-danger" style={{ marginTop: '0.75rem', fontSize: '0.85rem' }}>
                  {quickVideoError}
                </div>
              )}

              {quickVideoResult && (
                <div className="alert alert-info" style={{ marginTop: '0.75rem', fontSize: '0.85rem' }}>
                  <div>
                    <strong>System Status:</strong>
                    <p style={{ marginTop: '0.2rem' }}>{quickVideoResult.error || 'Video analysis status received.'}</p>
                  </div>
                </div>
              )}
            </div>

            <div className="action-card-footer">
              <Link to="/video-detection" className="card-link">
                Open Dedicated Video Detection Page →
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Dashboard Summary — Calculated from Logged-in User's Actual Stored History */}
      <section className="dashboard-section">
        <div className="section-header">
          <div className="label">Activity Metrics</div>
          <h2>Detection Summary</h2>
          <p>Calculated live from your authenticated scanning history.</p>
        </div>

        <div className="grid-3 stats-grid">
          <div className="card summary-card">
            <div className="summary-icon">🖼️</div>
            <div className="summary-info">
              <div className="summary-val">
                {statsLoading ? <span className="spinner spinner-sm" /> : stats.images_analyzed}
              </div>
              <div className="summary-label">Images Analyzed</div>
              <div className="summary-sub">Total image forensic scans</div>
            </div>
          </div>

          <div className="card summary-card">
            <div className="summary-icon">🎥</div>
            <div className="summary-info">
              <div className="summary-val">
                {statsLoading ? <span className="spinner spinner-sm" /> : stats.videos_analyzed}
              </div>
              <div className="summary-label">Videos Analyzed</div>
              <div className="summary-sub">Temporal pipeline runs</div>
            </div>
          </div>

          <div className="card summary-card">
            <div className="summary-icon">⚡</div>
            <div className="summary-info">
              <div className="summary-val">
                {statsLoading ? <span className="spinner spinner-sm" /> : stats.recent_activity_count}
              </div>
              <div className="summary-label">Recent Activity (7d)</div>
              <div className="summary-sub">Scans executed this week</div>
            </div>
          </div>
        </div>
      </section>

      {/* Previous Searches Section: "Recent Detection History" */}
      <section className="dashboard-section" id="recent-history">
        <div className="section-header-flex">
          <div>
            <div className="label">Audit Trail</div>
            <h2>Recent Detection History</h2>
            <p>Review and filter your previous media scans.</p>
          </div>
          <Link to="/history" className="btn btn-outline btn-sm">
            View All History ({stats.total_analyzed}) →
          </Link>
        </div>

        {/* History Filters */}
        <div className="card filter-bar">
          <div className="search-input-wrapper">
            <span className="search-icon">🔍</span>
            <input
              type="text"
              className="search-input"
              placeholder="Search by file name…"
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
            />
            {searchQuery && (
              <button className="clear-search" onClick={() => setSearchQuery('')}>×</button>
            )}
          </div>

          <div className="filter-group">
            <label className="filter-label">Media:</label>
            <div className="pill-filters">
              {['all', 'image', 'video'].map(type => (
                <button
                  key={type}
                  className={`pill-btn ${mediaFilter === type ? 'active' : ''}`}
                  onClick={() => setMediaFilter(type)}
                >
                  {type === 'all' ? 'All Media' : type.charAt(0).toUpperCase() + type.slice(1)}
                </button>
              ))}
            </div>
          </div>

          <div className="filter-group">
            <label className="filter-label">Result:</label>
            <div className="pill-filters">
              {['all', 'REAL', 'DEEPFAKE'].map(res => (
                <button
                  key={res}
                  className={`pill-btn ${resultFilter === res ? 'active' : ''}`}
                  onClick={() => setResultFilter(res)}
                >
                  {res === 'all' ? 'All Results' : res}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* History Table / Empty State */}
        <div className="card history-card" style={{ marginTop: '1rem', padding: 0 }}>
          {historyLoading ? (
            <div className="history-loading-box">
              <div className="spinner" />
              <span>Loading detection records…</span>
            </div>
          ) : historyItems.length === 0 ? (
            <div className="history-empty-state">
              <div className="empty-icon">📂</div>
              <h3>No Detection History Found</h3>
              <p>
                {searchQuery || mediaFilter !== 'all' || resultFilter !== 'all'
                  ? 'No records match your selected search or filter criteria.'
                  : 'You have not analyzed any media yet. Upload an image above to start!'}
              </p>
              <div style={{ marginTop: '1rem' }}>
                <Link to="/image-detection" className="btn btn-primary btn-sm">
                  Analyze Your First Image
                </Link>
              </div>
            </div>
          ) : (
            <div className="table-responsive">
              <table className="table">
                <thead>
                  <tr>
                    <th>File Name</th>
                    <th>Type</th>
                    <th>Result</th>
                    <th>Confidence / Prob</th>
                    <th>Date & Time</th>
                    <th style={{ textAlign: 'right' }}>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {historyItems.map(item => (
                    <tr key={item.id}>
                      <td className="filename-cell" title={item.filename}>
                        <span className="file-icon">{item.media_type === 'video' ? '🎥' : '🖼️'}</span>
                        <span className="file-text">{item.filename}</span>
                      </td>
                      <td>
                        <span className="badge badge-navy">
                          {item.media_type ? item.media_type.toUpperCase() : 'IMAGE'}
                        </span>
                      </td>
                      <td>
                        {item.prediction ? (
                          <span className={`badge ${item.prediction === 'REAL' ? 'badge-real' : 'badge-fake'}`}>
                            {item.prediction === 'REAL' ? '✓ REAL' : '⚠️ DEEPFAKE'}
                          </span>
                        ) : (
                          <span className="badge badge-warning">IN PROGRESS</span>
                        )}
                      </td>
                      <td>
                        <div className="prob-cell">
                          <strong>{item.confidence != null ? `${(item.confidence * 100).toFixed(1)}%` : '—'}</strong>
                          {item.real_prob != null && item.deepfake_prob != null && (
                            <div className="prob-mini">
                              <span>R: {(item.real_prob * 100).toFixed(0)}%</span>
                              <span>·</span>
                              <span>F: {(item.deepfake_prob * 100).toFixed(0)}%</span>
                            </div>
                          )}
                        </div>
                      </td>
                      <td className="date-cell">
                        {item.created_at
                          ? new Date(item.created_at).toLocaleString(undefined, {
                              year: 'numeric',
                              month: 'short',
                              day: 'numeric',
                              hour: '2-digit',
                              minute: '2-digit',
                            })
                          : '—'}
                      </td>
                      <td style={{ textAlign: 'right' }}>
                        <div className="action-btn-group">
                          <button
                            className="btn btn-outline btn-sm"
                            onClick={() => setSelectedReport(item)}
                            title="View Forensic Report"
                          >
                            View Report
                          </button>
                          <button
                            className="btn btn-danger btn-sm"
                            onClick={(e) => handleDeleteHistory(item.id, e)}
                            title="Delete Record"
                          >
                            🗑️
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </section>

      {/* View Report Modal */}
      {selectedReport && (
        <div className="modal-backdrop" onClick={() => setSelectedReport(null)}>
          <div className="modal-dialog" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <div>
                <span className="badge badge-teal" style={{ marginBottom: '0.25rem' }}>Forensic Report</span>
                <h3>{selectedReport.filename}</h3>
              </div>
              <button className="btn-close" onClick={() => setSelectedReport(null)}>✕</button>
            </div>

            <div className="modal-body">
              <div className="report-summary-box">
                <div className="report-result-header">
                  <span
                    className={`badge ${selectedReport.prediction === 'REAL' ? 'badge-real' : 'badge-fake'}`}
                    style={{ fontSize: '1rem', padding: '0.5rem 1rem' }}
                  >
                    {selectedReport.prediction === 'REAL' ? 'AUTHENTIC (REAL)' : 'MANIPULATED (DEEPFAKE)'}
                  </span>
                  <div className="report-confidence">
                    Overall Confidence: <strong>{selectedReport.confidence != null ? `${(selectedReport.confidence * 100).toFixed(2)}%` : '—'}</strong>
                  </div>
                </div>

                <div className="report-metric-grid" style={{ marginTop: '1.25rem' }}>
                  <div className="metric-box">
                    <span className="metric-lbl">Media Type</span>
                    <span className="metric-val">{selectedReport.media_type?.toUpperCase()}</span>
                  </div>
                  <div className="metric-box">
                    <span className="metric-lbl">Analysis Timestamp</span>
                    <span className="metric-val">
                      {selectedReport.created_at
                        ? new Date(selectedReport.created_at).toLocaleString()
                        : '—'}
                    </span>
                  </div>
                  <div className="metric-box">
                    <span className="metric-lbl">Real Probability</span>
                    <span className="metric-val text-success">
                      {selectedReport.real_prob != null ? `${(selectedReport.real_prob * 100).toFixed(2)}%` : '—'}
                    </span>
                  </div>
                  <div className="metric-box">
                    <span className="metric-lbl">Deepfake Probability</span>
                    <span className="metric-val text-danger">
                      {selectedReport.deepfake_prob != null ? `${(selectedReport.deepfake_prob * 100).toFixed(2)}%` : '—'}
                    </span>
                  </div>
                </div>

                {selectedReport.face_detected != null && (
                  <div className="report-detail-row" style={{ marginTop: '1rem' }}>
                    <span className="text-secondary">Facial Region Detection:</span>
                    <strong>{selectedReport.face_detected ? '✓ Face Isolated and Analyzed' : 'Full Image Scan'}</strong>
                  </div>
                )}

                {selectedReport.details?.device && (
                  <div className="report-detail-row">
                    <span className="text-secondary">Execution Device:</span>
                    <code>{selectedReport.details.device}</code>
                  </div>
                )}
              </div>
            </div>

            <div className="modal-footer">
              <button className="btn btn-outline" onClick={() => setSelectedReport(null)}>
                Close
              </button>
              <button
                className="btn btn-primary"
                onClick={() => {
                  window.print()
                }}
              >
                Print / Save Report
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
