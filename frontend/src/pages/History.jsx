import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { apiGetHistory, apiGetHistoryStats, apiDeleteHistoryItem } from '../services/api'
import './History.css'

export default function History() {
  const [history, setHistory] = useState([])
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [search, setSearch] = useState('')
  const [mediaFilter, setMediaFilter] = useState('all')
  const [resultFilter, setResultFilter] = useState('all')
  const [selectedItem, setSelectedItem] = useState(null)
  const [deletingId, setDeletingId] = useState(null)

  const fetchStats = async () => {
    try {
      const { data } = await apiGetHistoryStats()
      setStats(data)
    } catch (err) {
      console.warn('Failed to fetch history stats:', err)
    }
  }

  const fetchHistory = async () => {
    setLoading(true)
    setError(null)
    try {
      const params = { limit: 100 }
      if (search.trim()) params.search = search.trim()
      if (mediaFilter !== 'all') params.media_type = mediaFilter
      if (resultFilter !== 'all') params.result = resultFilter
      const { data } = await apiGetHistory(params)
      setHistory(data.items || [])
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to load detection history.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchStats()
  }, [])

  useEffect(() => {
    const timer = setTimeout(() => {
      fetchHistory()
    }, 250)
    return () => clearTimeout(timer)
  }, [search, mediaFilter, resultFilter])

  const handleDelete = async (id, e) => {
    e.stopPropagation()
    if (!window.confirm('Are you sure you want to delete this detection record?')) return
    setDeletingId(id)
    try {
      await apiDeleteHistoryItem(id)
      setHistory((prev) => prev.filter((item) => item.id !== id))
      fetchStats()
    } catch (err) {
      alert(err.response?.data?.detail || 'Failed to delete history record.')
    } finally {
      setDeletingId(null)
    }
  }

  return (
    <div className="page-container history-page">
      <div className="section-header">
        <span className="badge badge-teal">Forensic Audit Log</span>
        <h1>Detection History</h1>
        <p>
          Comprehensive chronological log of your analyzed images and videos. Filter, inspect detailed
          confidence scores, and export forensic verification reports.
        </p>
      </div>

      {/* Summary Stats Row */}
      <div className="history-stats-grid">
        <div className="card stat-card">
          <span className="stat-label">Total Media Scanned</span>
          <span className="stat-val">{stats?.total_scans ?? '—'}</span>
          <span className="stat-sub">Lifetime VeriFrame scans</span>
        </div>
        <div className="card stat-card">
          <span className="stat-label">Authentic (REAL)</span>
          <span className="stat-val text-success">{stats?.real_count ?? '—'}</span>
          <span className="stat-sub">Verified authentic media</span>
        </div>
        <div className="card stat-card">
          <span className="stat-label">Manipulated (DEEPFAKE)</span>
          <span className="stat-val text-danger">{stats?.deepfake_count ?? '—'}</span>
          <span className="stat-sub">Flagged synthetic media</span>
        </div>
        <div className="card stat-card">
          <span className="stat-label">Recent Activity (7 Days)</span>
          <span className="stat-val text-accent">{stats?.recent_activity_count ?? '—'}</span>
          <span className="stat-sub">Active investigations</span>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="card history-filter-card">
        <div className="filter-row">
          <div className="search-wrap">
            <span className="search-icon">🔍</span>
            <input
              type="text"
              className="filter-search-input"
              placeholder="Search by file name…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
            {search && (
              <button className="clear-search-btn" onClick={() => setSearch('')}>
                ✕
              </button>
            )}
          </div>

          <div className="filter-controls">
            <div className="filter-group">
              <label>Media Type:</label>
              <select
                className="filter-select"
                value={mediaFilter}
                onChange={(e) => setMediaFilter(e.target.value)}
              >
                <option value="all">All Media</option>
                <option value="image">Images Only</option>
                <option value="video">Videos Only</option>
              </select>
            </div>

            <div className="filter-group">
              <label>Classification:</label>
              <select
                className="filter-select"
                value={resultFilter}
                onChange={(e) => setResultFilter(e.target.value)}
              >
                <option value="all">All Verdicts</option>
                <option value="REAL">REAL</option>
                <option value="DEEPFAKE">DEEPFAKE</option>
              </select>
            </div>

            <button className="btn btn-outline btn-refresh" onClick={() => { fetchStats(); fetchHistory() }}>
              🔄 Refresh
            </button>
          </div>
        </div>
      </div>

      {/* History Table / List */}
      <div className="card history-table-card">
        {loading ? (
          <div className="table-loading">
            <div className="spinner" style={{ width: 32, height: 32, marginBottom: '0.75rem' }} />
            <p className="text-muted">Loading forensic history records…</p>
          </div>
        ) : error ? (
          <div className="alert alert-danger" style={{ margin: '1.5rem' }}>
            ⚠️ {error}
          </div>
        ) : history.length === 0 ? (
          <div className="table-empty">
            <div className="empty-icon">📁</div>
            <h3>No Detection Records Found</h3>
            <p className="text-muted">
              {search || mediaFilter !== 'all' || resultFilter !== 'all'
                ? 'No records match your active search filters.'
                : 'You have not analyzed any media files yet.'}
            </p>
            <div className="empty-actions" style={{ marginTop: '1rem' }}>
              <Link to="/image-detection" className="btn btn-primary">
                🖼️ Run Image Detection
              </Link>
            </div>
          </div>
        ) : (
          <div className="table-wrapper">
            <table className="history-table">
              <thead>
                <tr>
                  <th>Media File</th>
                  <th>Type</th>
                  <th>Verdict</th>
                  <th>Confidence</th>
                  <th>Probabilities</th>
                  <th>Date &amp; Time</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {history.map((item) => {
                  const isReal = item.prediction === 'REAL'
                  const isFake = item.prediction === 'DEEPFAKE'
                  const dateStr = item.created_at
                    ? new Date(item.created_at).toLocaleString(undefined, {
                        year: 'numeric',
                        month: 'short',
                        day: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit',
                      })
                    : '—'

                  return (
                    <tr key={item.id} className="history-row">
                      <td className="col-filename">
                        <div className="file-info-cell">
                          <span className="file-type-icon">
                            {item.media_type === 'video' ? '🎬' : '🖼️'}
                          </span>
                          <div>
                            <span className="filename-text" title={item.filename}>
                              {item.filename}
                            </span>
                            {item.file_size_bytes && (
                              <span className="filesize-sub">
                                {(item.file_size_bytes / 1024).toFixed(1)} KB
                              </span>
                            )}
                          </div>
                        </div>
                      </td>
                      <td>
                        <span className="media-type-badge">
                          {item.media_type.toUpperCase()}
                        </span>
                      </td>
                      <td>
                        <span className={`badge ${isReal ? 'badge-real' : isFake ? 'badge-fake' : 'badge-neutral'}`}>
                          {item.prediction}
                        </span>
                      </td>
                      <td>
                        <strong>{item.confidence != null ? `${(item.confidence * 100).toFixed(1)}%` : '—'}</strong>
                      </td>
                      <td className="col-probs">
                        {item.real_prob != null && item.deepfake_prob != null ? (
                          <div className="prob-micro-bar">
                            <span className="text-success">R: {(item.real_prob * 100).toFixed(0)}%</span>
                            <span className="prob-slash">/</span>
                            <span className="text-danger">F: {(item.deepfake_prob * 100).toFixed(0)}%</span>
                          </div>
                        ) : (
                          '—'
                        )}
                      </td>
                      <td className="col-date">{dateStr}</td>
                      <td className="col-actions">
                        <button
                          className="btn btn-outline btn-sm"
                          onClick={() => setSelectedItem(item)}
                          title="View detailed forensic report"
                        >
                          📄 Report
                        </button>
                        <button
                          className="btn btn-danger-outline btn-sm"
                          onClick={(e) => handleDelete(item.id, e)}
                          disabled={deletingId === item.id}
                          title="Delete record"
                        >
                          {deletingId === item.id ? '…' : '🗑️'}
                        </button>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Full Forensic Report Modal */}
      {selectedItem && (
        <div className="modal-backdrop" onClick={() => setSelectedItem(null)}>
          <div className="modal-card print-report" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div className="modal-brand">
                <span className="modal-brand-icon">🛡️</span>
                <div>
                  <h3>VeriFrame Forensic Inspection Report</h3>
                  <p className="text-muted" style={{ fontSize: '0.8rem' }}>AI-Powered Deepfake Detection System</p>
                </div>
              </div>
              <button className="modal-close" onClick={() => setSelectedItem(null)}>✕</button>
            </div>

            <div className="modal-body">
              <div
                className="report-verdict-banner"
                style={{
                  background:
                    selectedItem.prediction === 'REAL'
                      ? 'rgba(16, 185, 129, 0.08)'
                      : selectedItem.prediction === 'DEEPFAKE'
                      ? 'rgba(239, 68, 68, 0.08)'
                      : 'rgba(245, 158, 11, 0.08)',
                  border: `1px solid ${
                    selectedItem.prediction === 'REAL'
                      ? 'var(--success)'
                      : selectedItem.prediction === 'DEEPFAKE'
                      ? 'var(--danger)'
                      : 'var(--warning)'
                  }`,
                  padding: '1.25rem',
                  borderRadius: '8px',
                  marginBottom: '1.5rem',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                }}
              >
                <div>
                  <span style={{ fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)' }}>
                    Primary Determination
                  </span>
                  <div
                    style={{
                      fontSize: '1.6rem',
                      fontWeight: 800,
                      color: selectedItem.prediction === 'REAL' ? 'var(--success)' : 'var(--danger)',
                    }}
                  >
                    {selectedItem.prediction}
                  </div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Confidence Score</span>
                  <div style={{ fontSize: '1.4rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                    {selectedItem.confidence != null ? `${(selectedItem.confidence * 100).toFixed(1)}%` : '—'}
                  </div>
                </div>
              </div>

              <div className="report-section">
                <h4>Record Details</h4>
                <table className="report-table">
                  <tbody>
                    <tr>
                      <td>Record ID</td>
                      <td><code>{selectedItem.id}</code></td>
                    </tr>
                    <tr>
                      <td>File Name</td>
                      <td><strong>{selectedItem.filename}</strong></td>
                    </tr>
                    <tr>
                      <td>Media Type</td>
                      <td>{selectedItem.media_type.toUpperCase()}</td>
                    </tr>
                    <tr>
                      <td>Scanned Date</td>
                      <td>{new Date(selectedItem.created_at).toLocaleString()}</td>
                    </tr>
                    <tr>
                      <td>Face Alignment</td>
                      <td>{selectedItem.face_detected ? '✅ Face detected and aligned' : '⚠️ Full frame fallback'}</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <div className="report-section">
                <h4>Quantitative Classification Probabilities</h4>
                <table className="report-table">
                  <thead>
                    <tr>
                      <th>Class Target</th>
                      <th>Calculated Likelihood</th>
                      <th>Risk Classification</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>Authentic Content (REAL)</td>
                      <td>{(selectedItem.real_prob != null ? (selectedItem.real_prob * 100).toFixed(2) : '—')}%</td>
                      <td>
                        <span className={`badge ${selectedItem.real_prob > 0.5 ? 'badge-real' : 'badge-neutral'}`}>
                          {selectedItem.real_prob > 0.5 ? 'Primary Match' : 'Unmatched'}
                        </span>
                      </td>
                    </tr>
                    <tr>
                      <td>Synthetic / Manipulated (DEEPFAKE)</td>
                      <td>{(selectedItem.deepfake_prob != null ? (selectedItem.deepfake_prob * 100).toFixed(2) : '—')}%</td>
                      <td>
                        <span className={`badge ${selectedItem.deepfake_prob > 0.5 ? 'badge-fake' : 'badge-neutral'}`}>
                          {selectedItem.deepfake_prob > 0.5 ? 'Elevated Risk' : 'Low Risk'}
                        </span>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <div className="report-section">
                <h4>Forensic Findings &amp; Summary</h4>
                <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                  {selectedItem.prediction === 'REAL'
                    ? 'The verified media demonstrates natural spatial texture continuity and standard photographic illumination without significant generative synthesis artifacts.'
                    : 'The analyzed media exhibits strong indicators of digital manipulation or artificial facial synthesis, consistent with deepfake generative models.'}
                </p>
              </div>
            </div>

            <div className="modal-footer">
              <button className="btn btn-primary" onClick={() => window.print()}>
                🖨️ Print / Save PDF
              </button>
              <button className="btn btn-outline" onClick={() => setSelectedItem(null)}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
