import { useState, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { apiGetHistoryStats } from '../services/api'
import './Profile.css'

export default function Profile() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function loadStats() {
      try {
        const { data } = await apiGetHistoryStats()
        setStats(data)
      } catch (err) {
        console.warn('Failed to load user stats:', err)
      } finally {
        setLoading(false)
      }
    }
    loadStats()
  }, [])

  const handleLogout = async () => {
    await logout()
    navigate('/')
  }

  const memberSince = user?.created_at
    ? new Date(user.created_at).toLocaleDateString(undefined, {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
      })
    : 'Active'

  const userInitial = user?.full_name ? user.full_name.charAt(0).toUpperCase() : 'U'

  return (
    <div className="page-container profile-page">
      <div className="section-header">
        <span className="badge badge-teal">User Account</span>
        <h1>Account Profile</h1>
        <p>Manage your account credentials, view inspection usage statistics, and manage session security.</p>
      </div>

      <div className="profile-layout">
        {/* User Card */}
        <div className="card profile-user-card">
          <div className="profile-avatar-large">{userInitial}</div>
          <h2 className="profile-name">{user?.full_name || 'VeriFrame User'}</h2>
          <span className="profile-username">@{user?.username || 'user'}</span>
          <span className="badge badge-real" style={{ marginTop: '0.5rem' }}>
            ● Verified Account
          </span>

          <div className="profile-details-list">
            <div className="profile-detail-item">
              <span className="detail-label">Email Address</span>
              <span className="detail-value">{user?.email || '—'}</span>
            </div>
            <div className="profile-detail-item">
              <span className="detail-label">Username</span>
              <span className="detail-value">{user?.username || '—'}</span>
            </div>
            <div className="profile-detail-item">
              <span className="detail-label">Member Since</span>
              <span className="detail-value">{memberSince}</span>
            </div>
            <div className="profile-detail-item">
              <span className="detail-label">Account Identifier</span>
              <span className="detail-value text-muted" style={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>
                {user?.id || '—'}
              </span>
            </div>
          </div>

          <button className="btn btn-outline btn-logout-full" onClick={handleLogout}>
            🚪 Sign Out of VeriFrame
          </button>
        </div>

        {/* Usage & Security Column */}
        <div className="profile-content-col">
          {/* Usage Metrics */}
          <div className="card profile-metrics-card">
            <div className="card-header">
              <h3>Detection Activity Overview</h3>
              <p className="text-muted">Lifetime forensic scans associated with your profile</p>
            </div>

            <div className="metrics-grid">
              <div className="metric-box">
                <span className="metric-val">{stats?.total_scans ?? (loading ? '…' : 0)}</span>
                <span className="metric-name">Total Scans Run</span>
              </div>
              <div className="metric-box">
                <span className="metric-val text-success">{stats?.real_count ?? (loading ? '…' : 0)}</span>
                <span className="metric-name">Authentic (REAL)</span>
              </div>
              <div className="metric-box">
                <span className="metric-val text-danger">{stats?.deepfake_count ?? (loading ? '…' : 0)}</span>
                <span className="metric-name">Manipulated (DEEPFAKE)</span>
              </div>
              <div className="metric-box">
                <span className="metric-val text-accent">{stats?.recent_activity_count ?? (loading ? '…' : 0)}</span>
                <span className="metric-name">Recent Activity (7d)</span>
              </div>
            </div>

            <div className="metrics-footer">
              <Link to="/history" className="btn btn-outline btn-sm">
                📁 Open Full Detection History →
              </Link>
            </div>
          </div>

          {/* Security & System Info */}
          <div className="card profile-security-card">
            <div className="card-header">
              <h3>Security &amp; Data Protection</h3>
              <p className="text-muted">Enterprise forensic privacy guarantees</p>
            </div>

            <div className="security-items">
              <div className="security-item">
                <span className="sec-icon">🔒</span>
                <div>
                  <strong>Cryptographic Password Hashing</strong>
                  <p>Passwords are hashed with bcrypt and a high work factor. Plaintext credentials are never saved.</p>
                </div>
              </div>
              <div className="security-item">
                <span className="sec-icon">🛡️</span>
                <div>
                  <strong>HttpOnly Session Token</strong>
                  <p>JSON Web Tokens (JWT) are stored strictly in browser HttpOnly cookies to mitigate XSS exposure.</p>
                </div>
              </div>
              <div className="security-item">
                <span className="sec-icon">🗄️</span>
                <div>
                  <strong>Isolated MongoDB Audit Scoping</strong>
                  <p>All detection records and file hashes are scoped strictly to your unique authenticated user account.</p>
                </div>
              </div>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="card profile-actions-card">
            <div className="quick-actions-row">
              <Link to="/dashboard" className="btn btn-primary">
                📊 Go to Dashboard
              </Link>
              <Link to="/image-detection" className="btn btn-outline">
                🖼️ New Image Scan
              </Link>
              <Link to="/frequency-analysis" className="btn btn-outline">
                📡 Frequency Studio
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
