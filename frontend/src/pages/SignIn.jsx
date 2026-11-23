import { useState } from 'react'
import { Link, useNavigate, useLocation } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import './SignIn.css'

export default function SignIn() {
  const [identifier, setIdentifier] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const { login } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()
  const from = location.state?.from?.pathname || '/dashboard'
  const successMsg = location.state?.message

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError(null)

    if (!identifier.trim()) {
      setError('Please enter your email address or username.')
      return
    }
    if (!password) {
      setError('Please enter your password.')
      return
    }

    setLoading(true)
    try {
      await login({ identifier: identifier.trim(), password })
      navigate(from, { replace: true })
    } catch (err) {
      const respData = err.response?.data
      let errorMsg = ''
      if (typeof respData?.detail === 'string') {
        errorMsg = respData.detail
      } else if (Array.isArray(respData?.detail)) {
        errorMsg = respData.detail.map(d => d.msg || d.message).join(', ')
      } else if (typeof respData?.message === 'string') {
        errorMsg = respData.message
      } else if (err.response?.status === 503) {
        errorMsg = respData?.detail || 'Database service is currently unavailable. Ensure MongoDB is running.'
      } else if (err.message === 'Network Error' || !err.response) {
        errorMsg = 'Network Error: Unable to reach backend server at http://localhost:8000. Please ensure the backend is running.'
      } else if (err.message) {
        errorMsg = err.message
      } else {
        errorMsg = 'Failed to sign in. Please check your credentials or database status.'
      }
      setError(errorMsg)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="auth-page">
      <div className="auth-card card">
        <div className="auth-header">
          <div className="auth-logo">🛡️</div>
          <h2>Welcome Back</h2>
          <p className="text-muted">Sign in to access your detection dashboard & history</p>
        </div>

        {successMsg && (
          <div className="alert alert-success" style={{ marginBottom: '1.25rem' }}>
            ✓ {successMsg}
          </div>
        )}

        {error && (
          <div className="alert alert-danger" style={{ marginBottom: '1.25rem' }}>
            ⚠️ {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="auth-form" noValidate>
          <div className="form-group">
            <label htmlFor="identifier">Email Address or Username</label>
            <input
              id="identifier"
              type="text"
              className="auth-input"
              placeholder="e.g. user@example.com or researcher"
              value={identifier}
              onChange={(e) => setIdentifier(e.target.value)}
              autoComplete="username"
              disabled={loading}
              required
            />
          </div>

          <div className="form-group">
            <div className="password-label-row">
              <label htmlFor="password">Password</label>
              <button
                type="button"
                className="btn-toggle-pwd"
                onClick={() => setShowPassword(!showPassword)}
                tabIndex="-1"
              >
                {showPassword ? 'Hide' : 'Show'}
              </button>
            </div>
            <div className="password-input-wrap">
              <input
                id="password"
                type={showPassword ? 'text' : 'password'}
                className="auth-input"
                placeholder="Enter your password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                autoComplete="current-password"
                disabled={loading}
                required
              />
            </div>
          </div>

          <button
            type="submit"
            className="btn btn-primary btn-auth-submit"
            disabled={loading}
          >
            {loading ? (
              <>
                <span className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }} />
                Signing In…
              </>
            ) : (
              'Sign In'
            )}
          </button>
        </form>

        <div className="auth-footer">
          <p>
            Don't have an account?{' '}
            <Link to="/signup" className="text-accent auth-link">
              Create an Account
            </Link>
          </p>
        </div>
      </div>
    </div>
  )
}
