import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import './SignUp.css'

export default function SignUp() {
  const [formData, setFormData] = useState({
    full_name: '',
    email: '',
    username: '',
    password: '',
    confirm_password: '',
  })
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirm, setShowConfirm] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [fieldErrors, setFieldErrors] = useState({})

  const { signup } = useAuth()
  const navigate = useNavigate()

  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData(prev => ({ ...prev, [name]: value }))
    // Clear field-specific error as user types
    if (fieldErrors[name]) {
      setFieldErrors(prev => ({ ...prev, [name]: null }))
    }
  }

  const validate = () => {
    const errs = {}
    if (!formData.full_name.trim() || formData.full_name.trim().length < 2) {
      errs.full_name = 'Full name must be at least 2 characters.'
    }
    const emailRegex = /^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$/
    if (!formData.email.trim() || !emailRegex.test(formData.email.trim())) {
      errs.email = 'Please enter a valid email address.'
    }
    const usernameRegex = /^[a-zA-Z0-9_.-]{3,30}$/
    if (!formData.username.trim() || !usernameRegex.test(formData.username.trim())) {
      errs.username = 'Username must be 3–30 characters (letters, numbers, _, ., -).'
    }
    if (!formData.password || formData.password.length < 6) {
      errs.password = 'Password must be at least 6 characters long.'
    }
    if (formData.password !== formData.confirm_password) {
      errs.confirm_password = 'Passwords do not match.'
    }
    return errs
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError(null)
    const errs = validate()
    if (Object.keys(errs).length > 0) {
      setFieldErrors(errs)
      return
    }

    setLoading(true)
    try {
      await signup({
        full_name: formData.full_name.trim(),
        email: formData.email.trim().toLowerCase(),
        username: formData.username.trim().toLowerCase(),
        password: formData.password,
        confirm_password: formData.confirm_password,
      })
      // Redirect to Sign In with success message
      navigate('/signin', {
        state: { message: 'Registration successful! Please sign in with your credentials.' },
        replace: true,
      })
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
      } else if (err.response?.status === 500) {
        errorMsg = 'Internal server error during registration. Please check server logs.'
      } else if (err.message === 'Network Error' || !err.response) {
        errorMsg = 'Network Error: Unable to reach backend server at http://localhost:8000. Please ensure the backend is running.'
      } else if (err.message) {
        errorMsg = err.message
      } else {
        errorMsg = 'Registration failed. Please verify your details or database connectivity.'
      }
      setError(errorMsg)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="auth-page">
      <div className="auth-card signup-card card">
        <div className="auth-header">
          <div className="auth-brand-badge">
            <span className="auth-logo">🛡️</span>
            <span className="auth-brand-title">VeriFrame</span>
          </div>
          <h2>Create VeriFrame Account</h2>
          <p className="text-muted">AI-Powered Deepfake Detection System</p>
        </div>

        {error && (
          <div className="alert alert-danger" style={{ marginBottom: '1.25rem' }}>
            ⚠️ {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="auth-form" noValidate>
          <div className="form-group">
            <label htmlFor="full_name">Full Name</label>
            <input
              id="full_name"
              name="full_name"
              type="text"
              className={`auth-input ${fieldErrors.full_name ? 'input-error' : ''}`}
              placeholder="e.g. Dr. Alex Mercer"
              value={formData.full_name}
              onChange={handleChange}
              disabled={loading}
              required
            />
            {fieldErrors.full_name && <span className="field-error-text">{fieldErrors.full_name}</span>}
          </div>

          <div className="form-group">
            <label htmlFor="email">Email Address</label>
            <input
              id="email"
              name="email"
              type="email"
              className={`auth-input ${fieldErrors.email ? 'input-error' : ''}`}
              placeholder="e.g. alex@university.edu"
              value={formData.email}
              onChange={handleChange}
              autoComplete="email"
              disabled={loading}
              required
            />
            {fieldErrors.email && <span className="field-error-text">{fieldErrors.email}</span>}
          </div>

          <div className="form-group">
            <label htmlFor="username">Username</label>
            <input
              id="username"
              name="username"
              type="text"
              className={`auth-input ${fieldErrors.username ? 'input-error' : ''}`}
              placeholder="Choose a unique username"
              value={formData.username}
              onChange={handleChange}
              autoComplete="username"
              disabled={loading}
              required
            />
            {fieldErrors.username && <span className="field-error-text">{fieldErrors.username}</span>}
          </div>

          <div className="form-group">
            <div className="password-label-row">
              <label htmlFor="password">Password (min. 6 characters)</label>
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
                name="password"
                type={showPassword ? 'text' : 'password'}
                className={`auth-input ${fieldErrors.password ? 'input-error' : ''}`}
                placeholder="Create a strong password"
                value={formData.password}
                onChange={handleChange}
                autoComplete="new-password"
                disabled={loading}
                required
              />
            </div>
            {fieldErrors.password && <span className="field-error-text">{fieldErrors.password}</span>}
          </div>

          <div className="form-group">
            <div className="password-label-row">
              <label htmlFor="confirm_password">Confirm Password</label>
              <button
                type="button"
                className="btn-toggle-pwd"
                onClick={() => setShowConfirm(!showConfirm)}
                tabIndex="-1"
              >
                {showConfirm ? 'Hide' : 'Show'}
              </button>
            </div>
            <div className="password-input-wrap">
              <input
                id="confirm_password"
                name="confirm_password"
                type={showConfirm ? 'text' : 'password'}
                className={`auth-input ${fieldErrors.confirm_password ? 'input-error' : ''}`}
                placeholder="Re-enter your password"
                value={formData.confirm_password}
                onChange={handleChange}
                autoComplete="new-password"
                disabled={loading}
                required
              />
            </div>
            {fieldErrors.confirm_password && <span className="field-error-text">{fieldErrors.confirm_password}</span>}
          </div>

          <button
            type="submit"
            className="btn btn-primary btn-auth-submit"
            disabled={loading}
          >
            {loading ? (
              <>
                <span className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }} />
                Registering Account…
              </>
            ) : (
              'Create Account'
            )}
          </button>
        </form>

        <div className="auth-footer">
          <p>
            Already have an account?{' '}
            <Link to="/signin" className="text-accent auth-link">
              Sign In
            </Link>
          </p>
        </div>
      </div>
    </div>
  )
}
