import { Navigate, useLocation } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

export default function ProtectedRoute({ children }) {
  const { user, loading } = useAuth()
  const location = useLocation()

  if (loading) {
    return (
      <div className="page-container" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '50vh' }}>
        <div style={{ textAlign: 'center' }}>
          <div className="spinner" style={{ margin: '0 auto 1rem auto' }} />
          <p className="text-muted">Authenticating user session…</p>
        </div>
      </div>
    )
  }

  if (!user) {
    // Redirect to Sign In, preserving intended target
    return <Navigate to="/signin" state={{ from: location }} replace />
  }

  return children
}
