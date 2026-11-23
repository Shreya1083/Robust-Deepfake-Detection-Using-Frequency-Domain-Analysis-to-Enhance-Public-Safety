import { NavLink, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import './Navbar.css'

const navLinks = [
  { to: '/', label: 'Home' },
  { to: '/about', label: 'About' },
  { to: '/image-detection', label: 'Image Detection' },
  { to: '/video-detection', label: 'Video Detection' },
  { to: '/frequency-analysis', label: 'Frequency Analysis' },
  { to: '/dashboard', label: 'Dashboard', protected: true },
]

export default function Navbar() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()

  const handleLogout = async () => {
    await logout()
    navigate('/signin')
  }

  return (
    <nav className="navbar">
      <div className="navbar-inner">
        <NavLink to="/" className="navbar-brand">
          <span className="brand-icon">🛡️</span>
          <span className="brand-text">DeepGuard<span className="brand-accent"> AI</span></span>
        </NavLink>

        <ul className="navbar-links">
          {navLinks.map(({ to, label, protected: isProtected }) => (
            <li key={to}>
              <NavLink
                to={to}
                end={to === '/'}
                className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
              >
                {label}
                {isProtected && !user && <span className="nav-lock" title="Sign In Required"> 🔒</span>}
              </NavLink>
            </li>
          ))}
        </ul>

        <div className="navbar-auth">
          {user ? (
            <div className="user-profile-badge">
              <span className="user-greeting" title={user.email}>
                👤 <span className="user-name">{user.full_name || user.username}</span>
              </span>
              <button onClick={handleLogout} className="btn-logout" title="Sign Out">
                Sign Out
              </button>
            </div>
          ) : (
            <div className="auth-buttons">
              <NavLink to="/signin" className="nav-auth-link">
                Sign In
              </NavLink>
              <NavLink to="/signup" className="btn btn-primary btn-signup-nav">
                Sign Up
              </NavLink>
            </div>
          )}
        </div>
      </div>
    </nav>
  )
}
