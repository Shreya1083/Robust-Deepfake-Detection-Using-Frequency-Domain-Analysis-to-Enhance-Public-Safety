import { useState } from 'react'
import { NavLink, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import './Navbar.css'

export default function Navbar() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  const handleLogout = async () => {
    await logout()
    navigate('/')
  }

  const closeMenu = () => setMobileMenuOpen(false)

  return (
    <nav className="navbar">
      <div className="navbar-inner">
        {/* Brand */}
        <NavLink to={user ? "/dashboard" : "/"} className="navbar-brand" onClick={closeMenu}>
          <div className="brand-logo-container">
            <span className="brand-logo-icon">🛡️</span>
          </div>
          <div className="brand-text-group">
            <span className="brand-name">Veri<span className="brand-accent">Frame</span></span>
            <span className="brand-tagline">AI Detection</span>
          </div>
        </NavLink>

        {/* Mobile menu toggle */}
        <button
          className="mobile-toggle"
          onClick={() => setMobileMenuOpen(prev => !prev)}
          aria-label="Toggle navigation menu"
        >
          <span className="hamburger-bar" />
          <span className="hamburger-bar" />
          <span className="hamburger-bar" />
        </button>

        {/* Navigation Links */}
        <div className={`navbar-collapse ${mobileMenuOpen ? 'open' : ''}`}>
          <ul className="navbar-links">
            {!user ? (
              <>
                <li>
                  <NavLink to="/" end className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`} onClick={closeMenu}>
                    Home
                  </NavLink>
                </li>
                <li>
                  <NavLink to="/about" className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`} onClick={closeMenu}>
                    About
                  </NavLink>
                </li>
              </>
            ) : (
              <>
                <li>
                  <NavLink to="/dashboard" className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`} onClick={closeMenu}>
                    Dashboard
                  </NavLink>
                </li>
                <li>
                  <NavLink to="/image-detection" className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`} onClick={closeMenu}>
                    Image Detection
                  </NavLink>
                </li>
                <li>
                  <NavLink to="/video-detection" className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`} onClick={closeMenu}>
                    Video Detection
                  </NavLink>
                </li>
                <li>
                  <NavLink to="/history" className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`} onClick={closeMenu}>
                    History
                  </NavLink>
                </li>
                <li>
                  <NavLink to="/profile" className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`} onClick={closeMenu}>
                    Profile
                  </NavLink>
                </li>
              </>
            )}
          </ul>

          {/* Auth section */}
          <div className="navbar-auth">
            {user ? (
              <div className="user-profile-badge">
                <NavLink to="/profile" className="user-greeting" onClick={closeMenu} title={user.email}>
                  <span className="user-avatar-circle">
                    {(user.full_name || user.username || 'U')[0].toUpperCase()}
                  </span>
                  <span className="user-name">{user.full_name || user.username}</span>
                </NavLink>
                <button onClick={handleLogout} className="btn-logout" title="Sign Out">
                  Sign Out
                </button>
              </div>
            ) : (
              <div className="auth-buttons">
                <NavLink to="/signin" className="nav-auth-link" onClick={closeMenu}>
                  Sign In
                </NavLink>
                <NavLink to="/signup" className="btn btn-primary btn-signup-nav" onClick={closeMenu}>
                  Get Started
                </NavLink>
              </div>
            )}
          </div>
        </div>
      </div>
    </nav>
  )
}
