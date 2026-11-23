import { NavLink } from 'react-router-dom'
import './Navbar.css'

const links = [
  { to: '/', label: 'Home' },
  { to: '/about', label: 'About' },
  { to: '/image-detection', label: 'Image Detection' },
  { to: '/video-detection', label: 'Video Detection' },
  { to: '/frequency-analysis', label: 'Frequency Analysis' },
  { to: '/dashboard', label: 'Dashboard' },
]

export default function Navbar() {
  return (
    <nav className="navbar">
      <div className="navbar-inner">
        <NavLink to="/" className="navbar-brand">
          <span className="brand-icon">🛡️</span>
          <span className="brand-text">DeepGuard<span className="brand-accent"> AI</span></span>
        </NavLink>
        <ul className="navbar-links">
          {links.map(({ to, label }) => (
            <li key={to}>
              <NavLink
                to={to}
                end={to === '/'}
                className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
              >
                {label}
              </NavLink>
            </li>
          ))}
        </ul>
      </div>
    </nav>
  )
}
