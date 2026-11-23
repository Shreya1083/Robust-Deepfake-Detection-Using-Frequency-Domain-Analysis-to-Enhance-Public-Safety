import { createContext, useContext, useState, useEffect } from 'react'
import { apiGetMe, apiLogin, apiSignUp, apiLogout } from '../services/api'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)

  // Check current session on mount via HttpOnly cookie
  useEffect(() => {
    async function loadUser() {
      try {
        const { data } = await apiGetMe()
        setUser(data)
      } catch (err) {
        setUser(null)
      } finally {
        setLoading(false)
      }
    }
    loadUser()
  }, [])

  const login = async (credentials) => {
    const { data } = await apiLogin(credentials)
    if (data.user) {
      setUser(data.user)
    }
    return data
  }

  const signup = async (userData) => {
    const { data } = await apiSignUp(userData)
    return data
  }

  const logout = async () => {
    try {
      await apiLogout()
    } finally {
      setUser(null)
    }
  }

  return (
    <AuthContext.Provider value={{ user, loading, login, signup, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}
