import { useEffect, useState } from 'react'
import { checkStatus, checkHealth } from '../services/api'
import './Dashboard.css'

function StatCard({ icon, label, value, sub, color }) {
  return (
    <div className="card stat-card">
      <div className="stat-icon">{icon}</div>
      <div className="stat-info">
        <div className="stat-val" style={{ color: color || 'var(--text-primary)' }}>{value}</div>
        <div className="stat-lbl">{label}</div>
        {sub && <div className="stat-sub">{sub}</div>}
      </div>
    </div>
  )
}

export default function Dashboard() {
  const [status, setStatus] = useState(null)
  const [health, setHealth] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const load = async () => {
      try {
        const [h, s] = await Promise.all([checkHealth(), checkStatus()])
        setHealth(h.data)
        setStatus(s.data)
      } catch {
        setHealth({ status: 'error' })
        setStatus(null)
      } finally { setLoading(false) }
    }
    load()
  }, [])

  const meta = status?.checkpoint_meta

  return (
    <div className="page-container">
      <div className="section-header">
        <div className="label">System Overview</div>
        <h1>Dashboard</h1>
        <p style={{ marginTop: '0.5rem' }}>Live backend status, model info and project progress.</p>
      </div>

      {loading ? (
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', padding: '2rem 0' }}>
          <div className="spinner" />
          <span className="text-muted">Connecting to backend…</span>
        </div>
      ) : (
        <>
          {/* Backend status */}
          <section className="dash-section">
            <h3>Backend Health</h3>
            <div className="grid-3" style={{ marginTop: '1rem' }}>
              <StatCard
                icon={health?.status === 'ok' ? '✅' : '❌'}
                label="API Server"
                value={health?.status === 'ok' ? 'Online' : 'Offline'}
                color={health?.status === 'ok' ? 'var(--success)' : 'var(--danger)'}
              />
              <StatCard
                icon={status?.model_ready ? '🧠' : '⏳'}
                label="CNN Model"
                value={status?.model_ready ? 'Ready' : 'Awaiting Training'}
                color={status?.model_ready ? 'var(--success)' : 'var(--warning)'}
              />
              <StatCard
                icon="💻"
                label="Device"
                value={status?.device || '—'}
                sub="CPU / GPU"
              />
            </div>
          </section>

          {/* Model metadata */}
          <section className="dash-section">
            <h3>Model Details</h3>
            <div className="card">
              {!status?.model_ready ? (
                <div className="alert alert-warning">
                  ⚠️ No trained checkpoint found at <code>{status?.checkpoint_path}</code>.
                  Run <code>scripts/train_model.py</code> once datasets are available.
                </div>
              ) : (
                <table className="meta-table">
                  <tbody>
                    {[
                      ['Architecture', 'ResNet18'],
                      ['Classes', 'REAL (0) · DEEPFAKE (1)'],
                      ['Best Val Accuracy', meta?.best_val_accuracy != null ? `${(meta.best_val_accuracy * 100).toFixed(2)}%` : '—'],
                      ['Epoch', meta?.epoch ?? '—'],
                      ['Checkpoint', status?.checkpoint_path ?? '—'],
                    ].map(([k, v]) => (
                      <tr key={k}><td className="meta-key">{k}</td><td className="meta-val">{v}</td></tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </section>

          {/* Phase tracker */}
          <section className="dash-section">
            <h3>Project Phases</h3>
            <div className="phase-list">
              {[
                { phase: 'Phase 0', name: 'Project Setup', done: true },
                { phase: 'Phase 1', name: 'Streamlit Prototype', done: true },
                { phase: 'Phase 2', name: 'Dataset Preparation', done: true },
                { phase: 'Phase 3', name: 'CNN Spatial Detection', done: true },
                { phase: 'Phase 3A', name: 'React + FastAPI Application', done: true },
                { phase: 'Phase 4', name: 'FFT Frequency Analysis', done: false },
                { phase: 'Phase 5', name: 'DWT Wavelet Analysis', done: false },
                { phase: 'Phase 6', name: 'Feature Fusion & Final Model', done: false },
              ].map(p => (
                <div key={p.phase} className="phase-row">
                  <span className={`phase-dot ${p.done ? 'done' : 'pending'}`} />
                  <span className="phase-tag badge badge-info">{p.phase}</span>
                  <span className={`phase-name ${p.done ? '' : 'text-muted'}`}>{p.name}</span>
                  <span className={`phase-status ${p.done ? 'text-success' : 'text-muted'}`}>
                    {p.done ? '✓ Complete' : 'Pending'}
                  </span>
                </div>
              ))}
            </div>
          </section>

          {/* Dataset */}
          <section className="dash-section">
            <h3>Dataset Status</h3>
            <div className="card">
              <div className="alert alert-info">
                📂 FaceForensics++ and Celeb-DF v2 require manual download (access approval needed).
                See <code>data/README.md</code> for placement instructions.
              </div>
              <div className="grid-2" style={{ marginTop: '1rem' }}>
                {[
                  { ds: 'FaceForensics++', role: 'Training / Validation / Test', status: 'Manual download required' },
                  { ds: 'Celeb-DF v2',     role: 'Independent Evaluation Only',  status: 'Reserved — not for training' },
                ].map(d => (
                  <div key={d.ds} className="card" style={{ border: '1px solid var(--border)' }}>
                    <h4>{d.ds}</h4>
                    <p style={{ marginTop: '0.4rem', fontSize: '0.875rem' }}>{d.role}</p>
                    <div className="badge badge-warning" style={{ marginTop: '0.5rem' }}>{d.status}</div>
                  </div>
                ))}
              </div>
            </div>
          </section>
        </>
      )}
    </div>
  )
}
