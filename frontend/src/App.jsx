import { useMemo, useState, useRef } from 'react'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || ''

const SENTIMENT_CONFIG = {
  positive: {
    label: 'Positive',
    color: '#00e5a0',
    bg: 'rgba(0, 229, 160, 0.08)',
    border: 'rgba(0, 229, 160, 0.25)',
    scoreIndex: 1,
  },
  neutral: {
    label: 'Neutral',
    color: '#4da6ff',
    bg: 'rgba(77, 166, 255, 0.08)',
    border: 'rgba(77, 166, 255, 0.25)',
    scoreIndex: 0,
  },
  negative: {
    label: 'Negative',
    color: '#ff4d6a',
    bg: 'rgba(255, 77, 106, 0.08)',
    border: 'rgba(255, 77, 106, 0.25)',
    scoreIndex: 2,
  },
}

const SCORE_LABELS = [
  { key: 'neutral', label: 'Neutral', color: '#4da6ff', bg: 'rgba(77, 166, 255, 0.06)' },
  { key: 'positive', label: 'Positive', color: '#00e5a0', bg: 'rgba(0, 229, 160, 0.06)' },
  { key: 'negative', label: 'Negative', color: '#ff4d6a', bg: 'rgba(255, 77, 106, 0.06)' },
]

function ConfidenceRing({ value, color }) {
  const CIRCUMFERENCE = 175.9
  const offset = CIRCUMFERENCE - (value / 100) * CIRCUMFERENCE

  return (
    <div className="confidence-ring-wrap">
      <svg viewBox="0 0 60 60" width="72" height="72">
        <circle className="conf-track" cx="30" cy="30" r="28" />
        <circle
          className="conf-fill"
          cx="30"
          cy="30"
          r="28"
          stroke={color}
          strokeDashoffset={offset}
        />
      </svg>
      <div className="conf-label">
        <span className="conf-pct" style={{ color }}>{Math.round(value)}%</span>
        <span className="conf-sub">conf</span>
      </div>
    </div>
  )
}

function ScoreCard({ label, color, bg, value, isActive }) {
  return (
    <div
      className={`score-card ${isActive ? 'active' : ''}`}
      style={isActive ? {
        '--active-color': `${color}55`,
        '--active-bg': bg,
      } : {}}
    >
      <span className="score-label">{label}</span>
      <strong style={{ color: isActive ? color : 'var(--text-primary)' }}>
        {(value * 100).toFixed(1)}%
      </strong>
      <div
        className="score-bar"
        style={{
          width: `${value * 100}%`,
          background: color,
          opacity: isActive ? 0.8 : 0.3,
        }}
      />
    </div>
  )
}

export default function App() {
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)
  const textareaRef = useRef(null)

  const theme = useMemo(() => {
    if (!result?.label) return SENTIMENT_CONFIG.neutral
    return SENTIMENT_CONFIG[result.label] || SENTIMENT_CONFIG.neutral
  }, [result])

  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0
  const charCount = text.trim().length

  const handleAnalyze = async (event) => {
    event.preventDefault()
    const trimmed = text.trim()

    if (!trimmed) {
      setError('Please enter some text to analyze.')
      setResult(null)
      return
    }

    try {
      setLoading(true)
      setError('')
      setResult(null)

      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: trimmed }),
      })

      const raw = await response.text()
      let data = null

      try {
        data = raw ? JSON.parse(raw) : null
      } catch {
        throw new Error(`Backend returned invalid JSON: ${raw || 'empty response'}`)
      }

      if (!response.ok) {
        throw new Error(data?.detail || `Request failed with status ${response.status}`)
      }

      const prediction = data?.predictions?.[0]

      if (!prediction) {
        throw new Error('No prediction returned from backend.')
      }

      setResult(prediction)
    } catch (err) {
      setError(err.message || 'Something went wrong.')
    } finally {
      setLoading(false)
    }
  }

  const handleClear = () => {
    setText('')
    setError('')
    setResult(null)
    textareaRef.current?.focus()
  }

  const confidence = Number(result?.confidence || 0) * 100

  return (
    <div className="page-shell">
      <div className="glow glow-one" />
      <div className="glow glow-two" />

      <header className="header-bar">
        <div className="header-brand">
          <div className="brand-mark">
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <path d="M1 11 L4 7 L7 9 L10 4 L13 6" stroke="#4da6ff" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
          <span className="brand-name">FinSent · v2.1</span>
        </div>
        <div className="header-status">
          <span className="status-dot">Model Online</span>
        </div>
      </header>

      <main className="app-card">
        <section className="hero">
          <p className="hero-eyebrow">
            <span className="eyebrow-icon">
              <span/><span/><span/>
            </span>
            Financial NLP Engine
          </p>
          <h1>Sentiment<br />Analysis</h1>
          <p className="subtitle">
            Analyze financial text — earnings calls, market news, analyst reports —
            and classify sentiment with confidence scoring.
          </p>
        </section>

        <div className="workspace">
          <form className="input-panel" onSubmit={handleAnalyze}>
            <p className="panel-title">
              <svg className="panel-title-icon" viewBox="0 0 18 18" fill="none">
                <rect x="2" y="2" width="14" height="10" rx="2" stroke="currentColor" strokeWidth="1.3"/>
                <path d="M5 16 L9 13 L13 16" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              Text Input
            </p>

            <label htmlFor="sentiment-input" style={{ display: 'none' }}>Enter financial text</label>
            <textarea
              ref={textareaRef}
              id="sentiment-input"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Paste a financial excerpt — e.g. &quot;Q3 revenue exceeded expectations, driven by strong institutional demand and improved margins.&quot;"
              rows={9}
            />

            <div className="form-meta">
              <div className="meta-item">
                <strong>{charCount}</strong>
                <span>chars</span>
              </div>
              <div className="meta-divider" />
              <div className="meta-item">
                <strong>{wordCount}</strong>
                <span>words</span>
              </div>
            </div>

            <div className="button-row">
              <button type="submit" className="primary-btn" disabled={loading}>
                {loading ? (
                  <span className="btn-loading">
                    <span className="btn-spinner" />
                    Analyzing
                    <span className="loading-dots">
                      <span>.</span><span>.</span><span>.</span>
                    </span>
                  </span>
                ) : (
                  'Run Analysis'
                )}
              </button>
              <button
                type="button"
                className="secondary-btn"
                onClick={handleClear}
                disabled={loading}
              >
                Clear
              </button>
            </div>

            {error && (
              <div className="alert error-alert" role="alert">
                <svg className="alert-icon" viewBox="0 0 16 16" fill="none">
                  <circle cx="8" cy="8" r="7" stroke="#ff4d6a" strokeWidth="1.3"/>
                  <path d="M8 5V8.5M8 11h.01" stroke="#ff4d6a" strokeWidth="1.5" strokeLinecap="round"/>
                </svg>
                {error}
              </div>
            )}
          </form>

          <section className="result-panel">
            <div className="panel-header">
              <p className="panel-title" style={{ margin: 0 }}>
                <svg className="panel-title-icon" viewBox="0 0 18 18" fill="none">
                  <path d="M3 13 L6.5 8.5 L9.5 11 L13 6 L15 8" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round"/>
                  <rect x="2" y="2" width="14" height="14" rx="2" stroke="currentColor" strokeWidth="1.3"/>
                </svg>
                Prediction Output
              </p>
              {result && (
                <span className="result-badge">
                  {result.label?.toUpperCase()}
                </span>
              )}
            </div>

            {!result && !loading && !error && (
              <div className="empty-state">
                <div className="empty-icon">
                  <svg width="22" height="22" viewBox="0 0 22 22" fill="none">
                    <path d="M3 16 L7 11 L10 14 L14 8 L19 10" stroke="#4a6580" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </div>
                <p>Awaiting input</p>
                <span>Results will appear here after analysis</span>
              </div>
            )}

            {loading && (
              <div className="loading-state">
                <div className="loading-rings">
                  <div className="ring ring-1" />
                  <div className="ring ring-2" />
                  <div className="ring ring-3" />
                </div>
                <p>
                  Processing
                  <span className="loading-dots">
                    <span>.</span><span>.</span><span>.</span>
                  </span>
                </p>
              </div>
            )}

            {result && (
              <div
                className="result-card"
                style={{ backgroundColor: theme.bg, borderColor: theme.border }}
              >
                <div className="result-header">
                  <div
                    className="sentiment-pill"
                    style={{ color: theme.color, borderColor: `${theme.color}66` }}
                  >
                    <span className="pill-dot" style={{ color: theme.color }} />
                    {theme.label}
                  </div>
                  <ConfidenceRing value={confidence} color={theme.color} />
                </div>

                <div className="result-body">
                  <div className="result-text-block">
                    <p className="section-title">Analyzed text</p>
                    <p
                      className="analyzed-text-content"
                      style={{ color: theme.color, borderLeftColor: `${theme.color}66` }}
                    >
                      {result.text}
                    </p>
                  </div>

                  {Array.isArray(result.scores) && (
                    <div>
                      <p
                        className="section-title"
                        style={{
                          fontFamily: "'Space Mono', monospace",
                          fontSize: '0.68rem',
                          letterSpacing: '0.1em',
                          textTransform: 'uppercase',
                          color: 'var(--text-muted)',
                          margin: '0 0 10px',
                          display: 'block'
                        }}
                      >
                        Score Breakdown
                      </p>
                      <div className="scores-grid">
                        {SCORE_LABELS.map((s, i) => (
                          <ScoreCard
                            key={s.key}
                            label={s.label}
                            color={s.color}
                            bg={s.bg}
                            value={Number(result.scores[i] || 0)}
                            isActive={result.label === s.key}
                          />
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </section>
        </div>
      </main>
    </div>
  )
}