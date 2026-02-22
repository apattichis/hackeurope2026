import { useState, useEffect, useCallback } from 'react'
import PipelineOverview from './components/PipelineOverview'
import LineageTree from './components/LineageTree'
import TearSheet from './components/TearSheet'
import Leaderboard from './components/Leaderboard'
import CorrelationMatrix from './components/CorrelationMatrix'
import RegimeExplainer from './components/RegimeExplainer'

const TABS = [
  { id: 'pipeline', label: 'Pipeline Overview' },
  { id: 'lineage', label: 'Lineage Tree' },
  { id: 'regimes', label: 'Regime Filters' },
  { id: 'tearsheet', label: 'Strategy Tear Sheet' },
  { id: 'leaderboard', label: 'Leaderboard' },
  { id: 'correlation', label: 'Correlation Matrix' },
]

/* ── Simulation steps (condensed) ── */
const SIM_STEPS = [
  { text: 'Loading 36,311 bars...', duration: 350 },
  { text: 'Generating 12 strategies across 4 families...', duration: 450 },
  { text: 'Selecting 4 champions...', duration: 300 },
  { text: 'Building 3 hybrid templates...', duration: 350 },
  { text: 'Applying regime filter and ranking...', duration: 400 },
]

const SIM_RESULTS = [
  { rank: '#1', name: 'Consensus Gate', fitness: '1.037', sharpe: '1.76', color: '#fbbf24' },
  { rank: '#2', name: 'Regime Router', fitness: '0.120', sharpe: '0.54', color: '#94a3b8' },
  { rank: '#3', name: 'Weighted Combination', fitness: '-0.159', sharpe: '-1.76', color: '#78716c' },
]

function PipelineSimulation({ onComplete }) {
  const [step, setStep] = useState(-1)
  const [showRanking, setShowRanking] = useState(false)
  const [rankVisible, setRankVisible] = useState(-1)
  const [fadeOut, setFadeOut] = useState(false)

  const run = useCallback(() => {
    setStep(0)
  }, [])

  useEffect(() => {
    if (step < 0) return
    if (step >= SIM_STEPS.length) {
      const t1 = setTimeout(() => setShowRanking(true), 150)
      const t2 = setTimeout(() => setRankVisible(0), 300)
      const t3 = setTimeout(() => setRankVisible(1), 450)
      const t4 = setTimeout(() => setRankVisible(2), 600)
      const t5 = setTimeout(() => {
        setFadeOut(true)
        setTimeout(onComplete, 500)
      }, 1600)
      return () => { clearTimeout(t1); clearTimeout(t2); clearTimeout(t3); clearTimeout(t4); clearTimeout(t5) }
    }
    const t = setTimeout(() => setStep(step + 1), SIM_STEPS[step].duration)
    return () => clearTimeout(t)
  }, [step, onComplete])

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-slate-950 transition-opacity duration-500"
      style={{ opacity: fadeOut ? 0 : 1 }}
    >
      <div className="w-full max-w-md mx-auto px-8">
        {/* Logo + title */}
        <div className="flex items-center gap-3 mb-6">
          <span className="text-3xl font-bold select-none" style={{ fontFamily: 'serif', background: 'linear-gradient(135deg, #60a5fa, #a78bfa)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
            α
          </span>
          <span className="text-xl font-bold text-white tracking-tight">Council of Alphas</span>
        </div>

        {step < 0 ? (
          <div className="flex flex-col items-start gap-4">
            <p className="text-sm text-slate-400 leading-relaxed">
              Evolutionary multi-agent trading framework.<br />
              SOL/USD, 1h candles, 4 years of data.
            </p>
            <button
              onClick={run}
              className="px-6 py-2.5 rounded-lg font-semibold text-sm text-white transition-all duration-200 hover:shadow-lg hover:shadow-blue-600/20 cursor-pointer"
              style={{ background: 'linear-gradient(135deg, #2563eb, #7c3aed)' }}
            >
              <span className="flex items-center gap-2">
                Run Pipeline
                <svg width="14" height="14" viewBox="0 0 24 24">
                  <polygon points="5 3 19 12 5 21 5 3" fill="currentColor" />
                </svg>
              </span>
            </button>
          </div>
        ) : (
          <div className="font-mono text-[13px]">
            {SIM_STEPS.map((s, i) => {
              if (i > step) return null
              const done = i < step
              return (
                <div key={i} className="flex items-center gap-3 h-7" style={{ animation: 'fadeSlideIn 0.2s ease-out' }}>
                  <span className="w-4 text-center flex-shrink-0">
                    {done ? (
                      <span className="text-emerald-400">&#10003;</span>
                    ) : (
                      <span className="text-blue-400 animate-pulse">&#9679;</span>
                    )}
                  </span>
                  <span className={done ? 'text-slate-600' : 'text-slate-200'}>{s.text}</span>
                </div>
              )
            })}

            {showRanking && (
              <div className="mt-4 pt-3 border-t border-slate-800">
                <div className="text-[10px] text-slate-500 uppercase tracking-widest mb-2 pl-7">Ranked Survivors</div>
                {SIM_RESULTS.map((r, i) => (
                  <div
                    key={i}
                    className="flex items-baseline h-7 pl-7 transition-all duration-300"
                    style={{ opacity: rankVisible >= i ? 1 : 0, transform: rankVisible >= i ? 'translateX(0)' : 'translateX(-8px)' }}
                  >
                    <span className="font-black text-sm w-7" style={{ color: r.color }}>{r.rank}</span>
                    <span className="text-slate-200 text-sm w-44">{r.name}</span>
                    <span className="text-xs text-slate-600">
                      S <span className={parseFloat(r.sharpe) >= 0 ? 'text-emerald-400' : 'text-red-400'}>{r.sharpe}</span>
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default function App() {
  const [activeTab, setActiveTab] = useState('pipeline')
  const [simDone, setSimDone] = useState(false)

  return (
    <div className="min-h-screen bg-slate-950 text-slate-300 font-[Inter]">
      {/* Simulation overlay */}
      {!simDone && <PipelineSimulation onComplete={() => setSimDone(true)} />}

      {/* Top bar */}
      <header className="sticky top-0 z-50 bg-slate-950/95 backdrop-blur-sm border-b border-slate-800">
        <div className="max-w-[1400px] mx-auto px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-xl font-bold select-none flex items-center" style={{ fontFamily: 'serif', background: 'linear-gradient(135deg, #60a5fa, #a78bfa)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', lineHeight: 1 }}>α</span>
            <h1 className="text-lg font-bold text-white tracking-tight leading-none">Council of Alphas</h1>
            <span className="text-xs text-slate-500 font-mono border border-slate-700 rounded px-2 py-0.5">
              HackEurope 2026
            </span>
          </div>
          <nav className="flex gap-1">
            {TABS.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-3 py-1.5 text-sm rounded-md transition-all duration-200 ${
                  activeTab === tab.id
                    ? 'bg-blue-600 text-white font-medium shadow-lg shadow-blue-600/20'
                    : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      {/* Content */}
      <main className="max-w-[1400px] mx-auto px-6 py-6">
        {activeTab === 'pipeline' && <PipelineOverview />}
        {activeTab === 'lineage' && <LineageTree />}
        {activeTab === 'regimes' && <RegimeExplainer />}
        {activeTab === 'tearsheet' && <TearSheet />}
        {activeTab === 'leaderboard' && <Leaderboard />}
        {activeTab === 'correlation' && <CorrelationMatrix />}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800 mt-12">
        <div className="max-w-[1400px] mx-auto px-6 py-4 flex items-center justify-between text-xs text-slate-600">
          <span className="flex items-center gap-2">
            Team: The Greeks (<a href="https://github.com/apattichis" target="_blank" rel="noopener noreferrer" className="text-slate-500 hover:text-slate-300 transition-colors underline underline-offset-2">Andreas</a> + <a href="https://github.com/MarkosMarkides" target="_blank" rel="noopener noreferrer" className="text-slate-500 hover:text-slate-300 transition-colors underline underline-offset-2">Markos</a>)
            <a href="https://github.com/apattichis/hackeurope2026" target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1 text-slate-500 hover:text-slate-300 transition-colors underline underline-offset-2">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/></svg>
              Source Code
            </a>
          </span>
          <a href="https://www.hackeurope.com/" target="_blank" rel="noopener noreferrer" className="font-mono text-slate-500 hover:text-slate-300 transition-colors underline underline-offset-2">HackEurope 2026</a>
        </div>
      </footer>
    </div>
  )
}
