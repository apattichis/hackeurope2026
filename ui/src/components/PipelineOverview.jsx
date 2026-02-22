import { config, familyIndicators, hybrids } from '../data.js';

const FAMILY_COLORS = {
  trend:      { bg: 'bg-blue-500/10',    border: 'border-blue-500/20',    text: 'text-blue-400' },
  momentum:   { bg: 'bg-purple-500/10',  border: 'border-purple-500/20',  text: 'text-purple-400' },
  volatility: { bg: 'bg-amber-500/10',   border: 'border-amber-500/20',   text: 'text-amber-400' },
  volume:     { bg: 'bg-emerald-500/10', border: 'border-emerald-500/20', text: 'text-emerald-400' },
};

function ParamBadge({ children }) {
  return (
    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-mono bg-slate-800/60 border border-slate-700/40 text-slate-400">
      {children}
    </span>
  );
}

function StageCard({ number, title, subtitle, children, accent }) {
  return (
    <div className="flex flex-col bg-slate-900/50 border border-slate-800/80 rounded-lg p-3.5 min-w-0 flex-1 group hover:border-slate-700/80 transition-colors">
      <div className="flex items-center gap-2 mb-2.5">
        <span className={`inline-flex items-center justify-center w-5 h-5 rounded-full text-[10px] font-bold flex-shrink-0 ${
          accent ? 'bg-blue-600 text-white' : 'bg-slate-800 text-slate-400 border border-slate-700'
        }`}>
          {number}
        </span>
        <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest">
          Stage {number}
        </span>
      </div>
      <div className="text-sm font-semibold text-slate-200 leading-snug">{title}</div>
      <div className="text-[11px] text-slate-500 mt-0.5 mb-3">{subtitle}</div>
      <div className="mt-auto">{children}</div>
    </div>
  );
}

function Arrow() {
  return (
    <div className="flex items-center justify-center flex-shrink-0 px-0.5">
      <svg width="16" height="16" viewBox="0 0 16 16" className="text-slate-700">
        <path d="M6 3l5 5-5 5" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </div>
  );
}

function FamilyPill({ family }) {
  const c = FAMILY_COLORS[family] || FAMILY_COLORS.trend;
  const count = familyIndicators[family]?.length ?? 0;
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono border ${c.bg} ${c.border} ${c.text}`}>
      {family}
      <span className="text-slate-600">({count})</span>
    </span>
  );
}

export default function PipelineOverview() {
  const feePercent = (config.fee * 100).toFixed(3) + '%';
  const riskPercent = (config.riskPerTrade * 100).toFixed(1) + '%';

  return (
    <div className="flex flex-col gap-7 text-slate-300">

      {/* Hero */}
      <div>
        <h1 className="text-2xl font-bold text-white tracking-tight">Council of Alphas</h1>
        <p className="text-slate-500 text-sm mt-1 mb-3">Evolutionary Multi-Agent Trading Framework</p>
        <div className="flex flex-wrap gap-1.5">
          <ParamBadge>{config.asset} {config.timeframe}</ParamBadge>
          <ParamBadge>{config.dateRange}</ParamBadge>
          <ParamBadge>{config.bars.toLocaleString()} bars</ParamBadge>
          <ParamBadge>${(config.initialCapital / 1000).toFixed(0)}k initial</ParamBadge>
          <ParamBadge>{riskPercent} risk</ParamBadge>
          <ParamBadge>{feePercent} fee</ParamBadge>
          <ParamBadge>{config.model}</ParamBadge>
          <ParamBadge>{config.runtime}s runtime</ParamBadge>
        </div>
      </div>

      {/* Pipeline flow */}
      <div>
        <h2 className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest mb-3">Pipeline Architecture</h2>
        <div className="flex items-stretch gap-0.5">

          <StageCard number={1} title="12 Strategies Generated" subtitle="3 per family, Claude Opus temp=0">
            <div className="flex flex-wrap gap-1">
              {Object.keys(familyIndicators).map((f) => (
                <FamilyPill key={f} family={f} />
              ))}
            </div>
          </StageCard>

          <Arrow />

          <StageCard number={2} title="4 Champions Selected" subtitle="Best fitness per family">
            <div className="flex flex-col gap-0.5">
              {Object.keys(familyIndicators).map((f) => {
                const c = FAMILY_COLORS[f] || FAMILY_COLORS.trend;
                return (
                  <div key={f} className="flex items-center gap-1.5">
                    <span className={`w-1.5 h-1.5 rounded-full ${c.text.replace('text-', 'bg-')}`} />
                    <span className={`text-[10px] font-mono ${c.text}`}>{f}</span>
                  </div>
                );
              })}
            </div>
          </StageCard>

          <Arrow />

          <StageCard number={3} title="3 Hybrids Built" subtitle="Pure Python, zero LLM">
            <div className="flex flex-col gap-0.5">
              {['Regime Router', 'Consensus Gate', 'Weighted Combination'].map((name) => (
                <div key={name} className="flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                  <span className="text-[10px] font-mono text-emerald-400">{name}</span>
                </div>
              ))}
            </div>
          </StageCard>

          <Arrow />

          <StageCard number={4} title="2D Regime Filter" subtitle="Deterministic optimization">
            <p className="text-[10px] text-slate-500 leading-relaxed">
              Keeps trades only in 2D regime buckets where Sharpe &gt; 0 and evidence is sufficient. No lookahead.
            </p>
          </StageCard>

          <Arrow />

          <StageCard number={5} title="3 Survivors Ranked" subtitle="By fitness score">
            <div className="text-[10px] text-slate-500 mb-1.5 font-mono">
              Fitness = Sharpe &times; ln(N) &times; Coverage
            </div>
            {hybrids.map((h) => (
              <div key={h.name} className="flex items-center gap-1.5">
                <span className="font-mono text-[10px] text-slate-600">#{h.rank}</span>
                <span className="text-[10px] text-emerald-400 font-mono">{h.displayName}</span>
              </div>
            ))}
          </StageCard>

        </div>
      </div>

      {/* Config details */}
      <div>
        <h2 className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest mb-3">Configuration</h2>
        <div className="grid grid-cols-2 gap-3">

          {/* Triple Barrier */}
          <div className="bg-slate-900/40 border border-slate-800/60 rounded-lg p-4">
            <div className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider mb-3">Triple Barrier Method</div>
            <div className="flex flex-col gap-2">
              {[
                { label: 'Take Profit', value: `${config.tbmWin}x ATR`, cls: 'text-emerald-400' },
                { label: 'Stop Loss', value: `${config.tbmLoss}x ATR`, cls: 'text-red-400' },
                { label: 'Time Horizon', value: `${config.tbmHorizon}h`, cls: 'text-slate-300' },
                { label: 'Risk/Trade', value: riskPercent, cls: 'text-slate-300' },
                { label: 'Taker Fee', value: feePercent, cls: 'text-slate-300' },
              ].map((row, i) => (
                <div key={row.label}>
                  <div className="flex justify-between items-center">
                    <span className="text-[11px] text-slate-500">{row.label}</span>
                    <span className={`font-mono text-[11px] ${row.cls}`}>{row.value}</span>
                  </div>
                  {i < 4 && <div className="w-full h-px bg-slate-800/60 mt-2" />}
                </div>
              ))}
            </div>
          </div>

          {/* Indicator pools */}
          <div className="bg-slate-900/40 border border-slate-800/60 rounded-lg p-4">
            <div className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider mb-3">Indicator Pools by Family</div>
            <div className="flex flex-col gap-2.5">
              {Object.entries(familyIndicators).map(([family, indicators]) => {
                const c = FAMILY_COLORS[family] || FAMILY_COLORS.trend;
                return (
                  <div key={family}>
                    <div className={`text-[10px] font-semibold mb-1 ${c.text} uppercase tracking-wider`}>{family}</div>
                    <div className="flex flex-wrap gap-1">
                      {indicators.map((ind) => (
                        <span
                          key={ind}
                          className={`px-1.5 py-px rounded text-[10px] font-mono border ${c.bg} ${c.border} ${c.text}`}
                        >
                          {ind}
                        </span>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}
