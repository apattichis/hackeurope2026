import { speciation, champions, hybrids } from '../data.js';

const FAMILY_META = {
  trend:      { label: 'Trend',      color: '#60a5fa', border: '#2563eb40', bg: '#1e3a5f18' },
  momentum:   { label: 'Momentum',   color: '#c084fc', border: '#7e22ce40', bg: '#2e1a4a18' },
  volatility: { label: 'Volatility', color: '#fbbf24', border: '#b4530940', bg: '#3b2a0a18' },
  volume:     { label: 'Volume',     color: '#22d3ee', border: '#0e749040', bg: '#0a2d3518' },
};

const FAMILIES = ['trend', 'momentum', 'volatility', 'volume'];

function fmt(n) {
  if (n == null) return '--';
  return (n >= 0 ? '+' : '') + n.toFixed(4);
}

function truncate(str, max = 24) {
  if (!str) return '';
  const s = str.replace(/_/g, ' ');
  return s.length > max ? s.slice(0, max - 1) + '\u2026' : s;
}

/* ── connectors ── */
function VerticalConnector({ label, color = '#334155' }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', margin: '5px 0' }}>
      <div style={{ width: 1, height: 12, background: color }} />
      {label && (
        <span style={{ fontSize: 12, color: '#475569', fontFamily: 'monospace', letterSpacing: '0.04em', margin: '1px 0' }}>
          {label}
        </span>
      )}
      <div style={{ width: 1, height: 6, background: color }} />
      <div style={{ width: 0, height: 0, borderLeft: '3px solid transparent', borderRight: '3px solid transparent', borderTop: `5px solid ${color}` }} />
    </div>
  );
}

function ChampionConnector() {
  return (
    <div style={{ display: 'flex', width: '100%', gap: 6, margin: '4px 0' }}>
      {FAMILIES.map((f) => (
        <div key={f} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <div style={{ width: 1, height: 14, background: '#10b981', opacity: 0.5 }} />
          <div style={{ width: 0, height: 0, borderLeft: '3px solid transparent', borderRight: '3px solid transparent', borderTop: '4px solid #10b981', opacity: 0.6 }} />
        </div>
      ))}
    </div>
  );
}

function FanOutConnector() {
  return (
    <div style={{ width: '100%', margin: '3px 0', display: 'flex', flexDirection: 'column', alignItems: 'stretch' }}>
      <div style={{ display: 'flex', gap: 6 }}>
        {[0, 1, 2, 3].map((i) => (
          <div key={i} style={{ flex: 1, display: 'flex', justifyContent: 'center' }}>
            <div style={{ width: 1, height: 10, background: '#334155' }} />
          </div>
        ))}
      </div>
      <div style={{ position: 'relative', height: 1, background: '#334155', margin: '0 12%' }} />
      <div style={{ display: 'flex', justifyContent: 'center', marginTop: -6, position: 'relative', zIndex: 1 }}>
        <span style={{ fontSize: 12, color: '#475569', background: '#020617', padding: '0 6px', fontFamily: 'monospace', letterSpacing: '0.04em' }}>
          all 4 champions feed each hybrid
        </span>
      </div>
      <div style={{ display: 'flex', gap: 6, marginTop: 6 }}>
        <div style={{ flex: 1 }} />
        {[0, 1, 2].map((i) => (
          <div key={i} style={{ flex: 2, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <div style={{ width: 1, height: 8, background: '#334155' }} />
            <div style={{ width: 0, height: 0, borderLeft: '3px solid transparent', borderRight: '3px solid transparent', borderTop: '4px solid #334155' }} />
          </div>
        ))}
        <div style={{ flex: 1 }} />
      </div>
    </div>
  );
}

/* ── Layer 1: Speciation ── */
function StrategyCard({ strategy }) {
  const isCh = strategy.isChampion;
  return (
    <div
      style={{
        border: isCh ? '1px solid #059669' : '1px solid #1e293b',
        borderRadius: 5,
        padding: '5px 7px',
        marginBottom: 4,
        background: isCh ? '#052e1640' : '#0f172a80',
        position: 'relative',
      }}
    >
      {isCh && (
        <span style={{
          position: 'absolute', top: 3, right: 5, fontSize: 7, fontWeight: 700, letterSpacing: '0.06em',
          color: '#10b981', background: '#052e16', border: '1px solid #059669', borderRadius: 2, padding: '0px 4px',
        }}>
          CHAMP
        </span>
      )}
      <div
        style={{
          fontSize: 12, fontWeight: 600, color: isCh ? '#bbf7d0' : '#64748b',
          fontFamily: 'monospace', lineHeight: 1.3, paddingRight: isCh ? 44 : 0,
        }}
        title={strategy.name.replace(/_/g, ' ')}
      >
        {truncate(strategy.name, 20)}
      </div>
      <div style={{ display: 'flex', gap: 6, marginTop: 2, fontSize: 12, fontFamily: 'monospace', color: '#475569' }}>
        <span>fit <span style={{ color: isCh ? '#34d399' : '#475569', fontWeight: 600 }}>{fmt(strategy.fitness)}</span></span>
        <span>{strategy.trades}t</span>
      </div>
    </div>
  );
}

function FamilyColumn({ family }) {
  const m = FAMILY_META[family];
  const strats = speciation[family] || [];
  return (
    <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column' }}>
      <div style={{
        textAlign: 'center', fontSize: 12, fontWeight: 700, letterSpacing: '0.08em',
        color: m.color, background: m.bg, border: `1px solid ${m.border}`,
        borderRadius: 4, padding: '3px 0', marginBottom: 6,
      }}>
        {m.label.toUpperCase()}
      </div>
      {strats.map((s, i) => (
        <StrategyCard key={i} strategy={s} />
      ))}
    </div>
  );
}

/* ── Layer 2: Champions ── */
function ChampionCard({ champion }) {
  const m = FAMILY_META[champion.family];
  return (
    <div style={{
      flex: 1, border: '1px solid #065f4620', borderRadius: 5, padding: '6px 8px',
      background: '#0a1f1560', minWidth: 0,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 5, marginBottom: 4 }}>
        <span style={{
          fontSize: 12, fontWeight: 700, color: m.color, background: m.bg,
          border: `1px solid ${m.border}`, borderRadius: 2, padding: '0px 5px', letterSpacing: '0.06em',
        }}>
          {m.label.toUpperCase()}
        </span>
      </div>
      <div style={{ fontSize: 12, fontWeight: 600, color: '#bbf7d0', fontFamily: 'monospace', lineHeight: 1.3, marginBottom: 3 }}
        title={champion.name.replace(/_/g, ' ')}>
        {truncate(champion.name, 20)}
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 2, marginBottom: 3 }}>
        {(champion.indicators || []).map((ind, i) => (
          <span key={i} style={{
            fontSize: 12, color: '#94a3b8', background: '#1e293b', border: '1px solid #334155',
            borderRadius: 2, padding: '0px 3px', fontFamily: 'monospace',
          }}>
            {ind}
          </span>
        ))}
      </div>
      <div style={{ display: 'flex', gap: 6, fontSize: 12, fontFamily: 'monospace', color: '#475569' }}>
        <span>S <span style={{ color: champion.sharpe >= 0 ? '#34d399' : '#f87171' }}>{fmt(champion.sharpe)}</span></span>
        <span>WR <span style={{ color: '#94a3b8' }}>{champion.winRate?.toFixed(1)}%</span></span>
        <span>{champion.trades}t</span>
      </div>
    </div>
  );
}

/* ── Layer 3: Hybrids ── */
function HybridCard({ hybrid }) {
  return (
    <div style={{
      flex: 1, border: '1px solid #1e293b', borderRadius: 5, padding: '6px 8px',
      background: '#0f172a80', minWidth: 0,
    }}>
      <div style={{ fontSize: 13, fontWeight: 700, color: '#e2e8f0', fontFamily: 'monospace', marginBottom: 2 }}>
        {hybrid.displayName}
      </div>
      <div style={{ fontSize: 12, color: '#64748b', lineHeight: 1.4, marginBottom: 4 }}>
        {hybrid.description}
      </div>
      <div style={{ display: 'flex', gap: 8, fontSize: 12, fontFamily: 'monospace', color: '#475569' }}>
        <span>ann.S <span style={{ color: hybrid.annSharpe >= 0 ? '#34d399' : '#f87171', fontWeight: 600 }}>{hybrid.annSharpe?.toFixed(2)}</span></span>
        <span>{hybrid.trades}t</span>
        <span>ret <span style={{ color: hybrid.totalReturn >= 0 ? '#34d399' : '#f87171' }}>
          {hybrid.totalReturn >= 0 ? '+' : ''}{hybrid.totalReturn?.toFixed(1)}%
        </span></span>
      </div>
    </div>
  );
}

/* ── Regime filter row ── */
function RegimeFilterRow({ sorted }) {
  return (
    <div style={{ display: 'flex', gap: 6, width: '100%', margin: '3px 0' }}>
      {sorted.map((h) => {
        const n = h.filteredBuckets?.length || 0;
        return (
          <div key={h.name} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <div style={{ width: 1, height: 8, background: '#334155' }} />
            <span style={{
              fontSize: 12, color: '#64748b', background: '#0f172a', border: '1px solid #1e293b',
              borderRadius: 3, padding: '1px 5px', fontFamily: 'monospace', whiteSpace: 'nowrap',
            }}>
              {n === 0 ? 'no filter' : `${n} buckets filtered`}
            </span>
            <div style={{ width: 1, height: 6, background: '#334155' }} />
            <div style={{ width: 0, height: 0, borderLeft: '3px solid transparent', borderRight: '3px solid transparent', borderTop: '4px solid #334155' }} />
          </div>
        );
      })}
    </div>
  );
}

/* ── Layer 4: Final ranking ── */
const RANK = [
  { border: '#92400e', bg: '#1c120060', num: '#fbbf24', name: '#fef3c7' },
  { border: '#334155', bg: '#0f172a60', num: '#94a3b8', name: '#cbd5e1' },
  { border: '#2c2520', bg: '#1c191760', num: '#78716c', name: '#a8a29e' },
];

function FinalCard({ hybrid, rank }) {
  const r = RANK[rank - 1] || RANK[2];
  return (
    <div style={{ flex: 1, border: `1.5px solid ${r.border}`, borderRadius: 5, padding: '6px 8px', background: r.bg, minWidth: 0 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 5, marginBottom: 3 }}>
        <span style={{ fontSize: 16, fontWeight: 900, color: r.num, fontFamily: 'monospace', lineHeight: 1 }}>#{rank}</span>
        <span style={{ fontSize: 13, fontWeight: 700, color: r.name, fontFamily: 'monospace' }}>{hybrid.displayName}</span>
      </div>
      <div style={{ display: 'flex', gap: 8, fontSize: 12, fontFamily: 'monospace', color: '#475569' }}>
        <span>fitness <span style={{ color: hybrid.fitness >= 0 ? '#34d399' : '#f87171', fontWeight: 700 }}>{fmt(hybrid.fitness)}</span></span>
        <span>ann.S <span style={{ color: hybrid.annSharpe >= 0 ? '#34d399' : '#f87171', fontWeight: 700 }}>{hybrid.annSharpe?.toFixed(2)}</span></span>
      </div>
      <div style={{ display: 'flex', gap: 8, fontSize: 12, fontFamily: 'monospace', color: '#475569', marginTop: 1 }}>
        <span>{hybrid.trades}t</span>
        <span>WR {hybrid.winRate?.toFixed(1)}%</span>
        <span>ret <span style={{ color: hybrid.totalReturn >= 0 ? '#34d399' : '#f87171' }}>
          {hybrid.totalReturn >= 0 ? '+' : ''}{hybrid.totalReturn?.toFixed(1)}%
        </span></span>
        <span>DD <span style={{ color: '#f87171' }}>{hybrid.maxDrawdown?.toFixed(1)}%</span></span>
      </div>
    </div>
  );
}

/* ── Section label ── */
function LayerLabel({ text, sub }) {
  return (
    <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 5, paddingLeft: 1 }}>
      <span style={{ fontSize: 12, fontWeight: 700, color: '#334155', letterSpacing: '0.1em', fontFamily: 'monospace' }}>
        {text}
      </span>
      <span style={{ fontSize: 12, color: '#1e293b', fontFamily: 'monospace' }}>{sub}</span>
    </div>
  );
}

/* ── Main ── */
export default function LineageTree() {
  const sorted = [...hybrids].sort((a, b) => a.rank - b.rank);

  return (
    <div style={{ background: '#020617', minHeight: '100%', padding: '16px 10px', fontFamily: 'ui-sans-serif, system-ui, sans-serif', color: '#e2e8f0', overflowX: 'hidden' }}>

      <div style={{ marginBottom: 14 }}>
        <h2 style={{ fontSize: 16, fontWeight: 700, color: '#f8fafc', margin: 0, letterSpacing: '-0.01em' }}>
          Pipeline Lineage
        </h2>
        <div style={{ fontSize: 12, color: '#334155', marginTop: 2, fontFamily: 'monospace' }}>
          {"12 candidates -> 4 champions -> 3 hybrids -> regime filter -> ranked survivors"}
        </div>
      </div>

      {/* Layer 1 */}
      <LayerLabel text="LAYER 1" sub="Speciation (12 strategies, 3 per family)" />
      <div style={{ overflowX: 'auto', WebkitOverflowScrolling: 'touch' }}>
        <div style={{ display: 'flex', gap: 6, minWidth: 600 }}>
          {FAMILIES.map((f) => <FamilyColumn key={f} family={f} />)}
        </div>
      </div>

      <ChampionConnector />

      {/* Layer 2 */}
      <LayerLabel text="LAYER 2" sub="Champions (1 per family)" />
      <div style={{ overflowX: 'auto', WebkitOverflowScrolling: 'touch' }}>
        <div style={{ display: 'flex', gap: 6, minWidth: 600 }}>
          {champions.map((c, i) => <ChampionCard key={i} champion={c} />)}
        </div>
      </div>

      <FanOutConnector />

      {/* Layer 3 */}
      <LayerLabel text="LAYER 3" sub="Hybrid templates" />
      <div style={{ overflowX: 'auto', WebkitOverflowScrolling: 'touch' }}>
        <div style={{ display: 'flex', gap: 6, minWidth: 500 }}>
          {sorted.map((h) => <HybridCard key={h.name} hybrid={h} />)}
        </div>
      </div>

      <div style={{ overflowX: 'auto', WebkitOverflowScrolling: 'touch' }}>
        <div style={{ minWidth: 600 }}>
          <RegimeFilterRow sorted={sorted} />
        </div>
      </div>

      {/* Layer 4 */}
      <LayerLabel text="LAYER 4" sub="Final ranking (post regime filter)" />
      <div style={{ overflowX: 'auto', WebkitOverflowScrolling: 'touch' }}>
        <div style={{ display: 'flex', gap: 6, minWidth: 600 }}>
          {sorted.map((h) => <FinalCard key={h.rank} hybrid={h} rank={h.rank} />)}
        </div>
      </div>

    </div>
  );
}
