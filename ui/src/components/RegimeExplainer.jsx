import { routingTable, hybrids, config } from '../data.js';

const FAMILY_META = {
  trend:      { label: 'Trend',      color: '#60a5fa', bg: '#1e3a5f30' },
  momentum:   { label: 'Momentum',   color: '#c084fc', bg: '#2e1a4a30' },
  volatility: { label: 'Volatility', color: '#fbbf24', bg: '#3b2a0a30' },
  volume:     { label: 'Volume',     color: '#22d3ee', bg: '#0a2d3530' },
};

const SESSIONS = ['ASIA', 'LONDON', 'NY', 'OTHER'];
const TRENDS = ['UPTREND', 'DOWNTREND', 'CONSOLIDATION'];
const VOLS = ['HIGH_VOL', 'LOW_VOL'];

/* ── Section wrapper ── */
function Section({ title, children }) {
  return (
    <div style={{ marginBottom: 32 }}>
      <h3 style={{
        fontSize: 14, fontWeight: 700, color: '#e2e8f0', marginBottom: 12,
        letterSpacing: '-0.01em',
      }}>
        {title}
      </h3>
      {children}
    </div>
  );
}

/* ── Dimension card ── */
function DimensionCard({ icon, name, description, values, detail }) {
  return (
    <div style={{
      flex: 1, minWidth: 240, border: '1px solid #1e293b', borderRadius: 8,
      background: '#0f172a', padding: '16px 18px',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
        <span style={{ fontSize: 20 }}>{icon}</span>
        <span style={{ fontSize: 14, fontWeight: 700, color: '#f1f5f9', letterSpacing: '-0.01em' }}>
          {name}
        </span>
      </div>
      <p style={{ fontSize: 13, color: '#94a3b8', lineHeight: 1.5, margin: '0 0 12px' }}>
        {description}
      </p>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 10 }}>
        {values.map((v) => (
          <span key={v.label} style={{
            fontSize: 12, fontFamily: 'monospace', fontWeight: 600,
            color: v.color || '#cbd5e1', background: v.bg || '#1e293b',
            border: `1px solid ${v.borderColor || '#334155'}`, borderRadius: 4, padding: '3px 8px',
          }}>
            {v.label}
          </span>
        ))}
      </div>
      <p style={{ fontSize: 12, color: '#64748b', lineHeight: 1.5, margin: 0, fontFamily: 'monospace' }}>
        {detail}
      </p>
    </div>
  );
}

/* ── Session timeline ── */
function SessionTimeline() {
  const sessions = [
    { name: 'ASIA', start: 0, end: 7, color: '#f87171', bg: '#7f1d1d30', hours: '00:00 - 07:00' },
    { name: 'LONDON', start: 8, end: 12, color: '#60a5fa', bg: '#1e3a5f30', hours: '08:00 - 12:00' },
    { name: 'NY', start: 13, end: 20, color: '#34d399', bg: '#052e1630', hours: '13:00 - 20:00' },
    { name: 'OTHER', start: 21, end: 23, color: '#94a3b8', bg: '#1e293b30', hours: '21:00 - 23:00' },
  ];

  return (
    <div style={{ marginTop: 8 }}>
      {/* Timeline bar */}
      <div style={{
        display: 'flex', height: 32, borderRadius: 6, overflow: 'hidden',
        border: '1px solid #1e293b',
      }}>
        {sessions.map((s) => {
          const width = ((s.end - s.start + 1) / 24) * 100;
          return (
            <div key={s.name} style={{
              width: `${width}%`, background: s.bg, display: 'flex',
              alignItems: 'center', justifyContent: 'center',
              borderRight: '1px solid #1e293b',
            }}>
              <span style={{ fontSize: 11, fontWeight: 700, color: s.color, fontFamily: 'monospace', letterSpacing: '0.04em' }}>
                {s.name}
              </span>
            </div>
          );
        })}
      </div>

      {/* Hour labels */}
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4, padding: '0 2px' }}>
        {[0, 6, 12, 18, 23].map((h) => (
          <span key={h} style={{ fontSize: 10, color: '#475569', fontFamily: 'monospace' }}>
            {String(h).padStart(2, '0')}:00
          </span>
        ))}
      </div>

      {/* Legend */}
      <div style={{ display: 'flex', gap: 16, marginTop: 8, flexWrap: 'wrap' }}>
        {sessions.map((s) => (
          <div key={s.name} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div style={{ width: 10, height: 10, borderRadius: 2, background: s.color, opacity: 0.7 }} />
            <span style={{ fontSize: 12, color: '#94a3b8', fontFamily: 'monospace' }}>
              {s.name} <span style={{ color: '#64748b' }}>{s.hours} UTC</span>
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── Trend visual ── */
function TrendVisual() {
  const trendTypes = [
    {
      name: 'UPTREND', color: '#34d399', symbol: '\u2197',
      desc: 'SMA(50) slope > +0.05%',
      path: 'M 0 28 Q 15 24, 30 18 Q 45 12, 60 8 Q 75 4, 90 2',
    },
    {
      name: 'DOWNTREND', color: '#f87171', symbol: '\u2198',
      desc: 'SMA(50) slope < -0.05%',
      path: 'M 0 4 Q 15 8, 30 14 Q 45 20, 60 24 Q 75 26, 90 30',
    },
    {
      name: 'CONSOLIDATION', color: '#fbbf24', symbol: '\u2194',
      desc: 'Slope between -0.05% and +0.05%',
      path: 'M 0 16 Q 10 12, 20 16 Q 30 20, 40 16 Q 50 12, 60 16 Q 70 20, 80 16 Q 90 14, 100 16',
    },
  ];

  return (
    <div style={{ display: 'flex', gap: 10, marginTop: 8, flexWrap: 'wrap' }}>
      {trendTypes.map((t) => (
        <div key={t.name} style={{
          flex: 1, minWidth: 160, border: '1px solid #1e293b', borderRadius: 6,
          background: '#0f172a', padding: '12px 14px', textAlign: 'center',
        }}>
          <svg width="100%" height="36" viewBox="0 0 100 32" preserveAspectRatio="none"
            style={{ marginBottom: 6, opacity: 0.8 }}>
            <path d={t.path} fill="none" stroke={t.color} strokeWidth="2" />
          </svg>
          <div style={{ fontSize: 13, fontWeight: 700, color: t.color, fontFamily: 'monospace', marginBottom: 2 }}>
            {t.symbol} {t.name}
          </div>
          <div style={{ fontSize: 11, color: '#64748b', fontFamily: 'monospace' }}>
            {t.desc}
          </div>
        </div>
      ))}
    </div>
  );
}

/* ── Volatility visual ── */
function VolatilityVisual() {
  return (
    <div style={{ display: 'flex', gap: 10, marginTop: 8 }}>
      <div style={{
        flex: 1, border: '1px solid #1e293b', borderRadius: 6,
        background: '#0f172a', padding: '12px 14px', textAlign: 'center',
      }}>
        <svg width="100%" height="36" viewBox="0 0 100 32" preserveAspectRatio="none" style={{ marginBottom: 6, opacity: 0.8 }}>
          <path d="M 0 16 Q 5 2, 10 16 Q 15 30, 20 16 Q 25 0, 30 16 Q 35 32, 40 16 Q 45 2, 50 16 Q 55 30, 60 16 Q 65 2, 70 16 Q 75 30, 80 16" fill="none" stroke="#f87171" strokeWidth="2" />
        </svg>
        <div style={{ fontSize: 13, fontWeight: 700, color: '#f87171', fontFamily: 'monospace', marginBottom: 2 }}>
          HIGH_VOL
        </div>
        <div style={{ fontSize: 11, color: '#64748b', fontFamily: 'monospace' }}>
          ATR(24) {'>'} SMA(20) of ATR
        </div>
      </div>
      <div style={{
        flex: 1, border: '1px solid #1e293b', borderRadius: 6,
        background: '#0f172a', padding: '12px 14px', textAlign: 'center',
      }}>
        <svg width="100%" height="36" viewBox="0 0 100 32" preserveAspectRatio="none" style={{ marginBottom: 6, opacity: 0.8 }}>
          <path d="M 0 16 Q 10 12, 20 16 Q 30 20, 40 16 Q 50 14, 60 16 Q 70 18, 80 16 Q 90 15, 100 16" fill="none" stroke="#60a5fa" strokeWidth="2" />
        </svg>
        <div style={{ fontSize: 13, fontWeight: 700, color: '#60a5fa', fontFamily: 'monospace', marginBottom: 2 }}>
          LOW_VOL
        </div>
        <div style={{ fontSize: 11, color: '#64748b', fontFamily: 'monospace' }}>
          ATR(24) {'<='} SMA(20) of ATR
        </div>
      </div>
    </div>
  );
}

/* ── 3D Bucket matrix (routing table) ── */
function RoutingMatrix() {
  // Build lookup from routing table
  const lookup = {};
  routingTable.forEach((r) => {
    lookup[`${r.session}-${r.trend}-${r.vol}`] = r.champion;
  });

  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, fontFamily: 'monospace' }}>
        <thead>
          <tr>
            <th style={{ padding: '8px 10px', textAlign: 'left', color: '#64748b', borderBottom: '1px solid #1e293b', fontWeight: 600, fontSize: 11, letterSpacing: '0.06em' }}>
              SESSION
            </th>
            <th style={{ padding: '8px 10px', textAlign: 'left', color: '#64748b', borderBottom: '1px solid #1e293b', fontWeight: 600, fontSize: 11, letterSpacing: '0.06em' }}>
              TREND
            </th>
            {VOLS.map((v) => (
              <th key={v} style={{ padding: '8px 10px', textAlign: 'center', color: '#64748b', borderBottom: '1px solid #1e293b', fontWeight: 600, fontSize: 11, letterSpacing: '0.06em' }}>
                {v}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {SESSIONS.map((session, si) =>
            TRENDS.map((trend, ti) => {
              const isFirst = ti === 0;
              const sessionColors = {
                ASIA: '#f87171', LONDON: '#60a5fa', NY: '#34d399', OTHER: '#94a3b8',
              };
              return (
                <tr key={`${session}-${trend}`} style={{
                  background: (si * 3 + ti) % 2 === 0 ? '#0f172a' : '#0f172a80',
                  borderTop: isFirst && si > 0 ? '1px solid #334155' : undefined,
                }}>
                  {isFirst ? (
                    <td rowSpan={3} style={{
                      padding: '6px 10px', fontWeight: 700, verticalAlign: 'middle',
                      color: sessionColors[session], borderRight: '1px solid #1e293b',
                    }}>
                      {session}
                    </td>
                  ) : null}
                  <td style={{
                    padding: '6px 10px', color: '#94a3b8',
                    borderRight: '1px solid #1e293b',
                  }}>
                    {trend}
                  </td>
                  {VOLS.map((vol) => {
                    const champ = lookup[`${session}-${trend}-${vol}`];
                    const m = champ ? FAMILY_META[champ] : null;
                    return (
                      <td key={vol} style={{ padding: '6px 10px', textAlign: 'center' }}>
                        {m ? (
                          <span style={{
                            color: m.color, background: m.bg, border: `1px solid ${m.color}30`,
                            borderRadius: 3, padding: '2px 8px', fontSize: 11, fontWeight: 600,
                          }}>
                            {m.label}
                          </span>
                        ) : (
                          <span style={{ color: '#334155' }}>-</span>
                        )}
                      </td>
                    );
                  })}
                </tr>
              );
            })
          )}
        </tbody>
      </table>
    </div>
  );
}

/* ── Bucket count / formula ── */
function BucketFormula() {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10,
      padding: '14px 16px', background: '#0f172a', border: '1px solid #1e293b',
      borderRadius: 8, fontFamily: 'monospace', flexWrap: 'wrap',
    }}>
      <span style={{ fontSize: 14, fontWeight: 700, color: '#f87171' }}>4</span>
      <span style={{ fontSize: 12, color: '#64748b' }}>sessions</span>
      <span style={{ fontSize: 14, color: '#334155' }}>&times;</span>
      <span style={{ fontSize: 14, fontWeight: 700, color: '#fbbf24' }}>3</span>
      <span style={{ fontSize: 12, color: '#64748b' }}>trends</span>
      <span style={{ fontSize: 14, color: '#334155' }}>&times;</span>
      <span style={{ fontSize: 14, fontWeight: 700, color: '#60a5fa' }}>2</span>
      <span style={{ fontSize: 12, color: '#64748b' }}>volatility</span>
      <span style={{ fontSize: 14, color: '#334155' }}>=</span>
      <span style={{ fontSize: 18, fontWeight: 900, color: '#e2e8f0' }}>24</span>
      <span style={{ fontSize: 12, color: '#64748b' }}>regime buckets</span>
    </div>
  );
}

/* ── Filtered buckets per hybrid ── */
function FilteredBucketsTable() {
  const sorted = [...hybrids].sort((a, b) => a.rank - b.rank).filter((h) => h.filteredBuckets?.length > 0);

  return (
    <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
      {sorted.map((h) => {
        const n = h.filteredBuckets?.length || 0;
        return (
          <div key={h.name} style={{
            flex: 1, minWidth: 200, border: '1px solid #1e293b', borderRadius: 6,
            background: '#0f172a', padding: '12px 14px',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
              <span style={{ fontSize: 13, fontWeight: 700, color: '#e2e8f0', fontFamily: 'monospace' }}>
                #{h.rank} {h.displayName}
              </span>
              <span style={{
                fontSize: 11, fontWeight: 700, fontFamily: 'monospace',
                color: n === 0 ? '#34d399' : '#fbbf24',
                background: n === 0 ? '#052e1640' : '#3b2a0a40',
                border: `1px solid ${n === 0 ? '#059669' : '#92400e'}40`,
                borderRadius: 3, padding: '1px 6px',
              }}>
                {n === 0 ? 'NO FILTER' : `${n} FILTERED`}
              </span>
            </div>
            {n > 0 ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                {h.filteredBuckets.map((b, i) => (
                  <span key={i} style={{
                    fontSize: 11, fontFamily: 'monospace', color: '#f87171',
                    background: '#7f1d1d15', border: '1px solid #7f1d1d30',
                    borderRadius: 3, padding: '2px 6px',
                  }}>
                    {b}
                  </span>
                ))}
              </div>
            ) : (
              <span style={{ fontSize: 12, color: '#475569', fontFamily: 'monospace' }}>
                All 2D regime buckets are tradable
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}

/* ── How filtering works ── */
function FilteringProcess() {
  const steps = [
    {
      num: '1',
      title: 'Compute 2D diagnostics',
      desc: 'For each hybrid, calculate Sharpe ratio and trade count across all 2D regime pairs (session-trend, session-vol, trend-vol).',
    },
    {
      num: '2',
      title: 'Check sufficient evidence',
      desc: 'A bucket needs at least 30 trades to be considered. Buckets with fewer trades are ignored (not filtered).',
    },
    {
      num: '3',
      title: 'Filter negative Sharpe buckets',
      desc: 'Any 2D bucket with sufficient evidence but negative Sharpe is marked as non-tradable. Signals in matching bars are zeroed out.',
    },
    {
      num: '4',
      title: '3D gate check',
      desc: 'A 3D bucket (session, trend, vol) is only tradable if ALL three of its parent 2D buckets are tradable.',
    },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
      {steps.map((s, i) => (
        <div key={s.num} style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
          {/* Vertical line + number */}
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', minWidth: 28, paddingTop: 2 }}>
            <div style={{
              width: 24, height: 24, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center',
              background: '#1e293b', border: '1px solid #334155', fontSize: 12, fontWeight: 700, color: '#e2e8f0', fontFamily: 'monospace',
            }}>
              {s.num}
            </div>
            {i < steps.length - 1 && (
              <div style={{ width: 1, height: 24, background: '#334155' }} />
            )}
          </div>
          {/* Text */}
          <div style={{ paddingBottom: i < steps.length - 1 ? 8 : 0, flex: 1 }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: '#e2e8f0', marginBottom: 2 }}>
              {s.title}
            </div>
            <div style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.5 }}>
              {s.desc}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

/* ── Main ── */
export default function RegimeExplainer() {
  return (
    <div style={{
      background: '#020617', minHeight: '100%', padding: '16px 20px',
      fontFamily: 'ui-sans-serif, system-ui, sans-serif', color: '#e2e8f0',
    }}>
      {/* Header */}
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ fontSize: 16, fontWeight: 700, color: '#f8fafc', margin: 0, letterSpacing: '-0.01em' }}>
          Regime Classification
        </h2>
        <p style={{ fontSize: 13, color: '#64748b', marginTop: 4, lineHeight: 1.5 }}>
          Every bar in the dataset is classified across 3 independent dimensions.
          The pipeline uses these regimes to evaluate strategy performance in different market conditions
          and filter out unprofitable regime combinations.
        </p>
      </div>

      {/* Three dimensions */}
      <Section title="The 3 Dimensions">
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
          <DimensionCard
            icon={<span style={{ filter: 'none' }}>&#x1F552;</span>}
            name="Session (Market Hours)"
            description="Which global trading session is active, based on UTC hour of the candle."
            values={[
              { label: 'ASIA', color: '#f87171', bg: '#7f1d1d20', borderColor: '#7f1d1d40' },
              { label: 'LONDON', color: '#60a5fa', bg: '#1e3a5f20', borderColor: '#1e3a5f40' },
              { label: 'NY', color: '#34d399', bg: '#052e1620', borderColor: '#05292040' },
              { label: 'OTHER', color: '#94a3b8', bg: '#1e293b', borderColor: '#334155' },
            ]}
            detail="Hour ranges: ASIA 00-07, LONDON 08-12, NY 13-20, OTHER 21-23 UTC"
          />
          <DimensionCard
            icon={<span style={{ filter: 'none' }}>&#x1F4C8;</span>}
            name="Trend Regime"
            description="Direction of the market based on 50-period SMA slope over 3 bars."
            values={[
              { label: 'UPTREND', color: '#34d399', bg: '#052e1620', borderColor: '#05292040' },
              { label: 'DOWNTREND', color: '#f87171', bg: '#7f1d1d20', borderColor: '#7f1d1d40' },
              { label: 'CONSOLIDATION', color: '#fbbf24', bg: '#3b2a0a20', borderColor: '#3b2a0a40' },
            ]}
            detail={`Threshold: slope > +${config.tbmLoss * 0.05}% = UPTREND, < -0.05% = DOWNTREND`}
          />
          <DimensionCard
            icon={<span style={{ filter: 'none' }}>&#x26A1;</span>}
            name="Volatility Regime"
            description="Whether current volatility is above or below its recent average."
            values={[
              { label: 'HIGH_VOL', color: '#f87171', bg: '#7f1d1d20', borderColor: '#7f1d1d40' },
              { label: 'LOW_VOL', color: '#60a5fa', bg: '#1e3a5f20', borderColor: '#1e3a5f40' },
            ]}
            detail="ATR(24) compared to its 20-period SMA. Above = high, below = low."
          />
        </div>
      </Section>

      {/* Visuals */}
      <Section title="Session Timeline (UTC)">
        <SessionTimeline />
      </Section>

      <Section title="Trend Detection">
        <p style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.5, marginBottom: 8 }}>
          Computed from the 3-bar percentage change of SMA(50). If the slope exceeds the threshold in either
          direction, the market is trending. Otherwise it is consolidating.
        </p>
        <TrendVisual />
      </Section>

      <Section title="Volatility Detection">
        <p style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.5, marginBottom: 8 }}>
          ATR (Average True Range) over 24 bars captures recent price movement amplitude. When ATR is above
          its own 20-period moving average, volatility is elevated.
        </p>
        <VolatilityVisual />
      </Section>

      {/* 3D buckets */}
      <Section title="3D Regime Buckets">
        <p style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.5, marginBottom: 12 }}>
          Combining all three dimensions produces 24 unique market conditions. Each bar belongs to exactly
          one bucket. Strategy performance is evaluated separately in every bucket.
        </p>
        <BucketFormula />
      </Section>

      {/* Routing table */}
      <Section title="Regime Router - Champion Assignment">
        <p style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.5, marginBottom: 12 }}>
          The Regime Router hybrid assigns the best-performing champion to each 3D bucket based on
          historical Sharpe ratio. Each regime combination routes to the champion that performed best
          in that specific market condition.
        </p>
        <div style={{ border: '1px solid #1e293b', borderRadius: 8, overflow: 'hidden' }}>
          <RoutingMatrix />
        </div>
      </Section>

      {/* Filtering process */}
      <Section title="How Regime Filtering Works">
        <p style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.5, marginBottom: 12 }}>
          After backtesting, the Scientist agent identifies regime combinations where a hybrid
          loses money and disables trading in those conditions.
        </p>
        <FilteringProcess />
      </Section>

      {/* Filtered buckets */}
      <Section title="Filtered Buckets per Hybrid">
        <p style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.5, marginBottom: 12 }}>
          2D regime pairs where the hybrid showed negative Sharpe with sufficient evidence.
          Signals in these regime combinations are zeroed out (no trades taken).
        </p>
        <FilteredBucketsTable />
      </Section>
    </div>
  );
}
