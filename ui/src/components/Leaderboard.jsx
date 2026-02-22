import { useState } from 'react';
import { hybrids } from '../data.js';

// --- Formatters ---

function fmtFitness(v) {
  return v.toFixed(3);
}

function fmtSharpe(v) {
  return v.toFixed(2);
}

function fmtWinRate(v) {
  return v.toFixed(1) + '%';
}

function fmtTrades(v) {
  return v.toLocaleString();
}

function fmtReturn(v) {
  return (v >= 0 ? '+' : '') + v.toFixed(1) + '%';
}

function fmtDD(v) {
  return v.toFixed(1) + '%';
}

function fmtHold(v) {
  return v.toFixed(1) + 'h';
}

// --- Rank accent colors ---

const RANK_CONFIG = {
  1: {
    label: '1st',
    border: '#10b981',        // emerald-500
    shadow: '0 0 24px 4px rgba(16,185,129,0.35)',
    glow: '0 0 48px 8px rgba(16,185,129,0.18)',
    bg: 'linear-gradient(160deg, #0f2a1e 0%, #0f172a 100%)',
    accent: '#10b981',
    badgeBg: '#065f46',
    badgeText: '#6ee7b7',
    rankColor: '#10b981',
    dimText: '#6ee7b7',
    tableAccent: '#10b981',
  },
  2: {
    label: '2nd',
    border: '#94a3b8',        // slate-400
    shadow: '0 0 16px 2px rgba(148,163,184,0.22)',
    glow: 'none',
    bg: 'linear-gradient(160deg, #1a2333 0%, #0f172a 100%)',
    accent: '#94a3b8',
    badgeBg: '#1e293b',
    badgeText: '#cbd5e1',
    rankColor: '#94a3b8',
    dimText: '#94a3b8',
    tableAccent: '#3b82f6',
  },
  3: {
    label: '3rd',
    border: '#78350f',        // amber-900 / bronze tone
    shadow: '0 0 10px 1px rgba(180,120,60,0.15)',
    glow: 'none',
    bg: 'linear-gradient(160deg, #1c1610 0%, #0f172a 100%)',
    accent: '#b45309',
    badgeBg: '#292019',
    badgeText: '#d97706',
    rankColor: '#b45309',
    dimText: '#92400e',
    tableAccent: '#64748b',
  },
};

// Ordered for podium display: left=2nd, center=1st, right=3rd
const PODIUM_ORDER = [2, 1, 3];

// Heights for podium blocks (px)
const PODIUM_HEIGHTS = { 1: 260, 2: 210, 3: 180 };

// Column definitions for the table
const COLUMNS = [
  { key: 'rank',            label: 'Rank',          align: 'center' },
  { key: 'displayName',     label: 'Strategy',      align: 'left'   },
  { key: 'fitness',         label: 'Fitness',       align: 'right'  },
  { key: 'annSharpe',       label: 'Ann. Sharpe',   align: 'right'  },
  { key: 'winRate',         label: 'Win Rate',      align: 'right'  },
  { key: 'trades',          label: 'Trades',        align: 'right'  },
  { key: 'totalReturn',     label: 'Return',        align: 'right'  },
  { key: 'maxDrawdown',     label: 'Max DD',        align: 'right'  },
  { key: 'maxConsecLosses', label: 'Consec Losses', align: 'right'  },
  { key: 'avgHoldHours',    label: 'Avg Hold',      align: 'right'  },
];

// Which columns have "higher is better" (true) vs "lower is better" (false)
const HIGHER_BETTER = {
  fitness:         true,
  annSharpe:       true,
  winRate:         true,
  trades:          true,
  totalReturn:     true,
  maxDrawdown:     false,  // less negative = better
  maxConsecLosses: false,
  avgHoldHours:    null,   // no highlight for this
};

// Compute best value per column across hybrids
function computeBest(data) {
  const best = {};
  for (const key of Object.keys(HIGHER_BETTER)) {
    if (HIGHER_BETTER[key] === null) continue;
    const vals = data.map((h) => h[key]);
    best[key] = HIGHER_BETTER[key] ? Math.max(...vals) : Math.min(...vals);
  }
  return best;
}

// --- Podium Card ---

function PodiumCard({ hybrid, isCenter }) {
  const cfg = RANK_CONFIG[hybrid.rank];
  const height = PODIUM_HEIGHTS[hybrid.rank];

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        width: isCenter ? 240 : 200,
        position: 'relative',
      }}
    >
      {/* Card */}
      <div
        style={{
          width: '100%',
          background: cfg.bg,
          border: `1.5px solid ${cfg.border}`,
          borderRadius: 12,
          boxShadow: cfg.shadow,
          padding: isCenter ? '24px 20px 20px' : '18px 16px 16px',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: isCenter ? 10 : 8,
          position: 'relative',
          marginBottom: 8,
        }}
      >
        {/* Rank badge */}
        <div
          style={{
            position: 'absolute',
            top: -14,
            left: '50%',
            transform: 'translateX(-50%)',
            background: cfg.badgeBg,
            border: `1px solid ${cfg.border}`,
            borderRadius: 20,
            padding: '3px 14px',
            fontSize: 12,
            fontWeight: 700,
            color: cfg.badgeText,
            letterSpacing: '0.08em',
            fontFamily: 'monospace',
            whiteSpace: 'nowrap',
          }}
        >
          {cfg.label}
        </div>

        {/* Rank number */}
        <div
          style={{
            fontSize: isCenter ? 56 : 44,
            fontWeight: 900,
            color: cfg.rankColor,
            lineHeight: 1,
            fontFamily: 'monospace',
            letterSpacing: '-0.03em',
            textShadow: isCenter ? `0 0 20px ${cfg.rankColor}80` : 'none',
            marginTop: 8,
          }}
        >
          {hybrid.rank}
        </div>

        {/* Strategy name */}
        <div
          style={{
            fontSize: isCenter ? 15 : 13,
            fontWeight: 600,
            color: '#f1f5f9',
            textAlign: 'center',
            letterSpacing: '0.02em',
          }}
        >
          {hybrid.displayName}
        </div>

        {/* Divider */}
        <div
          style={{
            width: '80%',
            height: 1,
            background: `linear-gradient(90deg, transparent, ${cfg.border}60, transparent)`,
            margin: '2px 0',
          }}
        />

        {/* Fitness */}
        <div style={{ textAlign: 'center' }}>
          <div
            style={{
              fontSize: 11,
              color: '#64748b',
              textTransform: 'uppercase',
              letterSpacing: '0.1em',
              marginBottom: 2,
            }}
          >
            Fitness
          </div>
          <div
            style={{
              fontSize: isCenter ? 26 : 20,
              fontWeight: 800,
              color: cfg.accent,
              fontFamily: 'monospace',
            }}
          >
            {fmtFitness(hybrid.fitness)}
          </div>
        </div>

        {/* Secondary stats */}
        <div
          style={{
            display: 'flex',
            gap: 16,
            marginTop: 4,
          }}
        >
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 10, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
              Ann. Sharpe
            </div>
            <div style={{ fontSize: 14, fontWeight: 700, color: '#cbd5e1', fontFamily: 'monospace' }}>
              {fmtSharpe(hybrid.annSharpe)}
            </div>
          </div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 10, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
              Return
            </div>
            <div
              style={{
                fontSize: 14,
                fontWeight: 700,
                fontFamily: 'monospace',
                color: hybrid.totalReturn >= 0 ? '#10b981' : '#f43f5e',
              }}
            >
              {fmtReturn(hybrid.totalReturn)}
            </div>
          </div>
        </div>
      </div>

      {/* Podium block base */}
      <div
        style={{
          width: '90%',
          height: isCenter ? 56 : hybrid.rank === 2 ? 36 : 20,
          background: `linear-gradient(180deg, ${cfg.border}30 0%, ${cfg.border}10 100%)`,
          border: `1px solid ${cfg.border}40`,
          borderTop: 'none',
          borderRadius: '0 0 8px 8px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <div
          style={{
            fontSize: 10,
            color: cfg.dimText,
            fontFamily: 'monospace',
            opacity: 0.7,
            letterSpacing: '0.1em',
            textTransform: 'uppercase',
          }}
        >
          {fmtTrades(hybrid.trades)} trades
        </div>
      </div>
    </div>
  );
}

// --- Table cell renderer ---

function CellValue({ col, value, isBest }) {
  if (col === 'rank') {
    const cfg = RANK_CONFIG[value];
    return (
      <span
        style={{
          fontFamily: 'monospace',
          fontWeight: 700,
          color: cfg.badgeText,
          background: cfg.badgeBg,
          border: `1px solid ${cfg.border}`,
          borderRadius: 6,
          padding: '2px 10px',
          fontSize: 12,
          letterSpacing: '0.06em',
        }}
      >
        {cfg.label}
      </span>
    );
  }

  if (col === 'displayName') {
    return (
      <span style={{ color: '#f1f5f9', fontWeight: 600, fontSize: 14 }}>
        {value}
      </span>
    );
  }

  let formatted;
  let color = '#cbd5e1';

  if (col === 'fitness') {
    formatted = fmtFitness(value);
  } else if (col === 'annSharpe') {
    formatted = fmtSharpe(value);
    color = value >= 0 ? '#cbd5e1' : '#f43f5e';
  } else if (col === 'winRate') {
    formatted = fmtWinRate(value);
  } else if (col === 'trades') {
    formatted = fmtTrades(value);
  } else if (col === 'totalReturn') {
    formatted = fmtReturn(value);
    color = value >= 0 ? '#10b981' : '#f43f5e';
  } else if (col === 'maxDrawdown') {
    formatted = fmtDD(value);
    color = '#f43f5e';
  } else if (col === 'maxConsecLosses') {
    formatted = String(value);
  } else if (col === 'avgHoldHours') {
    formatted = fmtHold(value);
  } else {
    formatted = String(value);
  }

  return (
    <span
      style={{
        fontFamily: 'monospace',
        fontSize: 13,
        fontWeight: isBest ? 700 : 400,
        color: isBest && col !== 'totalReturn' && col !== 'maxDrawdown' && col !== 'annSharpe'
          ? '#f1f5f9'
          : color,
        textShadow: isBest ? `0 0 8px ${color}60` : 'none',
        position: 'relative',
      }}
    >
      {formatted}
      {isBest && (
        <span
          style={{
            display: 'inline-block',
            width: 5,
            height: 5,
            borderRadius: '50%',
            background: '#10b981',
            marginLeft: 4,
            verticalAlign: 'middle',
            boxShadow: '0 0 4px #10b981',
            position: 'relative',
            top: -1,
          }}
        />
      )}
    </span>
  );
}

// --- Expandable Strategy Card ---

function StrategyCard({ hybrid }) {
  const [open, setOpen] = useState(false);
  const cfg = RANK_CONFIG[hybrid.rank];

  const details = {
    1: {
      mechanism: '3/4 champions must agree',
      detail: 'Requires consensus from at least 3 of the 4 champion strategies before firing a signal. Direction is determined by majority vote. Conservative filter eliminates noise.',
      stats: [
        { label: 'Agreement threshold', value: '3 of 4 champions' },
        { label: 'Long signals', value: fmtTrades(hybrid.longs) },
        { label: 'Short signals', value: fmtTrades(hybrid.shorts) },
        { label: 'Avg hold time', value: fmtHold(hybrid.avgHoldHours) },
      ],
    },
    2: {
      mechanism: '24-entry routing table, fallback: volume',
      detail: 'Routes each regime bucket (volatility x trend x volume) to the champion with the highest 3D Sharpe in that regime. Falls back to the volume champion when no route is defined.',
      stats: [
        { label: 'Routing table entries', value: '24 regime buckets' },
        { label: 'Fallback champion', value: 'Volume' },
        { label: 'Long signals', value: fmtTrades(hybrid.longs) },
        { label: 'Short signals', value: fmtTrades(hybrid.shorts) },
      ],
    },
    3: {
      mechanism: 'Fitness-weighted signal sum',
      detail: 'Sums the signals from all 4 champions weighted by their fitness scores. Takes np.sign of the weighted sum as the final direction. Higher-fitness champions exert more influence.',
      stats: [
        { label: 'Weighting method', value: 'Fitness-proportional' },
        { label: 'Signal aggregation', value: 'np.sign(weighted sum)' },
        { label: 'Long signals', value: fmtTrades(hybrid.longs) },
        { label: 'Short signals', value: fmtTrades(hybrid.shorts) },
      ],
    },
  };

  const d = details[hybrid.rank];

  return (
    <div
      style={{
        background: '#0f172a',
        border: `1px solid ${open ? cfg.border : '#1e293b'}`,
        borderLeft: `3px solid ${cfg.tableAccent}`,
        borderRadius: 10,
        overflow: 'hidden',
        transition: 'border-color 0.2s',
      }}
    >
      {/* Header - always visible */}
      <button
        onClick={() => setOpen(!open)}
        style={{
          width: '100%',
          background: 'transparent',
          border: 'none',
          cursor: 'pointer',
          padding: '14px 20px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 16,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          {/* Rank badge */}
          <span
            style={{
              fontFamily: 'monospace',
              fontWeight: 700,
              color: cfg.badgeText,
              background: cfg.badgeBg,
              border: `1px solid ${cfg.border}`,
              borderRadius: 6,
              padding: '2px 10px',
              fontSize: 12,
              letterSpacing: '0.06em',
              whiteSpace: 'nowrap',
            }}
          >
            {cfg.label}
          </span>
          <div style={{ textAlign: 'left' }}>
            <div style={{ fontSize: 15, fontWeight: 700, color: '#f1f5f9' }}>
              {hybrid.displayName}
            </div>
            <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>
              {d.mechanism}
            </div>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 24 }}>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: 11, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
              Fitness
            </div>
            <div
              style={{
                fontSize: 16,
                fontWeight: 700,
                color: cfg.accent,
                fontFamily: 'monospace',
              }}
            >
              {fmtFitness(hybrid.fitness)}
            </div>
          </div>
          <div
            style={{
              color: '#475569',
              fontSize: 18,
              transform: open ? 'rotate(180deg)' : 'rotate(0deg)',
              transition: 'transform 0.2s',
              userSelect: 'none',
            }}
          >
            v
          </div>
        </div>
      </button>

      {/* Expandable body */}
      {open && (
        <div
          style={{
            padding: '0 20px 20px',
            borderTop: `1px solid ${cfg.border}30`,
          }}
        >
          <p
            style={{
              fontSize: 13,
              color: '#94a3b8',
              lineHeight: 1.6,
              margin: '14px 0 18px',
            }}
          >
            {d.detail}
          </p>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
              gap: 12,
            }}
          >
            {d.stats.map((s) => (
              <div
                key={s.label}
                style={{
                  background: '#1e293b',
                  borderRadius: 8,
                  padding: '10px 14px',
                  border: `1px solid #334155`,
                }}
              >
                <div
                  style={{
                    fontSize: 10,
                    color: '#64748b',
                    textTransform: 'uppercase',
                    letterSpacing: '0.1em',
                    marginBottom: 4,
                  }}
                >
                  {s.label}
                </div>
                <div
                  style={{
                    fontSize: 14,
                    fontWeight: 700,
                    color: '#e2e8f0',
                    fontFamily: 'monospace',
                  }}
                >
                  {s.value}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// --- Main Component ---

export default function Leaderboard() {
  const [hoveredRow, setHoveredRow] = useState(null);

  const sorted = [...hybrids].sort((a, b) => a.rank - b.rank);
  const best = computeBest(sorted);

  const podiumItems = PODIUM_ORDER.map((rank) =>
    sorted.find((h) => h.rank === rank)
  );

  return (
    <div
      style={{
        background: '#020617',
        minHeight: '100%',
        padding: '32px 24px 48px',
        color: '#f1f5f9',
      }}
    >
      {/* Header */}
      <div style={{ marginBottom: 40 }}>
        <div
          style={{
            fontSize: 11,
            color: '#475569',
            textTransform: 'uppercase',
            letterSpacing: '0.15em',
            marginBottom: 6,
            fontFamily: 'monospace',
          }}
        >
          Tab 4 - Strategy Rankings
        </div>
        <h2
          style={{
            fontSize: 28,
            fontWeight: 800,
            color: '#f8fafc',
            margin: 0,
            letterSpacing: '-0.02em',
          }}
        >
          Hybrid Leaderboard
        </h2>
        <p style={{ fontSize: 13, color: '#64748b', marginTop: 6 }}>
          Competitive ranking of the 3 hybrid strategies by fitness score
        </p>
      </div>

      {/* ---- PODIUM SECTION ---- */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'flex-end',
          gap: 20,
          marginBottom: 48,
          padding: '40px 0 0',
          position: 'relative',
        }}
      >
        {/* Subtle glow behind center podium */}
        <div
          style={{
            position: 'absolute',
            top: '30%',
            left: '50%',
            transform: 'translateX(-50%)',
            width: 300,
            height: 200,
            background: 'radial-gradient(ellipse, rgba(16,185,129,0.08) 0%, transparent 70%)',
            pointerEvents: 'none',
          }}
        />

        {podiumItems.map((hybrid) => (
          <PodiumCard
            key={hybrid.rank}
            hybrid={hybrid}
            isCenter={hybrid.rank === 1}
          />
        ))}
      </div>

      {/* ---- COMPARISON TABLE ---- */}
      <div style={{ marginBottom: 40 }}>
        <h3
          style={{
            fontSize: 14,
            fontWeight: 700,
            color: '#64748b',
            textTransform: 'uppercase',
            letterSpacing: '0.12em',
            marginBottom: 16,
          }}
        >
          Detailed Comparison
        </h3>

        <div
          style={{
            background: '#0f172a',
            border: '1px solid #1e293b',
            borderRadius: 12,
            overflow: 'hidden',
          }}
        >
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ background: '#0a0f1e' }}>
                {COLUMNS.map((col) => (
                  <th
                    key={col.key}
                    style={{
                      padding: '12px 16px',
                      fontSize: 10,
                      fontWeight: 600,
                      color: '#475569',
                      textTransform: 'uppercase',
                      letterSpacing: '0.1em',
                      textAlign: col.align,
                      borderBottom: '1px solid #1e293b',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {col.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sorted.map((hybrid, rowIdx) => {
                const cfg = RANK_CONFIG[hybrid.rank];
                const isHovered = hoveredRow === hybrid.rank;

                return (
                  <tr
                    key={hybrid.rank}
                    onMouseEnter={() => setHoveredRow(hybrid.rank)}
                    onMouseLeave={() => setHoveredRow(null)}
                    style={{
                      background: isHovered ? '#1e293b80' : 'transparent',
                      borderLeft: `3px solid ${cfg.tableAccent}`,
                      transition: 'background 0.15s',
                      cursor: 'default',
                    }}
                  >
                    {COLUMNS.map((col) => {
                      const value = col.key === 'rank' ? hybrid.rank : hybrid[col.key];
                      const isBest =
                        HIGHER_BETTER[col.key] !== null &&
                        HIGHER_BETTER[col.key] !== undefined &&
                        value === best[col.key];

                      return (
                        <td
                          key={col.key}
                          style={{
                            padding: '13px 16px',
                            textAlign: col.align,
                            borderBottom: rowIdx < sorted.length - 1 ? '1px solid #1e293b' : 'none',
                            verticalAlign: 'middle',
                          }}
                        >
                          <CellValue col={col.key} value={value} isBest={isBest} />
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Best value legend */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            marginTop: 10,
            paddingLeft: 4,
          }}
        >
          <div
            style={{
              width: 6,
              height: 6,
              borderRadius: '50%',
              background: '#10b981',
              boxShadow: '0 0 4px #10b981',
            }}
          />
          <span style={{ fontSize: 11, color: '#475569' }}>
            Best value in column
          </span>
        </div>
      </div>

      {/* ---- STRATEGY CARDS ---- */}
      <div>
        <h3
          style={{
            fontSize: 14,
            fontWeight: 700,
            color: '#64748b',
            textTransform: 'uppercase',
            letterSpacing: '0.12em',
            marginBottom: 16,
          }}
        >
          Strategy Details
        </h3>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {sorted.map((hybrid) => (
            <StrategyCard key={hybrid.rank} hybrid={hybrid} />
          ))}
        </div>
      </div>
    </div>
  );
}
